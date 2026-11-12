"""sqlib: quantization of additive lattice scorecards.

Core library for the CSDA manuscript. All functions operate on a *cell
representation* of an additive scorecard:

    cell_id : (N, P) int array, the flat index of the active bin of feature j
              for applicant i (each feature owns a disjoint block of cells);
    cvec    : (n_cells,) float array of continuous log-odds contributions;
    b0      : float intercept (not quantized).

The continuous score is f = b0 + cvec[cell_id].sum(1) and the deployed score
at lattice spacing D is fq = b0 + D * n[cell_id].sum(1), n = integer points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LN2 = np.log(2.0)


def delta_from_pdo(pdo: float, grid: float) -> float:
    """Effective log-odds lattice spacing Delta = g ln2 / PDO."""
    return grid * LN2 / pdo


# ---------------------------------------------------------------------------
# Irwin--Hall laws: sums of n iid U(-1/2, 1/2)
# ---------------------------------------------------------------------------
class IrwinHall:
    """Numerically exact cdf of S_n = sum of n iid U(-1/2,1/2), n = 0..nmax.

    Computed by repeated convolution with the unit box on a fine grid;
    S_0 = 0 is a point mass. Accuracy ~1e-6 for the default step.
    """

    def __init__(self, nmax: int, h: float = 1.0 / 1000):
        self.nmax = nmax
        self.h = h
        L = nmax / 2.0 + 1.0
        self.x = np.arange(-L, L + h / 2, h)
        half = int(round(0.5 / h))
        cdfs = [None]
        dens = np.where(np.abs(self.x) < 0.5, 1.0, 0.0)
        dens[np.isclose(np.abs(self.x), 0.5)] = 0.5
        for n in range(1, nmax + 1):
            if n > 1:
                c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * h)])
                up = np.concatenate([c[half:], np.full(half, c[-1])])
                lo = np.concatenate([np.zeros(half), c[:-half]])
                dens = up - lo
            c = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * h)])
            c /= c[-1]
            cdfs.append(c)
        self._cdf = cdfs

    def cdf(self, n, t):
        """P(S_n <= t); n may be an int or an int array broadcastable with t."""
        t = np.asarray(t, dtype=float)
        n = np.asarray(n)
        out = np.empty(np.broadcast(n, t).shape)
        nb = np.broadcast_to(n, out.shape)
        tb = np.broadcast_to(t, out.shape)
        for k in np.unique(nb):
            m = nb == k
            if k == 0:
                out[m] = (tb[m] >= 0).astype(float)
            else:
                out[m] = np.interp(tb[m], self.x, self._cdf[int(k)], left=0.0, right=1.0)
        return out

    def mean_abs(self, n: int) -> float:
        """kappa_n = E|S_n|."""
        if n == 0:
            return 0.0
        c = self._cdf[n]
        x = self.x
        # E|S| = int_0^inf P(|S|>t) dt = 2 int_0^inf (1 - F(t)) dt
        pos = x >= 0
        return float(2.0 * np.trapezoid(1.0 - c[pos], x[pos]))


# ---------------------------------------------------------------------------
# Scorecard construction (WoE binning + logistic regression)
# ---------------------------------------------------------------------------
def _numeric_edges(x, n_bins, min_share):
    xv = x[~np.isnan(x)]
    u = np.unique(xv)
    if len(u) <= n_bins:
        edges = (u[1:] + u[:-1]) / 2.0
    else:
        edges = np.unique(np.quantile(xv, np.linspace(0, 1, n_bins + 1)[1:-1]))
    # merge bins below the minimum share
    while len(edges) > 0:
        b = np.searchsorted(edges, xv, side="right")
        cnt = np.bincount(b, minlength=len(edges) + 1) / len(xv)
        k = int(np.argmin(cnt))
        if cnt[k] >= min_share:
            break
        if k == 0:
            drop = 0
        elif k == len(cnt) - 1:
            drop = len(edges) - 1
        else:
            drop = k - 1 if cnt[k - 1] <= cnt[k + 1] else k
        edges = np.delete(edges, drop)
    return edges


class Binner:
    """Quantile binning for numeric features, category bins for strings;
    missing values get their own bin. Fitted on training data only."""

    def __init__(self, n_bins=6, min_share=0.02):
        self.n_bins = n_bins
        self.min_share = min_share

    def fit(self, X: pd.DataFrame):
        self.spec = []
        for col in X.columns:
            s = X[col]
            if not pd.api.types.is_numeric_dtype(s):
                vc = s.value_counts(normalize=True)
                keep = list(vc[vc >= self.min_share].index)
                self.spec.append(("cat", col, keep))
            else:
                x = s.to_numpy(dtype=float)
                edges = _numeric_edges(x, self.n_bins, self.min_share)
                self.spec.append(("num", col, edges))
        return self

    def transform(self, X: pd.DataFrame):
        """Return (cell_id, sizes): per-feature bin index (local) and #bins."""
        N = len(X)
        B = np.zeros((N, len(self.spec)), dtype=np.int64)
        sizes = []
        for j, (kind, col, par) in enumerate(self.spec):
            s = X[col]
            if kind == "cat":
                m = {v: i for i, v in enumerate(par)}
                B[:, j] = s.map(lambda v: m.get(v, len(par))).to_numpy()
                sizes.append(len(par) + 1)  # last = other/rare
            else:
                x = s.to_numpy(dtype=float)
                b = np.searchsorted(par, x, side="right")
                b[np.isnan(x)] = len(par) + 1
                B[:, j] = b
                sizes.append(len(par) + 2)  # last = missing
        return B, np.array(sizes)


class Scorecard:
    """Additive WoE scorecard: c_{j,b} = beta_j * WoE_{j,b}; f = log-odds of the
    event (default). Features with a non-positive coefficient are dropped
    iteratively, as in industry practice."""

    def __init__(self, n_bins=6, min_share=0.02, C=1e4):
        self.n_bins, self.min_share, self.C = n_bins, min_share, C

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.binner = Binner(self.n_bins, self.min_share).fit(X)
        B, sizes = self.binner.transform(X)
        y = np.asarray(y).astype(int)
        nb, ng = y.sum(), (1 - y).sum()
        woe = []
        for j in range(B.shape[1]):
            k = sizes[j]
            bad = np.bincount(B[:, j], weights=y, minlength=k)
            good = np.bincount(B[:, j], weights=1 - y, minlength=k)
            w = np.log((bad + 0.5) / (nb + 0.5 * k)) - np.log((good + 0.5) / (ng + 0.5 * k))
            woe.append(w)
        keep = list(range(B.shape[1]))
        # drop near-constant features
        keep = [j for j in keep if len(np.unique(B[:, j])) > 1]
        while True:
            W = np.column_stack([woe[j][B[:, j]] for j in keep])
            lr = LogisticRegression(C=self.C, max_iter=5000).fit(W, y)
            beta = lr.coef_.ravel()
            if (beta > 0).all() or len(keep) == 1:
                break
            keep = [j for j, b in zip(keep, beta) if b > 0]
        self.keep = keep
        self.beta = beta
        self.b0 = float(lr.intercept_[0])
        self.sizes = sizes[keep]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.cvec = np.concatenate([beta[i] * woe[j] for i, j in enumerate(keep)])
        self.names = [self.binner.spec[j][1] for j in keep]
        return self

    @property
    def P(self):
        return len(self.keep)

    def cells(self, X: pd.DataFrame):
        B, _ = self.binner.transform(X)
        return B[:, self.keep] + self.offsets[None, :]

    def score(self, cell_id):
        return self.b0 + self.cvec[cell_id].sum(1)


# ---------------------------------------------------------------------------
# Synthetic additive-scorecard populations
# ---------------------------------------------------------------------------
def synthetic_population(N=20000, P=20, B=6, scale=1.0, base_rate=0.08,
                         rho=0.6, rng=None, strength=None):
    """Generator of the original study, parametrised. Returns dict with
    cell_id, cvec, b0, f, p, y. `scale` multiplies all contributions
    (controls discrimination); `rho` is the loading on the common factor."""
    rng = np.random.default_rng(rng)
    if strength is None:
        k = max(1, P // 5)
        strength = np.concatenate([rng.uniform(0.8, 1.3, k), rng.uniform(0.2, 0.6, P - k)])
    signal = rng.normal(size=(N, 1))
    z = rho * signal + np.sqrt(1 - rho ** 2) * rng.normal(size=(N, P))
    # bins are quantiles of the standard normal (population-level, fixed)
    from scipy.stats import norm
    edges = norm.ppf(np.linspace(0, 1, B + 1)[1:-1])
    binidx = np.searchsorted(edges, z)
    bin_vals = (np.arange(B) - (B - 1) / 2) / ((B - 1) / 2)
    # small idiosyncratic perturbation so that contributions are generic
    cmat = scale * 0.45 * strength[:, None] * bin_vals[None, :]
    cmat = cmat + rng.normal(scale=0.02 * scale, size=cmat.shape)
    cvec = cmat.ravel()
    cell_id = binidx + (np.arange(P) * B)[None, :]
    raw = cvec[cell_id].sum(1)
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (1 / (1 + np.exp(-(raw + mid)))).mean() > base_rate:
            hi = mid
        else:
            lo = mid
    b0 = 0.5 * (lo + hi)
    f = raw + b0
    p = 1 / (1 + np.exp(-f))
    y = (rng.random(N) < p).astype(int)
    return dict(cell_id=cell_id, cvec=cvec, b0=b0, f=f, p=p, y=y, P=P, B=B,
                strength=strength, cmat=cmat, edges=edges, rho=rho, scale=scale)


def new_applicants(pop, N, rng):
    """Fresh applicants from the same scorecard (for train/test studies)."""
    rng = np.random.default_rng(rng)
    P, B, rho = pop["P"], pop["B"], pop["rho"]
    signal = rng.normal(size=(N, 1))
    z = rho * signal + np.sqrt(1 - rho ** 2) * rng.normal(size=(N, P))
    binidx = np.searchsorted(pop["edges"], z)
    cell_id = binidx + (np.arange(P) * B)[None, :]
    f = pop["b0"] + pop["cvec"][cell_id].sum(1)
    p = 1 / (1 + np.exp(-f))
    y = (rng.random(N) < p).astype(int)
    return dict(cell_id=cell_id, f=f, p=p, y=y)


# ---------------------------------------------------------------------------
# Quantization and functionals
# ---------------------------------------------------------------------------
def quantize_points(cvec, D, mode="nearest", rng=None):
    """Integer points n (in units of D) per cell."""
    z = cvec / D
    if mode == "nearest":
        return np.round(z)
    if mode == "floor":
        return np.floor(z)
    if mode == "stochastic":
        rng = np.random.default_rng(rng)
        fl = np.floor(z)
        return fl + (rng.random(z.shape) < (z - fl))
    raise ValueError(mode)


def qscore(b0, n, cell_id, D):
    return b0 + D * n[cell_id].sum(1)


def cutoffs(f, K, scheme="ef"):
    if scheme == "ef":
        return np.quantile(f, np.linspace(0, 1, K + 1)[1:-1])
    if scheme == "ew":
        lo, hi = np.quantile(f, [0.005, 0.995])
        return np.linspace(lo, hi, K + 1)[1:-1]
    raise ValueError(scheme)


def grades(s, cuts):
    return np.searchsorted(cuts, s, side="right")


def migration(f, fq, cuts):
    return float(np.mean(grades(f, cuts) != grades(fq, cuts)))


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def sample_pairs(y, npair, rng):
    rng = np.random.default_rng(rng)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return rng.choice(pos, npair), rng.choice(neg, npair)


def realized(f, fq, y, cuts_list, pairs=None):
    """Realized gross and net functionals of a quantized score."""
    p, pq = sigmoid(f), sigmoid(fq)
    out = dict(
        auc0=roc_auc_score(y, f),
        dauc=roc_auc_score(y, fq) - roc_auc_score(y, f),
        mean_dp=float(np.mean(pq - p)),
        mad_p=float(np.mean(np.abs(pq - p))),
        mean_e=float(np.mean(fq - f)),
        sd_e=float(np.std(fq - f)),
        n_scores=int(len(np.unique(np.round(fq, 9)))),
    )
    for K, cuts in cuts_list.items():
        out[f"M{K}"] = migration(f, fq, cuts)
        g0, g1 = grades(f, cuts), grades(fq, cuts)
        sh0 = np.bincount(g0, minlength=K) / len(f)
        sh1 = np.bincount(g1, minlength=K) / len(f)
        out[f"share{K}"] = float(np.abs(sh1 - sh0).sum() / 2)  # total variation
    if pairs is not None:
        i, j = pairs
        m, mq = f[i] - f[j], fq[i] - fq[j]
        out["rev_lost"] = float(np.mean((m > 0) & (mq < 0)))
        out["rev_gain"] = float(np.mean((m < 0) & (mq > 0)))
        out["new_ties"] = float(np.mean((m != 0) & (mq == 0)))
    return out


# ---------------------------------------------------------------------------
# Predictors under the residual-randomization model
# ---------------------------------------------------------------------------
class Predictor:
    """Pre-deployment predictors of quantization effects.

    'ec'  : exact under the residual model (convolution with Irwin--Hall laws),
            computed from the empirical score distribution and pair Hamming
            distances; no density estimation.
    'cf'  : closed-form leading terms (Theorems 2 and 3) using kernel
            estimates of the score density at the cutoffs and of the derivative
            of the Hamming-weighted margin density at zero.
    """

    def __init__(self, f, y, cell_id, ih: IrwinHall, npair=400_000, rng=0):
        self.f, self.y, self.cell_id, self.ih = f, y, cell_id, ih
        self.P = cell_id.shape[1]
        i, j = sample_pairs(y, npair, rng)
        self.m = f[i] - f[j]
        self.Dham = (cell_id[i] != cell_id[j]).sum(1)
        self.kappa = ih.mean_abs(self.P)
        p = sigmoid(f)
        self.Epq = float(np.mean(p * (1 - p)))
        self.Es2 = float(np.mean(p * (1 - p) * (1 - 2 * p)))
        # kernel estimates for the closed form
        self._kde = gaussian_kde(f if len(f) <= 60000 else
                                 np.random.default_rng(rng).choice(f, 60000, replace=False))
        w = self.Dham + 1.0
        sd = np.std(self.m)
        b = 1.06 * sd * len(self.m) ** (-1 / 7)
        Kb = np.exp(-0.5 * (self.m / b) ** 2) / (b * np.sqrt(2 * np.pi))
        self.hprime0 = float(np.mean(w * (self.m / b ** 2) * Kb))
        self.phi_prime0 = float(np.mean((self.m / b ** 2) * Kb))
        self.phi0 = float(np.mean(Kb))

    # --- gross functionals -------------------------------------------------
    def migration_ec(self, D, cuts):
        f = self.f
        k = grades(f, cuts)
        up = np.append(cuts, np.inf)[k]
        lo = np.insert(cuts, 0, -np.inf)[k]
        # stay iff lo <= f + e < up  <=>  (lo-f)/D <= S_P < (up-f)/D
        stay = self.ih.cdf(self.P, (up - f) / D) - self.ih.cdf(self.P, (lo - f) / D)
        return float(1.0 - np.mean(stay))

    def migration_cf(self, D, cuts):
        return float(self.kappa * D * self._kde(cuts).sum())

    def mad_p_cf(self, D):
        return self.Epq * self.kappa * D

    # --- net functionals ---------------------------------------------------
    def dauc_ec(self, D):
        n = 2 * (self.Dham + 1)  # D cells per side + tie jitter on each side
        auc_q = np.mean(self.ih.cdf(n, self.m / D))
        auc_0 = np.mean((self.m > 0) + 0.5 * (self.m == 0))
        return float(auc_q - auc_0)

    def dauc_cf(self, D):
        return float(-(D ** 2) / 12.0 * self.hprime0)

    def mean_dp_cf(self, D):
        return float(self.P * D ** 2 / 24.0 * self.Es2)

    def share_tv_cf(self, D, cuts):
        """Leading term of the total-variation change of grade shares."""
        # d share_k = (Var e / 2) (rho'(q_k) - rho'(q_{k-1}))
        h = 1e-3 * np.std(self.f)
        rp = (self._kde(cuts + h) - self._kde(cuts - h)) / (2 * h)
        rp = np.concatenate([[0.0], rp, [0.0]])
        ds = 0.5 * self.P * D ** 2 / 12.0 * (rp[1:] - rp[:-1])
        return float(np.abs(ds).sum() / 2)

    # --- design rules ------------------------------------------------------
    def delta_max_migration(self, tol, cuts):
        return tol / (self.kappa * self._kde(cuts).sum())

    def delta_max_auc(self, tol):
        return np.sqrt(12.0 * tol / max(self.hprime0, 1e-12))


def auc_gradient(f, y, cell_id, n_cells, npair=400_000, rng=0):
    """Kernel estimate of the gradient of the AUC with respect to each cell
    contribution: g_c = E[K_b(m) (1{c in X+} - 1{c in X-})]."""
    i, j = sample_pairs(y, npair, rng)
    m = f[i] - f[j]
    b = 1.06 * np.std(m) * npair ** (-1 / 5)
    w = np.exp(-0.5 * (m / b) ** 2) / (b * np.sqrt(2 * np.pi))
    P = cell_id.shape[1]
    gp = np.bincount(cell_id[i].ravel(), weights=np.repeat(w, P), minlength=n_cells)
    gn = np.bincount(cell_id[j].ravel(), weights=np.repeat(w, P), minlength=n_cells)
    return (gp - gn) / npair


def dauc_fluct_sd(grad, D):
    """First-order fluctuation scale of the AUC change: D * ||g|| / sqrt(12)."""
    return float(D * np.sqrt((grad ** 2).sum() / 12.0))


def residual_mc(f, y, cell_id, n_cells, D, cuts_list, R=200, rng=0, with_auc=True):
    """Residual-randomization Monte Carlo: draws U ~ U(-1/2,1/2) per cell and
    returns realized functionals of f + D*sum(U) (plus tie jitter for AUC)."""
    rng = np.random.default_rng(rng)
    N = len(f)
    rows = []
    auc0 = roc_auc_score(y, f) if with_auc else np.nan
    for _ in range(R):
        U = rng.uniform(-0.5, 0.5, n_cells)
        e = D * U[cell_id].sum(1)
        fq = f + e
        r = {}
        for K, cuts in cuts_list.items():
            r[f"M{K}"] = migration(f, fq, cuts)
        if with_auc:
            J = D * rng.uniform(-0.5, 0.5, N)
            r["dauc"] = roc_auc_score(y, fq + J) - auc0
        rows.append(r)
    return pd.DataFrame(rows)


def residual_uniformity(cvec, cell_id, D):
    """Usage-weighted Kolmogorov--Smirnov distance between the normalised
    rounding residuals of the active cells and U(-1/2,1/2): a diagnostic for
    the residual model (large values flag lattice-commensurate tables)."""
    r = cvec / D - np.round(cvec / D)
    w = np.bincount(cell_id.ravel(), minlength=len(cvec)).astype(float)
    o = np.argsort(r)
    rs, ws = r[o], w[o] / w.sum()
    F = np.cumsum(ws)
    U = rs + 0.5
    return float(max(np.max(np.abs(F - U)), np.max(np.abs(F - ws - U))))


# ---------------------------------------------------------------------------
# Rounding algorithms: single-cell flips (ablation) and offset-aware rounding
# ---------------------------------------------------------------------------
def _members(cell_id, n_cells):
    N, P = cell_id.shape
    order = np.argsort(cell_id.ravel(), kind="stable")
    flat = cell_id.ravel()[order]
    rows = order // P
    starts = np.searchsorted(flat, np.arange(n_cells))
    ends = np.searchsorted(flat, np.arange(n_cells), side="right")
    return [rows[s:e] for s, e in zip(starts, ends)]


def cell_flip(cvec, cell_id, b0, f, D, cuts, bias_tol=0.05, max_sweeps=30, rng=0):
    """Greedy floor/ceil flips of individual cells that lower the training
    migration count (ablation baseline). Returns integer points n."""
    rng = np.random.default_rng(rng)
    z = cvec / D
    lo = np.floor(z)
    n = np.round(z)
    N = len(f)
    members = _members(cell_id, len(cvec))
    fq = b0 + D * n[cell_id].sum(1)
    g0 = grades(f, cuts)
    mis = grades(fq, cuts) != g0
    esum = float((fq - f).sum())
    tol = max(bias_tol * D * N, abs(esum))
    movable = np.flatnonzero(np.abs(z - lo) > 1e-12)
    for _ in range(max_sweeps):
        changed = 0
        for c in rng.permutation(movable):
            idx = members[c]
            if len(idx) == 0:
                continue
            step = D if n[c] == lo[c] else -D
            fnew = fq[idx] + step
            mnew = grades(fnew, cuts) != g0[idx]
            dm = int(mnew.sum()) - int(mis[idx].sum())
            enew = esum + step * len(idx)
            if abs(enew) > tol:
                continue
            if dm < 0 or (dm == 0 and abs(enew) < abs(esum) - 1e-12):
                n[c] = lo[c] + (1 if step > 0 else 0)
                fq[idx] = fnew
                mis[idx] = mnew
                esum = enew
                changed += 1
        if changed == 0:
            break
    return n


class OffsetRounding:
    """Offset-aware rounding (OAR).

    Feature j is rounded on the shifted lattice theta_j + D*Z:
        n_{j,b} = round((c_{j,b} - theta_j)/D),   theta_j in [-D/2, D/2),
    and the intercept absorbs sum_j theta_j, so the deployed score is
        fq = b0 + sum_j theta_j + D * sum_j n_{j,B_j}.
    Points stay integers, every per-cell error stays in [-D/2, D/2] (so all
    deterministic bounds are preserved), and theta = 0 is nearest rounding.
    Offsets are chosen by cyclic coordinate descent over a grid of G values to
    minimise the training migration count, subject to |mean e| <= tau*D.
    One evaluation costs O(N); a sweep costs O(N P G).
    """

    def __init__(self, D, grid=16, sweeps=6, bias_tol=0.05, rng=0):
        self.D, self.grid, self.sweeps, self.bias_tol, self.rng = D, grid, sweeps, bias_tol, rng

    def fit(self, cvec, cell_id, b0, f, cuts):
        D = self.D
        rng = np.random.default_rng(self.rng)
        N, P = cell_id.shape
        theta = np.zeros(P)
        cand = (np.arange(self.grid) / self.grid - 0.5) * D
        T = np.column_stack([D * np.round(cvec / D)[cell_id[:, j]] for j in range(P)])
        fq = b0 + T.sum(1)
        g0 = grades(f, cuts)
        obj = int(np.sum(grades(fq, cuts) != g0))
        esum = float((fq - f).sum())
        tol = max(self.bias_tol * D * N, abs(esum))
        self.history = [obj / N]
        for _ in range(self.sweeps):
            improved = False
            for j in rng.permutation(P):
                base = fq - T[:, j]
                best = (obj, abs(esum), theta[j], None, esum)
                for th in cand:
                    tj = th + D * np.round((cvec - th) / D)[cell_id[:, j]]
                    fc = base + tj
                    es = float((fc - f).sum())
                    if abs(es) > tol:
                        continue
                    o = int(np.sum(grades(fc, cuts) != g0))
                    if (o, abs(es)) < best[:2]:
                        best = (o, abs(es), th, tj, es)
                if best[3] is not None and (best[0], best[1]) < (obj, abs(esum)):
                    if best[0] < obj:
                        improved = True
                    theta[j], T[:, j] = best[2], best[3]
                    fq = base + best[3]
                    obj, esum = best[0], best[4]
            self.history.append(obj / N)
            if not improved:
                break
        self.theta = theta
        self.b0 = b0
        return self

    def points(self, cvec, feat_of_cell):
        return np.round((cvec - self.theta[feat_of_cell]) / self.D)

    def score(self, cvec, feat_of_cell, cell_id):
        n = self.points(cvec, feat_of_cell)
        return self.b0 + self.theta.sum() + self.D * n[cell_id].sum(1)


def random_offsets(cvec, feat_of_cell, cell_id, b0, f, D, cuts, R=50, bias_tol=0.05, rng=0):
    """Best of R random offset vectors on the training data (baseline)."""
    rng = np.random.default_rng(rng)
    N, P = cell_id.shape
    g0 = grades(f, cuts)
    fq0 = b0 + D * np.round(cvec / D)[cell_id].sum(1)
    tol = max(bias_tol * D * N, abs(float((fq0 - f).sum())))
    best = (int(np.sum(grades(fq0, cuts) != g0)), np.zeros(P))
    for _ in range(R):
        th = rng.uniform(-0.5, 0.5, P) * D
        n = np.round((cvec - th[feat_of_cell]) / D)
        fq = b0 + th.sum() + D * n[cell_id].sum(1)
        if abs(float((fq - f).sum())) > tol:
            continue
        o = int(np.sum(grades(fq, cuts) != g0))
        if o < best[0]:
            best = (o, th)
    return best[1]


def offset_score(cvec, feat_of_cell, cell_id, b0, D, theta):
    n = np.round((cvec - theta[feat_of_cell]) / D)
    return b0 + theta.sum() + D * n[cell_id].sum(1)


def model_diagnostics(f, fq, D, P):
    """Residual-model checks: sd ratio (target 1) and corr(e, f) (target 0)."""
    e = fq - f
    return dict(rms_ratio=float(np.sqrt(np.mean(e ** 2)) / (D * np.sqrt(P / 12.0))),
                sd_ratio=float(np.std(e) / (D * np.sqrt(P / 12.0))),
                corr_ef=float(np.corrcoef(e, f)[0, 1]) if np.std(e) > 0 else 0.0)


def recalibrate_cutoffs(f, fq, cuts, D, b0, span=8):
    """Per-cutoff choice of a lattice mid-point minimising disagreement with
    the continuous grade (cutoffs are then used on the quantized score)."""
    new = []
    for q in cuts:
        m0 = np.floor((q - b0) / D)
        cand = b0 + D * (m0 + 0.5 + np.arange(-span, span + 1))
        below = f < q
        best, bt = None, q
        for t in cand:
            err = np.sum(below & (fq >= t)) + np.sum(~below & (fq < t))
            if best is None or err < best:
                best, bt = err, t
        new.append(bt)
    return np.sort(np.array(new))


def migration_cr(f, fq, cuts_cont, cuts_q):
    return float(np.mean(grades(f, cuts_cont) != grades(fq, cuts_q)))
