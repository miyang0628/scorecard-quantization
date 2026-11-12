"""E3. Rounding algorithms on synthetic scorecards with a train/test split.

40 scorecards (P=20, B=6); each is fitted (rounded) on a training sample of
20,000 applicants and evaluated on an independent test sample of 20,000.
Methods: nearest, floor, stochastic, cutoff recalibration (CR), single-cell
flips, best-of-50 random offsets, offset-aware rounding (OAR), OAR+CR.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sqlib import *


def evaluate(f, fq, y, cuts, cuts_q=None):
    cq = cuts if cuts_q is None else cuts_q
    e = fq - f
    return dict(M=float(np.mean(grades(f, cuts) != grades(fq, cq))),
                dauc=roc_auc_score(y, fq) - roc_auc_score(y, f),
                bias=float(np.mean(e)), mad_p=float(np.mean(np.abs(sigmoid(fq) - sigmoid(f)))))


def run(seed):
    pop = synthetic_population(N=20000, P=20, B=6, scale=1.0, base_rate=0.08, rng=90_000 + seed)
    te = new_applicants(pop, 20000, rng=91_000 + seed)
    cvec, b0, P, B = pop["cvec"], pop["b0"], pop["P"], pop["B"]
    feat = np.repeat(np.arange(P), B)
    ftr, cid_tr = pop["f"], pop["cell_id"]
    fte, cid_te, yte = te["f"], te["cell_id"], te["y"]
    rows = []
    for D in [0.05, 0.1, 0.2]:
        for K in [7, 10, 20]:
            cuts = cutoffs(ftr, K)
            def add(method, fq_te, fq_tr, secs, cuts_q=None):
                r = evaluate(fte, fq_te, yte, cuts, cuts_q)
                r["M_train"] = float(np.mean(grades(ftr, cuts) != grades(fq_tr, cuts if cuts_q is None else cuts_q)))
                r.update(seed=seed, Delta=D, K=K, method=method, secs=secs, bias=r["bias"] / D)
                rows.append(r)
            n = quantize_points(cvec, D)
            add("nearest", qscore(b0, n, cid_te, D), qscore(b0, n, cid_tr, D), 0.0)
            n = quantize_points(cvec, D, "floor")
            add("floor", qscore(b0, n, cid_te, D), qscore(b0, n, cid_tr, D), 0.0)
            acc = []
            for s in range(10):
                n = quantize_points(cvec, D, "stochastic", rng=seed * 100 + s)
                acc.append(evaluate(fte, qscore(b0, n, cid_te, D), yte, cuts))
            r = pd.DataFrame(acc).mean().to_dict()
            r.update(seed=seed, Delta=D, K=K, method="stochastic", secs=0.0, M_train=np.nan, bias=r["bias"] / D)
            rows.append(r)
            n = quantize_points(cvec, D)
            fq_tr = qscore(b0, n, cid_tr, D)
            t = time.time(); cq = recalibrate_cutoffs(ftr, fq_tr, cuts, D, b0)
            add("CR", qscore(b0, n, cid_te, D), fq_tr, time.time() - t, cq)
            t = time.time(); nf = cell_flip(cvec, cid_tr, b0, ftr, D, cuts, rng=seed)
            add("cell-flip", qscore(b0, nf, cid_te, D), qscore(b0, nf, cid_tr, D), time.time() - t)
            t = time.time(); th = random_offsets(cvec, feat, cid_tr, b0, ftr, D, cuts, R=50, rng=seed)
            add("random-offset", offset_score(cvec, feat, cid_te, b0, D, th),
                offset_score(cvec, feat, cid_tr, b0, D, th), time.time() - t)
            t = time.time(); oar = OffsetRounding(D, rng=seed).fit(cvec, cid_tr, b0, ftr, cuts)
            secs = time.time() - t
            fq_tr = oar.score(cvec, feat, cid_tr); fq_te = oar.score(cvec, feat, cid_te)
            add("OAR", fq_te, fq_tr, secs)
            cq = recalibrate_cutoffs(ftr, fq_tr, cuts, D, b0 + oar.theta.sum())
            add("OAR+CR", fq_te, fq_tr, secs, cq)
    return rows


if __name__ == "__main__":
    t = time.time(); out = []
    with Pool(2) as pool:
        for k, rows in enumerate(pool.imap_unordered(run, range(40))):
            out.extend(rows)
            print(k, round(time.time() - t), "s", flush=True)
    pd.DataFrame(out).to_csv("results/e3_rounding_synthetic.csv", index=False)
    print("done", round(time.time() - t))
