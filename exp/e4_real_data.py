"""E4. Four public credit datasets.

For each dataset: stratified 70/30 split; WoE scorecard fitted on the training
part; deployment lattices from industry (PDO, grid) pairs; predictions from
the training data are compared with realized effects on the test data;
residual-randomization prediction intervals; rounding algorithms; and 100
bootstrap refits that put quantization migration next to the grade migration
caused by re-estimating the model.
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sqlib import *

DATA = "data"
LATTICES = [(40, 1), (20, 1), (20, 2), (20, 5), (20, 10)]
KS = [7, 10, 20]


def load(name):
    if name == "Taiwan":
        d = pd.read_csv(f"{DATA}/taiwan.csv")
        y = d["default.payment.next.month"].to_numpy()
        X = d.drop(columns=["ID", "SEX", "MARRIAGE", "AGE", "default.payment.next.month"])
        return X, y, 8
    if name == "GMSC":
        d = pd.read_csv(f"{DATA}/gmsc.csv", index_col=0)
        y = d["SeriousDlqin2yrs"].to_numpy()
        X = d.drop(columns=["SeriousDlqin2yrs", "age"])
        return X, y, 8
    if name == "HELOC":
        # original FICO file; rows with no bureau record (all -9) removed;
        # special values (-7, -8) are given the missing bin
        d = pd.read_csv(f"{DATA}/heloc_v1.csv")
        X = d.drop(columns=["RiskPerformance"]).astype(float)
        keep = ~(X == -9).all(axis=1)
        X, d = X[keep], d[keep]
        y = (d["RiskPerformance"] == "Bad").astype(int).to_numpy()
        X = X.mask(X < 0)
        return X.reset_index(drop=True), y, 6
    if name == "German":
        d = pd.read_csv(f"{DATA}/german.csv")
        y = (d["creditability"] == "bad").astype(int).to_numpy()
        X = d.drop(columns=["creditability", "personal_status_and_sex", "age_in_years", "foreign_worker"])
        return X, y, 4
    raise ValueError(name)


def feat_of_cells(sc):
    return np.repeat(np.arange(sc.P), sc.sizes)


def main(name, n_boot=100, R_mc=200):
    t0 = time.time()
    X, y, B = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = Scorecard(n_bins=B, min_share=0.02).fit(Xtr, ytr)
    ctr, cte = sc.cells(Xtr), sc.cells(Xte)
    ftr, fte = sc.score(ctr), sc.score(cte)
    feat = feat_of_cells(sc)
    ih = IrwinHall(2 * sc.P + 2)
    info = dict(dataset=name, N=len(y), N_train=len(ytr), N_test=len(yte), P=sc.P,
                n_cells=int(len(sc.cvec)), event_rate=float(y.mean()),
                auc_train=roc_auc_score(ytr, ftr), auc_test=roc_auc_score(yte, fte),
                features=sc.names)
    cl = {K: cutoffs(ftr, K) for K in KS}
    pr = Predictor(ftr, ytr, ctr, ih, npair=400_000, rng=0)
    grad = auc_gradient(ftr, ytr, ctr, len(sc.cvec), rng=0)
    pairs = sample_pairs(yte, 400_000, 1)
    rows, alg = [], []
    for pdo, g in LATTICES:
        D = delta_from_pdo(pdo, g)
        n = quantize_points(sc.cvec, D)
        fq_te = qscore(sc.b0, n, cte, D)
        fq_tr = qscore(sc.b0, n, ctr, D)
        r = realized(fte, fq_te, yte, cl, pairs)
        r.update(model_diagnostics(fte, fq_te, D, sc.P))
        r["ks"] = residual_uniformity(sc.cvec, ctr, D)
        mc = residual_mc(ftr, ytr, ctr, len(sc.cvec), D, cl, R=R_mc, rng=0)
        for K in KS:
            r[f"M{K}_ec"] = pr.migration_ec(D, cl[K])
            r[f"M{K}_cf"] = pr.migration_cf(D, cl[K])
            r[f"M{K}_lo"], r[f"M{K}_hi"] = mc[f"M{K}"].quantile([0.025, 0.975])
        r["dauc_ec"], r["dauc_cf"] = pr.dauc_ec(D), pr.dauc_cf(D)
        r["dauc_lo"], r["dauc_hi"] = mc["dauc"].quantile([0.025, 0.975])
        r["mad_cf"] = pr.mad_p_cf(D)
        r["dauc_sd_grad"] = dauc_fluct_sd(grad, D)
        r["dauc_mc_sd"] = float(mc["dauc"].std())
        r.update(dataset=name, PDO=pdo, grid=g, Delta=D)
        rows.append(r)
        # rounding algorithms (optimised on train, evaluated on test)
        for K in KS:
            cuts = cl[K]
            def add(method, fte_q, ftr_q, secs, cq=None):
                c = cuts if cq is None else cq
                alg.append(dict(dataset=name, PDO=pdo, grid=g, Delta=D, K=K, method=method, secs=secs,
                                M_test=float(np.mean(grades(fte, cuts) != grades(fte_q, c))),
                                M_train=float(np.mean(grades(ftr, cuts) != grades(ftr_q, c))),
                                dauc=roc_auc_score(yte, fte_q) - info["auc_test"],
                                bias=float(np.mean(fte_q - fte)) / D,
                                mad_p=float(np.mean(np.abs(sigmoid(fte_q) - sigmoid(fte))))))
            add("nearest", fq_te, fq_tr, 0.0)
            t = time.time(); cq = recalibrate_cutoffs(ftr, fq_tr, cuts, D, sc.b0)
            add("CR", fq_te, fq_tr, time.time() - t, cq)
            t = time.time(); th = random_offsets(sc.cvec, feat, ctr, sc.b0, ftr, D, cuts, R=50, rng=0)
            add("random-offset", offset_score(sc.cvec, feat, cte, sc.b0, D, th),
                offset_score(sc.cvec, feat, ctr, sc.b0, D, th), time.time() - t)
            t = time.time(); oar = OffsetRounding(D, rng=0).fit(sc.cvec, ctr, sc.b0, ftr, cuts)
            secs = time.time() - t
            o_tr, o_te = oar.score(sc.cvec, feat, ctr), oar.score(sc.cvec, feat, cte)
            add("OAR", o_te, o_tr, secs)
            cq = recalibrate_cutoffs(ftr, o_tr, cuts, D, sc.b0 + oar.theta.sum())
            add("OAR+CR", o_te, o_tr, secs, cq)
        print(name, pdo, g, round(time.time() - t0), "s", flush=True)
    # bootstrap refits: quantization migration vs re-estimation migration
    boot = []
    rng = np.random.default_rng(123)
    g_orig = {K: grades(fte, cl[K]) for K in KS}
    for b in range(n_boot):
        idx = rng.integers(0, len(ytr), len(ytr))
        Xb, yb = Xtr.iloc[idx], ytr[idx]
        sb = Scorecard(n_bins=B, min_share=0.02).fit(Xb, yb)
        cb_tr, cb_te = sb.cells(Xb), sb.cells(Xte)
        fb_tr, fb_te = sb.score(cb_tr), sb.score(cb_te)
        ihb = IrwinHall(2 * sb.P + 2) if sb.P != sc.P else ih
        prb = Predictor(fb_tr, yb, cb_tr, ihb, npair=50_000, rng=b)
        rec = dict(dataset=name, boot=b, P=sb.P, auc_test=roc_auc_score(yte, fb_te))
        for K in KS:
            cb = cutoffs(fb_tr, K)
            rec[f"Mrefit{K}"] = float(np.mean(grades(fb_te, cb) != g_orig[K]))
            for pdo, g in [(20, 2), (20, 5)]:
                D = delta_from_pdo(pdo, g)
                fq = qscore(sb.b0, quantize_points(sb.cvec, D), cb_te, D)
                rec[f"Mq{K}_g{g}"] = migration(fb_te, fq, cb)
                rec[f"Mq{K}_g{g}_ec"] = prb.migration_ec(D, cb)
        boot.append(rec)
        if b % 20 == 0:
            print(name, "boot", b, round(time.time() - t0), "s", flush=True)
    pd.DataFrame(rows).to_csv(f"results/e4_{name}_lattice.csv", index=False)
    pd.DataFrame(alg).to_csv(f"results/e4_{name}_algorithms.csv", index=False)
    pd.DataFrame(boot).to_csv(f"results/e4_{name}_bootstrap.csv", index=False)
    json.dump(info, open(f"results/e4_{name}_info.json", "w"), indent=1)
    print(name, "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
