"""E4b. Residual-randomization intervals computed on the evaluation (test)
sample itself, to separate model adequacy from train/test sampling noise."""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "exp")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sqlib import *
from e4_real_data import load, LATTICES, KS

rows = []
for name in ["Taiwan", "GMSC", "HELOC", "German"]:
    X, y, B = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    sc = Scorecard(n_bins=B, min_share=0.02).fit(Xtr, ytr)
    ctr, cte = sc.cells(Xtr), sc.cells(Xte)
    ftr, fte = sc.score(ctr), sc.score(cte)
    cl = {K: cutoffs(ftr, K) for K in KS}
    for pdo, g in LATTICES:
        D = delta_from_pdo(pdo, g)
        fq = qscore(sc.b0, quantize_points(sc.cvec, D), cte, D)
        r = realized(fte, fq, yte, cl)
        mc = residual_mc(fte, yte, cte, len(sc.cvec), D, cl, R=400, rng=1)
        row = dict(dataset=name, PDO=pdo, grid=g, Delta=D, dauc=r["dauc"],
                   dauc_lo=mc.dauc.quantile(0.025), dauc_hi=mc.dauc.quantile(0.975))
        for K in KS:
            row[f"M{K}"] = r[f"M{K}"]
            row[f"M{K}_lo"], row[f"M{K}_hi"] = mc[f"M{K}"].quantile([0.025, 0.975])
        rows.append(row)
    print(name, flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/e4b_test_intervals.csv", index=False)
cov = pd.concat([(df[f"M{K}"] >= df[f"M{K}_lo"]) & (df[f"M{K}"] <= df[f"M{K}_hi"]) for K in KS])
print("migration coverage", cov.mean(), "AUC coverage", ((df.dauc >= df.dauc_lo) & (df.dauc <= df.dauc_hi)).mean())
