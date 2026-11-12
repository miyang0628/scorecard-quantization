"""E2c. First-order AUC fluctuation: sd_R(dAUC) versus the gradient formula
Delta * sqrt(sum_c g_c^2 / 12) and versus the sample size N, for well-specified
and misspecified scores."""
import sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sqlib import *


def auc_gradient(f, y, cell_id, n_cells, npair=400_000, rng=0):
    """Kernel estimate of dAUC/dc_c for every cell c."""
    i, j = sample_pairs(y, npair, rng)
    m = f[i] - f[j]
    b = 1.06 * np.std(m) * npair ** (-1 / 5)
    w = np.exp(-0.5 * (m / b) ** 2) / (b * np.sqrt(2 * np.pi))
    P = cell_id.shape[1]
    gp = np.bincount(cell_id[i].ravel(), weights=np.repeat(w, P), minlength=n_cells)
    gn = np.bincount(cell_id[j].ravel(), weights=np.repeat(w, P), minlength=n_cells)
    return (gp - gn) / npair


rows = []
t = time.time()
for spec in ["well-specified", "misspecified"]:
    for N in [12_500, 50_000, 200_000]:
        for seed in range(4):
            pop = synthetic_population(N=N, P=20, B=6, rng=70_000 + seed)
            y, cid, cvec, b0 = pop["y"], pop["cell_id"], pop["cvec"], pop["b0"]
            if spec == "misspecified":
                a = np.random.default_rng(seed).uniform(0.4, 1.6, 20)
                cvec = cvec * np.repeat(a, 6)
            f = b0 + cvec[cid].sum(1)
            g = auc_gradient(f, y, cid, len(cvec), rng=seed)
            for D in [0.05]:
                mc = residual_mc(f, y, cid, len(cvec), D, {10: cutoffs(f, 10)}, R=60, rng=seed)
                rows.append(dict(spec=spec, N=N, seed=seed, Delta=D, sd_mc=mc.dauc.std(),
                                 sd_grad=D * np.sqrt((g ** 2).sum() / 12), mean_mc=mc.dauc.mean()))
            print(spec, N, seed, round(time.time() - t), flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/e2c_gradient.csv", index=False)
print(df.groupby(["spec", "N"])[["sd_mc", "sd_grad", "mean_mc"]].mean())
