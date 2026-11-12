"""E2. (A) Coverage of residual-randomization prediction intervals for the
realized migration and AUC change of nearest rounding; (B) stationarity of the
AUC: Monte Carlo mean and spread of the AUC change versus Delta for a
well-specified score (f = true log-odds) and a misspecified one.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sqlib import *

IH = IrwinHall(2 * 20 + 2)
KS = [7, 10, 20]


def part_a(seed):
    pop = synthetic_population(N=20000, P=20, B=6, scale=1.0, base_rate=0.08, rng=50_000 + seed)
    f, y, cid, cvec, b0 = pop["f"], pop["y"], pop["cell_id"], pop["cvec"], pop["b0"]
    cl = {K: cutoffs(f, K) for K in KS}
    pr = Predictor(f, y, cid, IH, npair=300_000, rng=seed)
    rows = []
    for D in [0.05, 0.1, 0.2]:
        n = quantize_points(cvec, D)
        fq = qscore(b0, n, cid, D)
        r = realized(f, fq, y, cl)
        mc = residual_mc(f, y, cid, len(cvec), D, cl, R=200, rng=seed)
        row = dict(seed=seed, Delta=D, dauc=r["dauc"], dauc_ec=pr.dauc_ec(D),
                   dauc_mc_mean=mc.dauc.mean(), dauc_mc_sd=mc.dauc.std(),
                   dauc_lo=mc.dauc.quantile(0.025), dauc_hi=mc.dauc.quantile(0.975))
        for K in KS:
            row.update({f"M{K}": r[f"M{K}"], f"M{K}_ec": pr.migration_ec(D, cl[K]),
                        f"M{K}_mc_mean": mc[f"M{K}"].mean(), f"M{K}_mc_sd": mc[f"M{K}"].std(),
                        f"M{K}_lo": mc[f"M{K}"].quantile(0.025), f"M{K}_hi": mc[f"M{K}"].quantile(0.975)})
        row.update(model_diagnostics(f, fq, D, 20))
        rows.append(row)
    return rows


def part_b(args):
    seed, spec = args
    pop = synthetic_population(N=50000, P=20, B=6, scale=1.0, base_rate=0.08, rng=70_000 + seed)
    y, cid, cvec, b0 = pop["y"], pop["cell_id"], pop["cvec"], pop["b0"]
    if spec == "misspecified":
        rng = np.random.default_rng(seed)
        a = rng.uniform(0.4, 1.6, pop["P"])
        cvec = cvec * np.repeat(a, pop["B"])
    f = b0 + cvec[cid].sum(1)
    cl = {10: cutoffs(f, 10)}
    rows = []
    for D in [0.0125, 0.025, 0.05, 0.1, 0.2]:
        mc = residual_mc(f, y, cid, len(cvec), D, cl, R=100, rng=seed)
        rows.append(dict(seed=seed, spec=spec, Delta=D, dauc_mean=mc.dauc.mean(),
                         dauc_sd=mc.dauc.std(), M10_mean=mc.M10.mean(), M10_sd=mc.M10.std()))
    return rows


if __name__ == "__main__":
    t = time.time()
    out = []
    with Pool(2) as pool:
        for k, rows in enumerate(pool.imap_unordered(part_a, range(100))):
            out.extend(rows)
            if k % 20 == 0:
                print("A", k, round(time.time() - t), "s", flush=True)
    pd.DataFrame(out).to_csv("results/e2a_coverage.csv", index=False)
    out = []
    jobs = [(s, sp) for s in range(30) for sp in ["well-specified", "misspecified"]]
    with Pool(2) as pool:
        for k, rows in enumerate(pool.imap_unordered(part_b, jobs)):
            out.extend(rows)
            if k % 10 == 0:
                print("B", k, round(time.time() - t), "s", flush=True)
    pd.DataFrame(out).to_csv("results/e2b_stationarity.csv", index=False)
    print("done", round(time.time() - t), "s")
