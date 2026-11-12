"""E1. Factorial Monte Carlo: accuracy of the predictors and order separation.

Design: P in {10,20,40} x scale in {0.7,1.0,1.5} x base rate in {0.02,0.08}
x bins B in {5,8}; R=10 replicate scorecards per cell (360 populations,
N=20,000 each). For every population and spacing Delta in
{0.02,0.05,0.1,0.2,0.4} we record realized functionals of nearest rounding
and the predictions of Section 4.
"""
import sys, itertools, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sqlib import *

DELTAS = [0.02, 0.05, 0.1, 0.2, 0.4]
KS = [7, 10, 20]
IH = IrwinHall(2 * 40 + 2)


def run(args):
    cfg_id, P, scale, base, B, rep = args
    seed = 10_000 * cfg_id + rep
    pop = synthetic_population(N=20000, P=P, B=B, scale=scale, base_rate=base, rng=seed)
    f, y, cid, cvec, b0 = pop["f"], pop["y"], pop["cell_id"], pop["cvec"], pop["b0"]
    cl = {K: cutoffs(f, K, "ef") for K in KS}
    cl_ew = cutoffs(f, 10, "ew")
    pr = Predictor(f, y, cid, IH, npair=300_000, rng=seed)
    pairs = sample_pairs(y, 300_000, seed + 1)
    rows = []
    for D in DELTAS:
        n = quantize_points(cvec, D)
        fq = qscore(b0, n, cid, D)
        r = realized(f, fq, y, cl, pairs)
        r.update(model_diagnostics(f, fq, D, P))
        r["ks"] = residual_uniformity(cvec, cid, D)
        r["M10ew"] = migration(f, fq, cl_ew)
        for K in KS:
            r[f"M{K}_ec"] = pr.migration_ec(D, cl[K])
            r[f"M{K}_cf"] = pr.migration_cf(D, cl[K])
        r["M10ew_ec"] = pr.migration_ec(D, cl_ew)
        r["M10ew_cf"] = pr.migration_cf(D, cl_ew)
        r["dauc_ec"] = pr.dauc_ec(D)
        r["dauc_cf"] = pr.dauc_cf(D)
        r["mad_cf"] = pr.mad_p_cf(D)
        r.update(cfg=cfg_id, P=P, scale=scale, base=base, B=B, rep=rep, Delta=D,
                 base_obs=float(y.mean()))
        rows.append(r)
    return rows


if __name__ == "__main__":
    grid = list(itertools.product([10, 20, 40], [0.7, 1.0, 1.5], [0.02, 0.08], [5, 8]))
    jobs = [(ci, P, s, b, B, rep) for ci, (P, s, b, B) in enumerate(grid) for rep in range(10)]
    t = time.time()
    out = []
    with Pool(2) as pool:
        for k, rows in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            out.extend(rows)
            if k % 40 == 0:
                print(k, "of", len(jobs), round(time.time() - t), "s", flush=True)
    df = pd.DataFrame(out)
    df.to_csv("results/e1_synthetic_accuracy.csv", index=False)
    print("done", df.shape, round(time.time() - t), "s")
