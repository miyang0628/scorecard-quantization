"""LaTeX tables (booktabs) generated directly from the result files, plus a
JSON of every number quoted in the text (results/numbers.json)."""
import json
import numpy as np, pandas as pd

R = "results"
T = "tables"
DATASETS = ["Taiwan", "GMSC", "HELOC", "German"]
LN2 = np.log(2)
NUM = {}


def ms(x, d=2):
    return f"{np.mean(x):.{d}f} ({np.std(x):.{d}f})"


def tab_sim_accuracy():
    d = pd.read_csv(f"{R}/e1_synthetic_accuracy.csv")
    lines = []
    for D, s in d.groupby("Delta"):
        r = [f"{D:g}"]
        for K in [7, 10, 20]:
            r.append(ms(s[f"M{K}"] / s[f"M{K}_ec"]))
        r.append(ms(s["M10ew"] / s["M10ew_ec"]))
        r.append(ms(s["mad_p"] / s["mad_cf"]))
        r.append(f"{-1e4 * s.dauc.mean():.2f}")
        r.append(f"{-1e4 * s.dauc_ec.mean():.2f}")
        r.append(f"{-1e4 * s.dauc_cf.mean():.2f}")
        r.append(f"{s.rms_ratio.mean():.2f}")
        r.append(f"{100 * (s.corr_ef.abs() > 0.5).mean():.0f}")
        lines.append(" & ".join(r) + r" \\")
        NUM[f"sim_M10_ratio_D{D:g}"] = float((s.M10 / s.M10_ec).mean())
        NUM[f"sim_M10_ratio_sd_D{D:g}"] = float((s.M10 / s.M10_ec).std())
        NUM[f"sim_flag_D{D:g}"] = float((s.corr_ef.abs() > 0.5).mean())
    s = d[d.Delta <= 0.2]
    for K in [7, 10, 20]:
        NUM[f"sim_M{K}_ratio_le02"] = float((s[f"M{K}"] / s[f"M{K}_ec"]).mean())
    NUM["sim_cf_ratio_le02"] = float((s.M10 / s.M10_cf).mean())
    NUM["sim_n_pops"] = int(d[["cfg", "rep"]].drop_duplicates().shape[0])
    body = "\n".join(lines)
    tex = r"""\begin{table}[t]
\centering
\caption{Simulation study (360 scorecards, $N=20{,}000$ each): ratio of realized to predicted
effects of nearest rounding, mean (sd) over scorecards. Migration columns use the exact
convolution predictor (\ref{eq:Mexact}); MAD uses (\ref{eq:madp}). AUC losses are means over
scorecards ($\times10^{4}$). The last two columns are the residual-model diagnostics of
Section~\ref{sec:diagnostics}: mean RMS ratio and percentage of scorecards with $|\mathrm{corr}(e,f)|>0.5$.}
\label{tab:sim}
\small
\setlength{\tabcolsep}{3.2pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccccc}
\toprule
 & \multicolumn{4}{c}{Migration: realized / predicted} & MAD$_p$ & \multicolumn{3}{c}{AUC loss $(\times10^4)$} & \multicolumn{2}{c}{Diagnostics}\\
\cmidrule(lr){2-5}\cmidrule(lr){7-9}\cmidrule(lr){10-11}
$\Delta$ & $K=7$ & $K=10$ & $K=20$ & $K=10$, EW & ratio & realized & exact & closed & RMS & flag (\%)\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(f"{T}/tab_sim.tex", "w").write(tex)


def tab_coverage():
    a = pd.read_csv(f"{R}/e2a_coverage.csv")
    lines = []
    for D, s in a.groupby("Delta"):
        cov = [((s[f"M{K}"] >= s[f"M{K}_lo"]) & (s[f"M{K}"] <= s[f"M{K}_hi"])).mean() for K in [7, 10, 20]]
        ca = ((s.dauc >= s.dauc_lo) & (s.dauc <= s.dauc_hi)).mean()
        lines.append(f"Synthetic, $\\Delta={D:g}$ & 100 & " + " & ".join(f"{100*c:.0f}" for c in cov) + f" & {100*ca:.0f}" + r" \\")
        NUM[f"cov_syn_M10_D{D:g}"] = float(cov[1]); NUM[f"cov_syn_A_D{D:g}"] = float(ca)
    L = pd.concat([pd.read_csv(f"{R}/e4_{n}_lattice.csv") for n in DATASETS])
    Tt = pd.read_csv(f"{R}/e4b_test_intervals.csv")
    for lab, df in [("Credit data, development-sample intervals", L), ("Credit data, evaluation-sample intervals", Tt)]:
        cov = [((df[f"M{K}"] >= df[f"M{K}_lo"]) & (df[f"M{K}"] <= df[f"M{K}_hi"])).mean() for K in [7, 10, 20]]
        ca = ((df.dauc >= df.dauc_lo) & (df.dauc <= df.dauc_hi)).mean()
        lines.append(f"{lab} & {len(df)} & " + " & ".join(f"{100*c:.0f}" for c in cov) + f" & {100*ca:.0f}" + r" \\")
        key = "dev" if "development" in lab else "eval"
        NUM[f"cov_real_{key}_M"] = float(np.mean(cov)); NUM[f"cov_real_{key}_A"] = float(ca)
    L["lr"] = np.log(L.M10 / L.M10_ec)
    lr = pd.concat([np.log(L[f"M{K}"] / L[f"M{K}_ec"]) for K in [7, 10, 20]])
    rr = pd.concat([np.log(L.rms_ratio)] * 3)
    NUM["real_corr_logratio_rms"] = float(np.corrcoef(lr, rr)[0, 1])
    tex = r"""\begin{table}[t]
\centering
\caption{Empirical coverage (\%) of nominal 95\% residual-randomization prediction intervals
($R=200$ draws; 400 for evaluation-sample intervals) for the realized migration and AUC change of
nearest rounding. Synthetic rows: 100 scorecards per spacing. Credit rows: 4 data sets $\times$ 5 lattices.}
\label{tab:coverage}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
 & cases & $M$, $K=7$ & $M$, $K=10$ & $M$, $K=20$ & $\Delta A$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(f"{T}/tab_coverage.tex", "w").write(tex)


def tab_rounding_synth():
    d = pd.read_csv(f"{R}/e3_rounding_synthetic.csv")
    order = ["nearest", "floor", "stochastic", "CR", "cell-flip", "random-offset", "OAR", "OAR+CR"]
    names = {"nearest": "Nearest", "floor": "Floor", "stochastic": "Stochastic (10 draws)",
             "CR": "Cutoff recalibration (CR)", "cell-flip": "Single-cell flips",
             "random-offset": "Best of 50 random offsets", "OAR": "OAR (Algorithm~\\ref{alg:oar})",
             "OAR+CR": "OAR + CR"}
    lines = []
    for D in [0.05, 0.1, 0.2]:
        s = d[(d.Delta == D)]
        base = s[s.method == "nearest"].set_index(["seed", "K"]).M
        lines.append(r"\multicolumn{7}{l}{\textit{$\Delta=" + f"{D:g}" + r"$}}\\")
        for m in order:
            t = s[s.method == m]
            mm = t.set_index(["seed", "K"]).M
            rel = 100 * (mm / base - 1).mean()
            lines.append(f"\\quad {names[m]} & {t.M.mean():.3f} & {rel:+.0f} & {1e4*t.dauc.mean():.1f} & "
                         f"{t.bias.mean():+.2f} & {1e3*t.mad_p.mean():.1f} & {t.secs.mean():.2f}" + r" \\")
            NUM[f"syn_{m}_rel_D{D:g}"] = float(rel)
    tex = r"""\begin{table}[t]
\centering
\caption{Rounding algorithms on 40 synthetic scorecards ($P=20$, $B=6$), fitted on a training sample and
evaluated on an independent test sample ($N=20{,}000$ each); averages over the 40 scorecards and
$K\in\{7,10,20\}$. Rel.: mean relative change of test migration against nearest rounding.}
\label{tab:rounding-syn}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
 & test $M$ & rel.\ (\%) & $\Delta A$ $(\times10^4)$ & mean $e/\Delta$ & MAD$_p$ $(\times10^3)$ & time (s)\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(f"{T}/tab_rounding_syn.tex", "w").write(tex)


def tab_data():
    lines = []
    for nm in DATASETS:
        info = json.load(open(f"{R}/e4_{nm}_info.json"))
        lines.append(f"{nm} & {info['N']:,} & {info['N_train']:,} & {info['P']} & {info['n_cells']} & "
                     f"{100*info['event_rate']:.1f} & {info['auc_train']:.3f} & {info['auc_test']:.3f}" + r" \\")
        NUM[f"{nm}_info"] = {k: info[k] for k in ["N", "P", "n_cells", "event_rate", "auc_train", "auc_test"]}
    tex = r"""\begin{table}[t]
\centering
\caption{Credit data sets and fitted WoE scorecards. $P$: features retained; cells: total bins
(including missing/other bins); event: default rate.}
\label{tab:data}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
Data set & $N$ & train & $P$ & cells & event (\%) & AUC train & AUC test\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(f"{T}/tab_data.tex", "w").write(tex)


def tab_real():
    lines = []
    for nm in DATASETS:
        L = pd.read_csv(f"{R}/e4_{nm}_lattice.csv")
        A = pd.read_csv(f"{R}/e4_{nm}_algorithms.csv")
        Bt = pd.read_csv(f"{R}/e4_{nm}_bootstrap.csv")
        lines.append(r"\multicolumn{11}{l}{\textit{" + nm + r"} (re-estimation migration " +
                     f"{Bt.Mrefit10.mean():.3f}" + r")}\\")
        NUM[f"{nm}_refit10"] = float(Bt.Mrefit10.mean())
        for _, r in L.iterrows():
            a = A[(A.K == 10) & (np.isclose(A.Delta, r.Delta))].set_index("method")
            oar = a.loc["OAR", "M_test"]
            red = 100 * (oar / r.M10 - 1)
            lines.append(f"\\quad {int(r.PDO)}/{int(r.grid)} & {r.Delta:.4f} & {int(r.n_scores)} & {r.M10:.3f} & "
                         f"{r.M10_ec:.3f} & [{r.M10_lo:.3f}, {r.M10_hi:.3f}] & {1e3*r.dauc:+.1f} & {1e3*r.dauc_ec:+.1f} & "
                         f"{1e3*r.dauc_sd_grad:.1f} & {r.rms_ratio:.2f} & {oar:.3f} ({red:+.0f}\\%)" + r" \\")
            NUM[f"{nm}_{int(r.PDO)}_{int(r.grid)}"] = dict(M10=r.M10, M10_ec=r.M10_ec, lo=r.M10_lo, hi=r.M10_hi,
                                                         dauc=r.dauc, dauc_ec=r.dauc_ec, oar=oar, red=red,
                                                         rms=r.rms_ratio, n_scores=int(r.n_scores),
                                                         cr=a.loc["CR", "M_test"], rnd=a.loc["random-offset", "M_test"])
    tex = r"""\begin{table}[t]
\centering
\caption{Credit data: effects of integerizing the WoE scorecard at industry lattices (PDO/grid),
$K=10$ equal-frequency grades, evaluated on the test sample. Predictions and 95\% intervals use the
development sample only. $\Delta A$ and its predicted first-order fluctuation sd
$\Delta\|\hat g\|/\sqrt{12}$ are $\times10^{3}$. Last column: test migration of offset-aware rounding
(relative change against nearest rounding).}
\label{tab:real}
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccccc}
\toprule
PDO/$g$ & $\Delta$ & scores & $M$ & $\hat M$ & 95\% interval & $\Delta A$ & $\widehat{\Delta A}$ & fluct.\ sd & RMS & OAR $M$\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    open(f"{T}/tab_real.tex", "w").write(tex)


def numbers_extra():
    b = pd.concat([pd.read_csv(f"{R}/e4_{n}_bootstrap.csv") for n in DATASETS])
    for nm, s in b.groupby("dataset"):
        for g in [2, 5]:
            NUM[f"{nm}_boot_g{g}"] = dict(mean=float(s[f"Mq10_g{g}"].mean()), sd=float(s[f"Mq10_g{g}"].std()),
                                        ec=float(s[f"Mq10_g{g}_ec"].mean()),
                                        frac_gt_refit=float((s[f"Mq10_g{g}"] > s.Mrefit10).mean()))
        NUM[f"{nm}_boot_refit"] = dict(mean=float(s.Mrefit10.mean()), sd=float(s.Mrefit10.std()))
    e2b = pd.read_csv(f"{R}/e2b_stationarity.csv")
    sl = {}
    for sp, s in e2b.groupby("spec"):
        v = []
        for _, ss in s.groupby("seed"):
            x = np.log(ss.Delta)
            v.append((np.polyfit(x, np.log(ss.M10_mean), 1)[0],
                      np.polyfit(x, np.log(-ss.dauc_mean), 1)[0] if (ss.dauc_mean < 0).all() else np.nan,
                      np.polyfit(x, np.log(ss.dauc_sd), 1)[0]))
        v = np.array(v)
        sl[sp] = dict(M=np.nanmean(v[:, 0]), M_sd=np.nanstd(v[:, 0]), Amean=np.nanmean(v[:, 1]),
                      Amean_sd=np.nanstd(v[:, 1]), Asd=np.nanmean(v[:, 2]), Asd_sd=np.nanstd(v[:, 2]),
                      n_neg=int(np.isfinite(v[:, 1]).sum()))
    NUM["slopes"] = sl
    m = e2b[e2b.spec == "misspecified"].groupby("Delta").dauc_mean.mean()
    NUM["mis_mean_dauc"] = m.to_dict()
    c = pd.read_csv(f"{R}/e2c_gradient.csv")
    NUM["grad_check"] = c.groupby(["spec", "N"])[["sd_mc", "sd_grad"]].mean().reset_index().to_dict("records")
    e3 = pd.read_csv(f"{R}/e3_rounding_synthetic.csv")
    NUM["oar_secs"] = float(e3[e3.method == "OAR"].secs.mean())
    alg = pd.concat([pd.read_csv(f"{R}/e4_{n}_algorithms.csv") for n in DATASETS])
    nr = alg[alg.method == "nearest"].set_index(["dataset", "Delta", "K"]).M_test
    for m in ["CR", "random-offset", "OAR", "OAR+CR"]:
        mm = alg[alg.method == m].set_index(["dataset", "Delta", "K"]).M_test
        rel = (mm / nr - 1)
        NUM[f"real_rel_{m}"] = dict(mean=float(rel.mean()), median=float(rel.median()),
                                  frac_better=float((rel < 0).mean()),
                                  by_dataset=rel.groupby(level=0).mean().to_dict(),
                                  excl_german=float(rel.drop("German", level=0).mean()))
    oar = alg[alg.method == "OAR"]
    NUM["real_oar_dauc_max"] = float(oar.dauc.abs().max()); NUM["real_oar_bias_max"] = float(oar.bias.abs().max())
    NUM["real_near_dauc_by_lattice"] = alg[alg.method == "nearest"].groupby("grid").dauc.agg(["min", "max"]).to_dict()
    json.dump(NUM, open(f"{R}/numbers.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    import os
    os.makedirs(T, exist_ok=True)
    tab_sim_accuracy(); tab_coverage(); tab_rounding_synth(); tab_data(); tab_real(); numbers_extra()
    print("tables written")
