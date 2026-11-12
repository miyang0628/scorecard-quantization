"""Figures for the CSDA manuscript (vector PDF)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
F = "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v"]
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})
DATASETS = ["Taiwan", "GMSC", "HELOC", "German"]
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter


def clean_x(ax, ticks, labels=None):
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels or [f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_accuracy():
    d = pd.read_csv(f"{R}/e1_synthetic_accuracy.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    lim = [0.004, 1.0]
    for k, K in enumerate([7, 10, 20]):
        s = d[d.Delta <= 0.2]
        ax[0].scatter(s[f"M{K}_ec"], s[f"M{K}"], s=5, color=C[k], marker=MK[k], alpha=0.35,
                      linewidths=0, label=f"K = {K}")
        s4 = d[d.Delta == 0.4]
        ax[0].scatter(s4[f"M{K}_ec"], s4[f"M{K}"], s=7, facecolors="none", edgecolors=C[k],
                      marker=MK[k], alpha=0.5, linewidths=0.5)
    ax[0].plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel="predicted migration (exact convolution)", ylabel="realized migration",
              title="(a) Migration, nearest rounding")
    ax[0].legend(loc="upper left", handletextpad=0.2, markerscale=2)
    ax[0].text(0.97, 0.04, "filled: $\\Delta\\leq0.2$\nhollow: $\\Delta=0.4$", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=6.5, color=INK2)
    g = d.groupby(["cfg", "Delta", "P"], as_index=False)[["dauc", "dauc_ec"]].mean()
    for k, P in enumerate([10, 20, 40]):
        s = g[g.P == P]
        ax[1].scatter(-s.dauc_ec, -s.dauc, s=12, color=C[k], marker=MK[k], alpha=0.8, linewidths=0,
                      label=f"P = {P}")
    lim2 = [5e-6, 0.1]
    ax[1].plot(lim2, lim2, color=INK2, lw=0.8, ls="--")
    ax[1].set(xscale="log", yscale="log", xlim=lim2, ylim=lim2, xlabel="predicted AUC loss",
              ylabel="realized AUC loss (cell mean)", title="(b) AUC loss, nearest rounding")
    ax[1].legend(loc="upper left", handletextpad=0.2, markerscale=1.5)
    s = d[d.Delta <= 0.2]
    for k, K in enumerate([7, 10, 20]):
        ax[2].scatter(s[f"M{K}_cf"], s[f"M{K}"], s=5, color=C[k], marker=MK[k], alpha=0.35, linewidths=0)
    ax[2].plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax[2].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel=r"closed form $\kappa_P\Delta\sum_k\hat\rho(q_k)$", ylabel="realized migration",
              title="(c) First-order closed form")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{F}/fig1_accuracy.pdf")
    plt.close(fig)


def fig_orders():
    b = pd.read_csv(f"{R}/e2b_stationarity.csv")
    c = pd.read_csv(f"{R}/e2c_gradient.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.45))
    w = b[b.spec == "well-specified"].groupby("Delta").mean(numeric_only=True)
    x = w.index.values
    ax[0].plot(x, w.M10_mean, color=C[0], marker=MK[0], label="migration $M$ ($K=10$)")
    ax[0].plot(x, -w.dauc_mean, color=C[1], marker=MK[1], label=r"AUC loss $-\mathrm{E}\,\Delta A$")
    ax[0].plot(x, w.M10_mean.iloc[2] * x / x[2], color=INK2, lw=0.7, ls=":")
    ax[0].plot(x, -w.dauc_mean.iloc[2] * (x / x[2]) ** 2, color=INK2, lw=0.7, ls=":")
    ax[0].text(x[3], w.M10_mean.iloc[3] * 0.35, "slope 1", fontsize=6.5, color=INK2)
    ax[0].text(x[3], -w.dauc_mean.iloc[3] * 0.25, "slope 2", fontsize=6.5, color=INK2)
    ax[0].set(xscale="log", yscale="log", xlabel=r"lattice spacing $\Delta$", ylabel="rate",
              title="(a) Gross versus net")
    ax[0].legend(loc="upper left")
    ax[0].set_ylim(3e-6, 2)
    clean_x(ax[0], [0.0125, 0.025, 0.05, 0.1, 0.2])
    for k, sp in enumerate(["well-specified", "misspecified"]):
        s = b[b.spec == sp].groupby("Delta").mean(numeric_only=True)
        ax[1].plot(s.index, s.dauc_sd, color=C[k + 2], marker=MK[k + 2], label=f"sd, {sp}")
    ax[1].plot(x, -w.dauc_mean, color=C[1], marker=MK[1], ls="--", lw=1.0, label="|mean|")
    ax[1].set(xscale="log", yscale="log", xlabel=r"lattice spacing $\Delta$",
              ylabel=r"AUC change $\Delta A$", title=r"(b) Mean and fluctuation of $\Delta A$")
    ax[1].legend(loc="upper left")
    clean_x(ax[1], [0.0125, 0.025, 0.05, 0.1, 0.2])
    for k, sp in enumerate(["well-specified", "misspecified"]):
        s = c[c.spec == sp].groupby("N").mean(numeric_only=True)
        ax[2].plot(s.index, s.sd_mc, color=C[k + 2], marker=MK[k + 2], label=f"Monte Carlo, {sp}")
        ax[2].plot(s.index, s.sd_grad, color=C[k + 2], ls="--", lw=0.9)
    s = c[c.spec == "well-specified"].groupby("N").mean(numeric_only=True)
    n = s.index.values
    ax[2].plot(n, s.sd_mc.iloc[0] * (n / n[0]) ** -0.5, color=INK2, lw=0.7, ls=":")
    ax[2].text(n[1] * 1.05, s.sd_mc.iloc[1] * 0.72, r"$N^{-1/2}$", fontsize=6.5, color=INK2)
    ax[2].set(xscale="log", yscale="log", xlabel="sample size $N$",
              ylabel=r"sd of $\Delta A$ ($\Delta=0.05$)", title="(c) Stationarity of the AUC")
    ax[2].legend(loc="lower left", fontsize=6.5)
    clean_x(ax[2], [12500, 50000, 200000], ["12.5k", "50k", "200k"])
    ax[2].text(0.98, 0.97, "dashed: gradient formula", transform=ax[2].transAxes, ha="right", va="top",
               fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{F}/fig2_orders.pdf")
    plt.close(fig)


def fig_real():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
    for ax, nm in zip(axs, DATASETS):
        L = pd.read_csv(f"{R}/e4_{nm}_lattice.csv")
        A = pd.read_csv(f"{R}/e4_{nm}_algorithms.csv")
        Bt = pd.read_csv(f"{R}/e4_{nm}_bootstrap.csv")
        x = L.Delta.values
        ax.fill_between(x, L.M10_lo, L.M10_hi, color=C[0], alpha=0.15, lw=0, label="95% interval")
        ax.plot(x, L.M10_ec, color=C[0], lw=1.2, label="prediction")
        ax.plot(x, L.M10, color=C[0], marker="o", ls="none", label="nearest")
        o = A[(A.K == 10) & (A.method == "OAR")].sort_values("Delta")
        ax.plot(o.Delta, o.M_test, color=C[1], marker="s", ls="none", label="OAR")
        ax.axhline(Bt.Mrefit10.mean(), color=C[2], ls="--", lw=1.0, label="re-estimation")
        ax.set(xscale="log", title=nm, xlabel=r"$\Delta$")
        clean_x(ax, [0.02, 0.05, 0.1, 0.2, 0.35])
        ax.set_ylim(0, 0.8)
    axs[0].set_ylabel("migration, $K=10$ (test)")
    axs[0].legend(loc="upper left", fontsize=6.3, handlelength=1.4)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f"{F}/fig3_real.pdf")
    plt.close(fig)


def fig_boot():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    labels = [("Mrefit10", "re-estimation"), ("Mq10_g2", "integerization, $g=2$"),
              ("Mq10_g5", "integerization, $g=5$")]
    pos = 0
    ticks = []
    for i, nm in enumerate(DATASETS):
        Bt = pd.read_csv(f"{R}/e4_{nm}_bootstrap.csv")
        for k, (col, lab) in enumerate(labels):
            bp = ax.boxplot(Bt[col], positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=C[k]),
                            capprops=dict(color=C[k]))
            for p in bp["boxes"]:
                p.set(facecolor=C[k], alpha=0.35, edgecolor=C[k])
            if i == 0:
                ax.plot([], [], color=C[k], lw=6, alpha=0.5, label=lab)
            pos += 1
        ticks.append(pos - 2)
        pos += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("migration, $K=10$ (test)")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{F}/fig4_bootstrap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_accuracy(); fig_orders(); fig_real(); fig_boot()
    print("figures written")
