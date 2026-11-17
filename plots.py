"""plots.py - all figures, drawn from the saved scores and CSVs (no retraining).

Where it fits in the report: Section 3 ("Step 3") and Section 6 (results).

  fig_roc_loglog.png     E2+E3: loss attack vs LiRA, log-log ROC, one panel per regime
  fig_gap_vs_attack.png  E4: attack strength vs generalization gap (mean +- std over targets)
  fig_loss_hist.png      E5: loss histograms, members vs non-members
"""
import os

import matplotlib
matplotlib.use("Agg")          # draw to files only, no window needed (works on any machine)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from metrics import roc

R = config.RESULTS_DIR
regimes = list(config.REGIMES)


def load_scores(regime, t=0):
    """Load the per-image scores saved by run_attacks.py (default: target 0)."""
    return np.load(os.path.join(R, f"scores_{regime}_t{t}.npz"))


def fig_roc():
    """Log-log ROC curves. The log axes stretch the low-FPR region, where privacy leaks show up."""
    fig, axs = plt.subplots(1, len(regimes), figsize=(4.5 * len(regimes), 4.5), sharey=True)
    for ax, reg in zip(axs, regimes):
        z = load_scores(reg)
        for key, label in (("s_loss", "loss threshold"), ("s_lira", "LiRA (fixed var)")):
            fpr, tpr = roc(z[key], z["member"])
            ax.plot(fpr, tpr, label=label)
        ax.plot([1e-3, 1], [1e-3, 1], "k--", lw=0.8, label="chance")   # TPR = FPR diagonal
        ax.set(xscale="log", yscale="log", xlim=(1e-3, 1), ylim=(1e-3, 1), title=reg,
               xlabel="False positive rate")
    axs[0].set_ylabel("True positive rate")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_roc_loglog.png"), dpi=150)


def fig_gap():
    """Attack strength (AUC and TPR@1%FPR) against the generalization gap, one point per regime and attack."""
    df = pd.read_csv(os.path.join(R, "results.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axs, ("auc", "tpr@1%fpr")):
        for atk, marker in (("loss", "o"), ("lira", "s")):
            # mean and std over the targets of each regime, in regime order
            g = df[df.attack == atk].groupby("regime")[["gap", metric]].agg(["mean", "std"]).loc[regimes]
            ax.errorbar(g[("gap", "mean")], g[(metric, "mean")], yerr=g[(metric, "std")].fillna(0),
                        fmt=marker, capsize=3, label=atk)
            for reg, (gx, gy) in zip(regimes, zip(g[("gap", "mean")], g[(metric, "mean")])):
                ax.annotate(reg, (gx, gy), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set(xlabel="generalization gap (train acc - test acc)", ylabel=metric)
        if metric == "auc":
            ax.axhline(0.5, color="k", ls="--", lw=0.8)    # AUC of a random guess
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_gap_vs_attack.png"), dpi=150)


def fig_hist():
    """Histograms of log10(true-class loss) for members vs non-members of target 0.

    The more the two histograms overlap, the less a single loss threshold can
    separate them.
    """
    fig, axs = plt.subplots(1, len(regimes), figsize=(4.5 * len(regimes), 3.8))
    for ax, reg in zip(axs, regimes):
        z = load_scores(reg)
        ll = np.log10(z["loss"] + 1e-8)         # 1e-8 avoids log10(0)
        bins = np.linspace(ll.min(), ll.max(), 50)
        ax.hist(ll[z["member"]], bins, alpha=0.6, label="members", density=True)
        ax.hist(ll[~z["member"]], bins, alpha=0.6, label="non-members", density=True)
        ax.set(title=reg, xlabel="log10(loss)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "fig_loss_hist.png"), dpi=150)


if __name__ == "__main__":
    fig_roc()
    fig_gap()
    fig_hist()
    print("saved figures to", R)
