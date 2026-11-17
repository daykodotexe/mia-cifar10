"""run_attacks.py - run the loss-threshold attack and LiRA on each regime / target (experiments E2, E3).

Where it fits in the report: Section 3 ("Step 2").

Target t is attacked using the OTHER models of the same regime as shadows
(leave-one-out), so every target is evaluated on a model the shadows never
include. Each attack scores all 10,000 pool images (5,000 members and 5,000
non-members of the target) and the scores are then evaluated against the
target's true membership mask.

Output: results/results.csv           one row per (regime, target, attack)
        results/results_summary.csv   mean and standard deviation over the targets
        results/scores_{regime}_t{t}.npz   per-image scores, used by plots.py
"""
import os

import numpy as np
import pandas as pd

import config
from attacks import phi_stat, loss_attack, lira_fixed_var, true_class_loss
from data import load_cifar
from metrics import evaluate_attack


def main():
    y = load_cifar()["pool_y"].numpy()      # true labels of the 10,000 pool images
    rows = []
    for regime in config.REGIMES:
        # load the saved outputs of all models of this regime (written by run_shadow.py)
        zs = [np.load(os.path.join(config.RAW_DIR, f"{regime}_{i}.npz")) for i in range(config.N_MODELS)]
        logits = np.stack([z["logits"] for z in zs])      # (M,P,10)
        masks = np.stack([z["mask"] for z in zs])          # (M,P)
        phi = np.stack([phi_stat(l, y) for l in logits])   # (M,P) LiRA statistic per model and image
        for t in range(config.N_TARGETS):
            # every model except the target acts as a shadow model
            shadows = [j for j in range(config.N_MODELS) if j != t]
            truth = masks[t]  # only used for EVALUATION below, never for fitting
            s_loss = loss_attack(logits[t], y)
            # LiRA only receives the SHADOW masks; the target's mask stays hidden
            s_lira = lira_fixed_var(phi[t], phi[shadows], masks[shadows])
            # save the per-image scores so that plots.py can draw ROC curves and histograms
            np.savez(os.path.join(config.RESULTS_DIR, f"scores_{regime}_t{t}.npz"),
                     s_loss=s_loss, s_lira=s_lira, member=truth,
                     loss=true_class_loss(logits[t].astype(np.float64), y))
            for name, s in (("loss", s_loss), ("lira", s_lira)):
                r = dict(regime=regime, target=t, attack=name,
                         test_acc=float(zs[t]["test_acc"]),
                         gap=float(zs[t]["train_acc"] - zs[t]["test_acc"]))
                r.update(evaluate_attack(s, truth))     # adds auc, best_acc, tpr@1%fpr, tpr@0.1%fpr
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "results.csv"), index=False)
    # mean and standard deviation across the targets of each (regime, attack) pair
    summ = df.groupby(["regime", "attack"]).agg(["mean", "std"]).drop(columns="target")
    summ.to_csv(os.path.join(config.RESULTS_DIR, "results_summary.csv"))


if __name__ == "__main__":
    main()
