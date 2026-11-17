"""run_shadow.py - train N_MODELS per regime and save raw logits + statistics.

Where it fits in the report: Section 3 (execution steps, "Step 1").

For every regime (overfit, normal, regularized) it trains N_MODELS models, each
on its own random half of the pool, and stores for each model:
  logits      outputs on ALL pool images (members and non-members)  - what an attacker sees
  mask        which pool images were this model's training members   - ground truth
  train/test  accuracy and cross-entropy loss

The script is resumable: models whose file already exists are skipped, so an
interrupted run can simply be started again.
Output: results/raw/{regime}_{i}.npz and results/models.csv (experiment E1).
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import config
from data import load_cifar, make_masks
from train import train_model, get_logits, evaluate


def main():
    os.makedirs(config.RAW_DIR, exist_ok=True)
    print(f"device={config.DEVICE} quick={config.QUICK}")
    d = load_cifar()
    masks = make_masks()          # (N_MODELS, POOL_SIZE) membership matrix
    rows = []                     # one summary row per model, written to models.csv
    for regime in config.REGIMES:
        for i in range(config.N_MODELS):
            path = os.path.join(config.RAW_DIR, f"{regime}_{i}.npz")
            if not os.path.exists(path):          # skip models that are already trained
                t0 = time.time()
                m = torch.from_numpy(masks[i])    # boolean mask of model i's training images
                # train on the members only; seed makes the run repeatable
                model = train_model(regime, d["pool_x"][m], d["pool_y"][m],
                                    d["val_x"], d["val_y"], seed=config.SEED + i)
                # logits on the whole pool: members AND non-members
                logits = get_logits(model, d["pool_x"]).numpy()
                # train metrics on the members, test metrics on held-out CIFAR test images
                tr_loss, tr_acc = evaluate(model, d["pool_x"][m], d["pool_y"][m])
                te_loss, te_acc = evaluate(model, d["test_x"], d["test_y"])
                np.savez(path, logits=logits, mask=masks[i], train_acc=tr_acc,
                         test_acc=te_acc, train_loss=tr_loss, test_loss=te_loss)
                print(f"[{regime} {i}] train={tr_acc:.3f} test={te_acc:.3f} "
                      f"gap={tr_acc-te_acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
            z = np.load(path)
            # generalization gap = train accuracy - test accuracy
            rows.append(dict(regime=regime, model=i, train_acc=float(z["train_acc"]),
                             test_acc=float(z["test_acc"]),
                             gap=float(z["train_acc"] - z["test_acc"]),
                             train_loss=float(z["train_loss"]), test_loss=float(z["test_loss"])))
    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, "models.csv"), index=False)


if __name__ == "__main__":
    main()
