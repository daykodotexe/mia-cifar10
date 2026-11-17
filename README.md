# Membership inference demo (CIFAR-10, small ResNet)

Course project for Introduction to Information Security (final assignment).
Author: Amanov Dayanch.

Question: does a well-generalized model still leak membership information?
Three training regimes (overfit, normal, well-regularized) are attacked with a
loss-threshold attack and with LiRA (fixed variance).

## Main results (CPU laptop run, mean over 3 target models)

| Regime | Train-test gap | Loss-threshold AUC | LiRA TPR@1%FPR (chance = 0.01) |
|---|---|---|---|
| Overfit | 0.369 | 0.742 | 0.0821 |
| Normal | 0.087 | 0.547 | 0.0213 |
| Well-regularized | 0.064 | 0.537 | 0.0183 |

A near-chance AUC and a small train-test gap do not make a model safe: LiRA still
identifies about 2% of the members of the well-generalized models at 1% false
positive rate. This is a small-scale demo (10,000-image pool, small ResNet, 3
targets per regime), so the numbers are not a benchmark. DP-SGD, output
restriction and per-sample analysis were not tested.

## Run (about 32 minutes on a CPU laptop, from this folder)

    pip install -r requirements.txt
    python run_shadow.py     # train 27 models, saves results/raw/*.npz and results/models.csv
    python run_attacks.py    # attacks, saves results/results.csv, results_summary.csv, scores_*.npz
    python plots.py          # saves the three figures in results/

Optional 2-minute smoke test: set the environment variable MIA_QUICK=1 before the
three commands, then delete the output (see the report, Section 3, Step 0).
CIFAR-10 is downloaded automatically into data/ on the first run.

## Files

| File | Role |
|---|---|
| config.py | every hyperparameter in one place |
| data.py | CIFAR-10 loading, fixed pool, per-model member masks |
| models.py | small ResNet (3 residual blocks, width 16) |
| train.py | trains one model for a given regime |
| attacks.py | loss-threshold attack and LiRA (fixed variance) |
| metrics.py | ROC, AUC, TPR at fixed FPR, best-threshold accuracy |
| run_shadow.py | trains all models, saves logits and accuracies |
| run_attacks.py | runs both attacks on each target |
| plots.py | draws the figures |
| mia_results_summary.xlsx | per-model and per-target results (per-model and per-target results) |

## License

MIT, see LICENSE.
