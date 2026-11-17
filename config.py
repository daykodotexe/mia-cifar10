"""config.py - every hyperparameter of the project lives in this one file.

Keeping all settings here means no "magic numbers" are hidden in the other
scripts: to change the pool size, the number of models, the training recipes or
the metrics, only this file has to be edited.

Where it fits in the report: Section 2 (environment), Section 5 (experiment
design) and Section 6 (results).

This is the LAPTOP-LIGHT version of my original plan. The deviations are listed
here and again in the Limitations section of the report:
  * a pool of 10,000 CIFAR-10 images instead of 50,000 (5,000 members per model)
  * a narrow ResNet (WIDTH=16, 3 stages) instead of ResNet-18
  * 9 models per regime (1 target + 8 shadows) instead of 17; targets 0, 1, 2
    are rotated and the other 8 models act as that target's shadow models
    (leave-one-out)
  * no augmentation queries in LiRA

Set the environment variable MIA_QUICK=1 for a 2-minute smoke test (tiny number
of epochs, 3 models, 1 target). Never mix smoke-test output with real results.
"""
import os
import torch

# True only when the smoke test is requested from the shell: $env:MIA_QUICK = "1"
QUICK = os.environ.get("MIA_QUICK") == "1"

# Master seed. Pool selection uses SEED, member masks use SEED + 1000 + i, and
# the weight initialisation of model i uses SEED + i, so a rerun on the same
# machine reproduces the same data splits.
SEED = 0
DATA_DIR = "data"                                  # CIFAR-10 is downloaded here
RESULTS_DIR = "results"                            # all CSVs, scores and figures
RAW_DIR = os.path.join(RESULTS_DIR, "raw")         # one .npz per trained model

# ---- data -------------------------------------------------------------
POOL_SIZE = 10_000      # samples shared by all models (members + non-members)
MEMBER_SIZE = 5_000     # each model trains on a random half of the pool
N_TEST = 2_000          # held-out CIFAR test images for reporting test accuracy
N_VAL = 1_000           # separate test images used ONLY for early stopping
                        # (disjoint from the 2,000 above, so no test leakage)

# ---- models -----------------------------------------------------------
N_MODELS = 3 if QUICK else 9     # models trained per regime
N_TARGETS = 1 if QUICK else 3    # how many of them are attacked (the rest are shadows)
WIDTH = 16                       # base channel width of the small ResNet
BATCH_SIZE = 128
MOMENTUM = 0.9                   # SGD momentum (Nesterov, see train.py)

# ---- training regimes (Section 5 of the report) -----------------------
# Only the recipe differs between regimes; architecture, optimiser, learning
# rate and batch size are the same. The fields mean:
#   epochs          number of passes over the 5,000 member images
#   lr              initial SGD learning rate
#   wd              L2 weight decay
#   aug             "none" | "light" (random crop + flip) | "strong" (crop + flip + cutout)
#   dropout         dropout probability before the classifier
#   label_smoothing cross-entropy label smoothing (0 = off)
#   cosine          True = cosine learning-rate decay, False = constant rate
#   early_stop      True = keep the weights with the lowest validation loss
_EPOCHS = {"overfit": 40, "normal": 30, "regularized": 30}
REGIMES = {
    # No regularisation at all, so the model is free to memorise its members.
    "overfit": dict(epochs=_EPOCHS["overfit"], lr=0.05, wd=0.0, aug="none",
                    dropout=0.0, label_smoothing=0.0, cosine=False, early_stop=False),
    # A standard recipe: mild weight decay, light augmentation, cosine decay, early stopping.
    "normal": dict(epochs=_EPOCHS["normal"], lr=0.05, wd=5e-4, aug="light",
                   dropout=0.0, label_smoothing=0.0, cosine=True, early_stop=True),
    # Everything that reduces memorisation is switched on at once.
    "regularized": dict(epochs=_EPOCHS["regularized"], lr=0.05, wd=5e-3, aug="strong",
                        dropout=0.3, label_smoothing=0.1, cosine=True, early_stop=True),
}
if QUICK:
    # Smoke test: shrink every regime to 2 epochs so the whole pipeline runs in minutes.
    for _cfg in REGIMES.values():
        _cfg["epochs"] = 2

# ---- attacks ----------------------------------------------------------
# The attack's true positive rate (TPR) is reported at these false positive
# rates (1% and 0.1%). Low-FPR numbers are the relevant privacy metric because
# they show whether some members are exposed with high confidence.
FPR_POINTS = (0.01, 0.001)

# ---- device -----------------------------------------------------------
# Use a GPU when one is available, otherwise fall back to the CPU. The reported
# experiments ran on the CPU of a laptop (DEVICE == "cpu").
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"      # Apple-silicon GPU
else:
    DEVICE = "cpu"
