"""metrics.py - AUC, TPR at a fixed FPR, best-threshold accuracy and ROC curve (report Section 4, "Metrics")."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

import config


def roc(scores, is_member):
    """ROC curve of an attack: returns (fpr, tpr) arrays.

    scores     one attack score per pool image (higher = more likely member)
    is_member  boolean ground truth: was the image a member of the target model?
    """
    fpr, tpr, _ = roc_curve(is_member, scores)
    return fpr, tpr


def tpr_at_fpr(fpr, tpr, target):
    """True positive rate at the false positive rate `target`, by linear interpolation."""
    # fpr is sorted ascending, so interpolation is valid
    return float(np.interp(target, fpr, tpr))


def evaluate_attack(scores, is_member):
    """Return dict of metrics. best_acc picks the threshold USING the labels,
    so it is optimistic - say so in the document.

    Keys: auc, best_acc, and one "tpr@<f>%fpr" entry for each f in config.FPR_POINTS
    (for example "tpr@1%fpr" and "tpr@0.1%fpr").
    """
    fpr, tpr = roc(scores, is_member)
    # best_acc: balanced accuracy 0.5*(TPR + TNR) at its best threshold; TNR = 1 - FPR.
    # Members and non-members are 50/50 here, so this equals plain accuracy.
    out = dict(auc=float(roc_auc_score(is_member, scores)),
               best_acc=float(np.max(0.5 * (tpr + 1 - fpr))))
    for f in config.FPR_POINTS:
        out[f"tpr@{f*100:g}%fpr"] = tpr_at_fpr(fpr, tpr, f)
    return out
