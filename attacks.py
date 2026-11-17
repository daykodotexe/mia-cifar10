"""attacks.py - loss-threshold attack (Yeom et al.) and LiRA with fixed variance (Carlini et al.).

Where it fits in the report: Section 4 (principles) and Section 5.

All functions take numpy arrays and NEVER see the target's membership mask; only
the shadow models' masks are used to fit the attack. The target's mask is used
afterwards, in metrics.py, purely to SCORE the attack.

Every attack returns one score per pool image: a HIGHER score means "more
likely to be a member of the target model".
"""
import numpy as np
from scipy.special import logsumexp


def true_class_loss(logits, y):
    """Cross-entropy of the true class, numerically stable. logits: (P,10), y: (P,)

    loss = -log softmax(z)_y = logsumexp(z) - z_y   (P = number of pool images)
    """
    return logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]


def phi_stat(logits, y):
    """LiRA confidence statistic log(p/(1-p)) computed stably as
    z_y - logsumexp(z_other), which avoids inf when p is ~1.

    Here p is the softmax probability of the true class y. Writing p = e^{z_y} / sum_k e^{z_k}
    gives log(p/(1-p)) = z_y - log(sum_{k != y} e^{z_k}), the true-class logit minus the
    log-sum-exp of the other logits. This statistic is roughly Gaussian across
    models trained on the same record, which is what LiRA's Gaussian fit assumes.
    """
    logits = logits.astype(np.float64)
    z_y = logits[np.arange(len(y)), y]
    other = logits.copy()
    other[np.arange(len(y)), y] = -np.inf     # remove the true class from the sum
    return z_y - logsumexp(other, axis=1)


def loss_attack(target_logits, y):
    """Score = -loss: lower loss -> more likely a member.

    This is the global-threshold attack of Yeom et al.: predict "member" when the
    loss is below a threshold. Using -loss as the score and sweeping the threshold
    over all values gives the ROC curve.
    """
    return -true_class_loss(target_logits.astype(np.float64), y)


def _log_gauss(x, mu, sigma):
    """Log-density of N(mu, sigma^2) at x, WITHOUT the constant -0.5*log(2*pi).

    The dropped constant is identical for the IN and OUT Gaussians, so it cancels
    in the log-likelihood ratio computed by lira_fixed_var().
    """
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)


def lira_fixed_var(phi_target, phi_shadow, shadow_masks):
    """phi_target: (P,)  phi_shadow: (S,P)  shadow_masks: (S,P) bool (True = sample was IN).

    Per sample: mean of phi over shadows where it was IN / OUT.
    Variance: ONE global sigma_in and sigma_out (the paper's fixed-variance
    variant), because 8 shadows give ~4 observations per sample, too few for a
    per-sample std. Score = log N(phi;mu_in,s_in) - log N(phi;mu_out,s_out).

    This uses BOTH the IN and the OUT distributions, which is the "online"
    version of LiRA in the terminology of the paper.
    """
    # inn[s, j] = pool image j was a training member of shadow model s; out is the complement
    inn, out = shadow_masks, ~shadow_masks

    def masked_mean(m):
        """Per-image mean of phi over the shadow models selected by the boolean matrix m."""
        cnt = m.sum(0)                                   # number of shadows per image
        mu = (phi_shadow * m).sum(0) / np.maximum(cnt, 1)
        # sample with zero IN (or OUT) observations: fall back to the global mean
        glob = phi_shadow[m].mean()
        return np.where(cnt > 0, mu, glob)

    mu_in, mu_out = masked_mean(inn), masked_mean(out)
    # Global standard deviations: the spread of phi around its per-image mean,
    # pooled over all images and all shadows. The 1e-8 avoids division by zero.
    s_in = (phi_shadow - mu_in)[inn].std() + 1e-8
    s_out = (phi_shadow - mu_out)[out].std() + 1e-8
    # log-likelihood ratio: positive when phi_target looks more like the IN distribution
    return _log_gauss(phi_target, mu_in, s_in) - _log_gauss(phi_target, mu_out, s_out)
