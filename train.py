"""train.py - train ONE model for a given regime on a given data subset.

Where it fits in the report: Section 5 (training regimes). Data augmentation is
done on-device, in whole batches, so it stays fast on a laptop CPU.

Public functions:
  augment()      apply the augmentation chosen by the regime to a batch
  get_logits()   raw model outputs on un-augmented inputs
  evaluate()     cross-entropy loss and accuracy
  train_model()  the training loop (SGD, optional cosine decay and early stopping)
"""
import torch
import torch.nn.functional as F

import config
from models import SmallResNet


def _random_crop(x, pad=4):
    """Per-sample random crop of a reflect-padded batch, via advanced indexing.

    Each image is padded by `pad` pixels on every side and a random 32x32 window
    is cut out, i.e. the image is shifted by up to +-pad pixels. Doing this with
    index tensors avoids a slow Python loop over the batch.
    """
    B, _, H, W = x.shape
    xp = F.pad(x, (pad,) * 4, mode="reflect")
    # random top-left corner of the crop window, one per image
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    rows = (i[:, None] + ar)[:, :, None]      # (B, H, 1): row indices of each window
    cols = (j[:, None] + ar)[:, None, :]      # (B, 1, W): column indices of each window
    b = torch.arange(B, device=x.device)[:, None, None]
    return xp[b, :, rows, cols].permute(0, 3, 1, 2)  # advanced dims come first -> B,H,W,C


def _random_flip(x):
    """Flip each image left-right with probability 0.5."""
    m = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(m[:, None, None, None], x.flip(3), x)


def _cutout(x, size=8):
    """Zero out one random size x size square per image (strong-aug regime only)."""
    B, _, H, W = x.shape
    # random centre of the square for every image
    cy = torch.randint(0, H, (B,), device=x.device)
    cx = torch.randint(0, W, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    my = (ar[None, :] - cy[:, None]).abs() < size // 2    # rows inside the square
    mx = (ar[None, :] - cx[:, None]).abs() < size // 2    # columns inside the square
    hole = my[:, :, None] & mx[:, None, :]                # (B, H, W) boolean mask of the hole
    return x * (~hole)[:, None].float()                   # set the hole to 0, keep the rest


def augment(x, mode):
    """Augment a batch. mode: "none" | "light" (crop + flip) | "strong" (crop + flip + cutout)."""
    if mode == "none":
        return x
    x = _random_flip(_random_crop(x))
    return _cutout(x) if mode == "strong" else x


@torch.no_grad()
def get_logits(model, x, bs=1000):
    """Raw logits on un-augmented inputs, as float32 on CPU.

    The model is put in eval mode (dropout off, BatchNorm uses running
    statistics) and inputs are processed in chunks of `bs` to bound memory.
    These logits are exactly what a black-box attacker would observe.
    """
    model.eval()
    out = [model(x[k:k + bs].to(config.DEVICE)).float().cpu() for k in range(0, len(x), bs)]
    return torch.cat(out)


def evaluate(model, x, y):
    """Plain cross-entropy (no label smoothing) and accuracy."""
    logits = get_logits(model, x)
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train_model(regime, x, y, val_x, val_y, seed):
    """Train one SmallResNet on (x, y) with the recipe config.REGIMES[regime].

    x, y        the member images/labels of THIS model (a random half of the pool)
    val_x/val_y separate images used only for early stopping
    seed        seeds weight initialisation and batch order, so runs are repeatable
    Returns the trained model (with the best-validation weights if early stopping is on).
    """
    cfg = config.REGIMES[regime]
    torch.manual_seed(seed)
    model = SmallResNet(config.WIDTH, cfg["dropout"]).to(config.DEVICE)
    # SGD with Nesterov momentum; wd is the L2 weight-decay strength of the regime.
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=config.MOMENTUM,
                          weight_decay=cfg["wd"], nesterov=True)
    # Cosine decay lowers the learning rate smoothly to ~0 over all epochs.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"]) if cfg["cosine"] else None
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    n = len(x)

    best_val, best_state = float("inf"), None
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=config.DEVICE)   # new random batch order each epoch
        for k in range(0, n, config.BATCH_SIZE):
            idx = perm[k:k + config.BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            # forward pass on an augmented batch; label smoothing is part of the regime
            loss = F.cross_entropy(model(augment(x[idx], cfg["aug"])), y[idx],
                                   label_smoothing=cfg["label_smoothing"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched:
            sched.step()   # one scheduler step per epoch
        if cfg["early_stop"]:
            # keep the weights with the lowest validation loss
            v, _ = evaluate(model, val_x, val_y)
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
