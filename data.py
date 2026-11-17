"""data.py - CIFAR-10 loading and member / non-member splits.

Where it fits in the report: Section 5 (data design).

Idea: there is ONE fixed pool of 10,000 CIFAR-10 training images. Every model i
trains on a random 50% subset of that pool (its "members"); the other half of
the pool are its "non-members". The boolean mask masks[i] records which pool
images were members of model i. It is the ground truth used to score the
attacks and must never be used to fit an attack on its own target.
"""
import numpy as np
import torch
from torchvision.datasets import CIFAR10

import config

# Per-channel mean and standard deviation of CIFAR-10 (R, G, B), used to
# normalise every image to roughly zero mean and unit variance.
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def _to_tensor(images_uint8):
    """Convert a numpy batch of images into a normalised float tensor.

    N x 32 x 32 x 3 uint8 (values 0..255)  ->  N x 3 x 32 x 32 float32,
    scaled to [0, 1] and then normalised with the CIFAR-10 mean and std.
    """
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (x - _MEAN) / _STD


def load_cifar():
    """Return a dict with the pool (x, y), test (x, y) and validation (x, y) sets.

    Everything stays in memory (10,000 + 3,000 small images), so no data loader
    is needed. CIFAR-10 is downloaded into config.DATA_DIR on the first call.
    """
    tr = CIFAR10(config.DATA_DIR, train=True, download=True)
    te = CIFAR10(config.DATA_DIR, train=False, download=True)

    # Fixed pool: the same images for every model and every regime (seeded),
    # chosen as the first POOL_SIZE entries of a random permutation of the
    # 50,000 CIFAR-10 training images.
    rng = np.random.RandomState(config.SEED)
    pool_idx = rng.permutation(len(tr.data))[: config.POOL_SIZE]

    pool_x = _to_tensor(tr.data[pool_idx])
    pool_y = torch.tensor(np.array(tr.targets)[pool_idx])

    test_x_all = _to_tensor(te.data)
    test_y_all = torch.tensor(te.targets)
    n_t, n_v = config.N_TEST, config.N_VAL
    return dict(
        pool_x=pool_x, pool_y=pool_y,
        test_x=test_x_all[:n_t], test_y=test_y_all[:n_t],
        # val is disjoint from test so early stopping never sees the test images
        val_x=test_x_all[n_t:n_t + n_v], val_y=test_y_all[n_t:n_t + n_v],
    )


def make_masks():
    """Build the membership matrix: masks[i, j] = True if pool sample j is a member of model i.

    Deterministic: model i always gets the same random half of the pool because
    its generator is seeded with SEED + 1000 + i. Shape: (N_MODELS, POOL_SIZE).
    """
    masks = np.zeros((config.N_MODELS, config.POOL_SIZE), dtype=bool)
    for i in range(config.N_MODELS):
        rs = np.random.RandomState(config.SEED + 1000 + i)
        # shuffle the pool indices and mark the first MEMBER_SIZE of them as members
        masks[i, rs.permutation(config.POOL_SIZE)[: config.MEMBER_SIZE]] = True
    return masks
