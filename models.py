"""models.py - small CIFAR ResNet (3 stages, one basic block each).

A laptop-light stand-in for ResNet-18 (this is stated in the Limitations
section of the report). Dropout before the classifier is only active in the
well-regularized regime (dropout=0.0 elsewhere, which makes it a no-op).
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block of He et al.: two 3x3 convolutions plus a skip connection.

    output = ReLU( BN(conv2( ReLU(BN(conv1(x))) )) + shortcut(x) )
    """

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        # First conv may downsample (stride 2) and change the number of channels.
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        # Second conv keeps the shape. bias=False because BatchNorm has its own shift.
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        # 1x1 projection when the shape changes so the residual add works
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                       nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # identity shortcut when shapes match, otherwise the 1x1 projection
        return F.relu(out + (x if self.short is None else self.short(x)))


class SmallResNet(nn.Module):
    """Stem conv -> 3 residual blocks (widths w, 2w, 4w) -> global avg pool -> dropout -> linear.

    With width=16 the channel widths are 16, 32 and 64, and the feature map
    shrinks 32x32 -> 32x32 -> 16x16 -> 8x8. The output is the raw logits
    (no softmax), which is what the attacks need.
    """

    def __init__(self, width=16, dropout=0.0, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU())
        self.layers = nn.Sequential(
            BasicBlock(width, width, 1),            # stage 1: 32x32, w channels
            BasicBlock(width, width * 2, 2),        # stage 2: 16x16, 2w channels
            BasicBlock(width * 2, width * 4, 2),    # stage 3: 8x8, 4w channels
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 4, n_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)   # global average pooling -> (B, 4w)
        return self.fc(self.drop(x))
