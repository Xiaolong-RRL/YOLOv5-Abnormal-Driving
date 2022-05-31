import torch
from torch import nn

c1, c2 = 1, 2
nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    groups=1,
    bias=True,
    padding_mode='zeros')
