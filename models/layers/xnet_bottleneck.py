import random
import torch.nn as nn

from models.layers import SEBlock


class XNetBottleneck(nn.Module):
    """XNet bottleneck block, similar to a pre-activation ResNeXt bottleneck block.

    Based on the paper:
    "Aggregated Residual Transformations for Deep Nerual Networks"
    by Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    (https://arxiv.org/abs/1611.05431).
    """

    expansion = 2

    def __init__(self, in_channels, channels, cardinality, survival_prob=1., stride=1, down_sample=None):
        super(XNetBottleneck, self).__init__()
        mid_channels = cardinality * int(channels / cardinality)
        out_channels = channels * self.expansion
        self.survival_prob = survival_prob

        self.down_sample = down_sample

        self.norm1 = nn.GroupNorm(in_channels // 16, in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)

        self.norm2 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)

        self.norm3 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.se_block = SEBlock(out_channels, reduction=16)

    def forward(self, x):
        x_skip = x if self.down_sample is None else self.down_sample(x)

        # Stochastic depth dropout
        if self.training and random.random() > self.survival_prob:
            return x_skip

        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.norm3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.se_block(x)

        x += x_skip

        return x
