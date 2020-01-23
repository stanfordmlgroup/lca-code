import torch.nn as nn

from models.layers.xnet import XNetBottleneck


class XNetEncoder(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, cardinality, block_idx, total_blocks, stride=1):
        super(XNetEncoder, self).__init__()

        # Get down-sampling layer for first bottleneck
        down_sample = None
        if stride != 1 or in_channels != channels * XNetBottleneck.expansion:
            down_sample = nn.Sequential(
                nn.Conv3d(in_channels, channels * XNetBottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(channels * XNetBottleneck.expansion // 16, channels * XNetBottleneck.expansion))

        # Get XNet blocks
        survival_prob = self._get_survival_prob(block_idx, total_blocks)
        xnet_blocks = [XNetBottleneck(in_channels, channels, cardinality, survival_prob, stride, down_sample)]

        for i in range(1, num_blocks):
            survival_prob = self._get_survival_prob(block_idx + i, total_blocks)
            xnet_blocks += [XNetBottleneck(channels * XNetBottleneck.expansion, channels, cardinality, survival_prob)]
        self.xnet_blocks = nn.Sequential(*xnet_blocks)

    @staticmethod
    def _get_survival_prob(block_idx, total_blocks, p_final=0.5):
        """Get survival probability for stochastic depth. Uses linearly decreasing
        survival probability as described in "Deep Networks with Stochastic Depth".

        Args:
            block_idx: Index of residual block within entire network.
            total_blocks: Total number of residual blocks in entire network.
            p_final: Survival probability of the final layer.
        """
        return 1. - block_idx / total_blocks * (1. - p_final)

    def forward(self, x):
        x = self.xnet_blocks(x)

        return x
