import torch
import torch.nn as nn


class XNetDecoder(nn.Module):
    """Decoder (up-sampling layer) for XNet"""
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=4, stride=2):
        super(XNetDecoder, self).__init__()

        self.norm1 = nn.GroupNorm(in_channels // 16, in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=1)

        self.norm3 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, 3, padding=1)

    def forward(self, x, x_skip=None):
        if x_skip is not None:
            x = torch.cat((x, x_skip), dim=1)

        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.norm3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        # TODO: Try making this a residual block?

        return x
