import torch.nn as nn
import torch.nn.functional as F


class UpsamplingResidualBlock(nn.Module):
    """
    A standard residual block for structural features.
    """
    def __init__(self, channels):
        super(UpsamplingResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class UpsamplingBranch(nn.Module):
    """
    Structural branch to process template-derived structural features.

    Input: Tensor of shape (batch, K * 3, L, L)
      - K * 3 channels: derived from K templates (each with 3 channels: distance, θ, φ)

    Processing:
      1. Initial 1x1 convolution to project the K * 3 channels to a higher dimension.
      2. A stack of residual blocks to learn spatial/geometric patterns.
      3. Final 1x1 convolution to produce an output tensor for fusion.

    Output: Tensor of shape (batch, out_channels, L, L)
    """
    def __init__(self, in_channels=15, mid_channels=64, num_blocks=3, out_channels=64):
        super(UpsamplingBranch, self).__init__()
        # Initial projection: increase channels from K * 3 to mid_channels
        self.initial_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.initial_bn   = nn.BatchNorm2d(mid_channels)
        self.relu         = nn.ReLU(inplace=True)

        # Stack a few residual blocks to refine features
        self.res_blocks = nn.Sequential(
            *[UpsamplingResidualBlock(mid_channels) for _ in range(num_blocks)]
        )

        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.final_bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x shape: (batch, K * 3, L, L)
        out = self.initial_conv(x)   # -> (batch, mid_channels, L, L)
        out = self.initial_bn(out)
        out = self.relu(out)

        out = self.res_blocks(out)   # -> (batch, mid_channels, L, L)

        out = self.final_conv(out)   # -> (batch, out_channels, L, L)
        out = self.final_bn(out)
        out = self.relu(out)
        return out