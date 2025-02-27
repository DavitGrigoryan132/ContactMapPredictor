import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A standard 2D residual block:
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        residual = out
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class DownsamplingBranch(nn.Module):
    """
    Downsampling branch processes the pairwise embeddings tensor and fused tensor.

    Input: Tensor of shape (batch, D, L, L)
    Steps:
      1. 1x1 Conv to reduce channels from D to 256.
      2. Apply several ResidualBlocks.
      3. Final 1x1 Conv to project features to desired output channels.

    Output: Tensor of shape (batch, out_channels, L, L)
    """
    def __init__(self, in_channels=640, num_blocks=10, out_channels=128):
        super(DownsamplingBranch, self).__init__()

        step = (in_channels - out_channels) // (num_blocks - 1)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *([ResidualBlock(in_channels - i * step, in_channels - (i + 1) * step) for i in range(num_blocks - 1)] +
              [ResidualBlock(in_channels - (num_blocks - 1) * step, out_channels)])
        )

    def forward(self, x):
        # x: (batch, D, L, L)
        out = self.res_blocks(x)
        return out