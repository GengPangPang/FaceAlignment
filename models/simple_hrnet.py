import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        return out


class HRNetSmall(nn.Module):
    """
    输入:  [B, 3, 256, 256]
    输出:  [B, 68, 64, 64]
    """

    def __init__(self, num_landmarks=68):
        super().__init__()

        # stem: 256 -> 128 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # high-resolution branch: 64x64
        self.branch1 = nn.Sequential(
            BasicBlock(64, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )

        # low-resolution branch: 32x32
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

        # lower-resolution branch: 16x16
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        # fuse all branches to 64x64
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + 64 + 128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            BasicBlock(128, 128),
            BasicBlock(128, 128),
        )

        self.head = nn.Conv2d(128, num_landmarks, 1)

    def forward(self, x):
        x = self.stem(x)      # [B, 64, 64, 64]

        x1 = self.branch1(x)  # [B, 32, 64, 64]

        x2 = self.down1(x1)   # [B, 64, 32, 32]
        x2 = self.branch2(x2)

        x3 = self.down2(x2)   # [B, 128, 16, 16]
        x3 = self.branch3(x3)

        x2_up = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x3_up = F.interpolate(x3, size=x1.shape[2:], mode="bilinear", align_corners=False)

        out = torch.cat([x1, x2_up, x3_up], dim=1)
        out = self.fuse(out)

        heatmaps = self.head(out)
        return heatmaps