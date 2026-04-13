import torch
import torch.nn as nn
import torch.nn.functional as F


class SCFM(nn.Module):
    """
    Spatial-Channel Feature Modulator (SCFM)

    双分支注意力：
    - Spatial Attention（空间）
    - Channel Attention（通道）
    """

    def __init__(self, c1):
        super(SCFM, self).__init__()

        # ===== Spatial Attention =====
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

        # ===== Channel Attention =====
        self.channel_conv1 = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.channel_conv2 = nn.Conv2d(c1, c1, kernel_size=1, bias=False)

        # ===== Output Fusion =====
        self.conv_spatial = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.conv_channel = nn.Conv2d(c1, c1, kernel_size=1, bias=False)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        # ===============================
        # 1️⃣ Spatial Attention Branch
        # ===============================
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        spatial_weight = torch.sigmoid(
            self.spatial_conv(torch.cat([max_pool, avg_pool], dim=1))
        )

        spatial_out = spatial_weight * x

        # ===============================
        # 2️⃣ Channel Attention Branch
        # ===============================
        ch = self.act(self.channel_conv1(x))
        ch = self.channel_conv2(ch)

        max_pool_c = F.adaptive_max_pool2d(ch, 1)
        avg_pool_c = F.adaptive_avg_pool2d(ch, 1)

        channel_weight = torch.sigmoid(max_pool_c + avg_pool_c)

        channel_out = channel_weight * x

        # ===============================
        # 3️⃣ Fusion
        # ===============================
        out = self.conv_spatial(spatial_out) + self.conv_channel(channel_out)

        return out