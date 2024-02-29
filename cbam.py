import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_attention = torch.sigmoid(self.fc(avg_out)).view(x.size(0), x.size(1), 1, 1)
        return x * channel_attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        max_pool_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([max_pool_out, avg_pool_out], dim=1)
        spatial_attention = F.relu(self.conv(spatial_attention))
        return torch.sigmoid(spatial_attention)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x_channel = self.channel_attention(x)
        x_spatial = self.spatial_attention(x_channel)
        return x_spatial * x_channel
