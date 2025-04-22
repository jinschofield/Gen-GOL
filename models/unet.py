import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.res_conv = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.res_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1,2,4), time_emb_dim=256):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        # Downsampling
        channels = [base_channels * m for m in channel_mults]
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.downs.append(ResidualBlock(prev_ch, ch, time_emb_dim))
            self.downs.append(nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = ResidualBlock(prev_ch, prev_ch, time_emb_dim)

        # Upsampling
        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=4, stride=2, padding=1))
            self.ups.append(ResidualBlock(prev_ch + ch, ch, time_emb_dim))
            prev_ch = ch

        # Final
        self.final_block = nn.Sequential(
            ResidualBlock(prev_ch, prev_ch, time_emb_dim),
            nn.Conv2d(prev_ch, in_channels, kernel_size=1)
        )

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B,)
        t_emb = self.time_emb(t)
        h = x
        skips = []
        # Down path
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:
                h = layer(h)
        # Bottleneck
        h = self.bottleneck(h, t_emb)
        # Up path
        for layer in self.ups:
            if isinstance(layer, nn.ConvTranspose2d):
                h = layer(h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
        return self.final_block(h)
