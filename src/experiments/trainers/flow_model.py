import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_emb(t, emb_dim):
    half_dim = emb_dim // 2
    emb_scale = math.log(10000.0) / (half_dim - 1)
    freq = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
    t_freq = t[:, None] * freq[None, :]
    emb_sin = torch.sin(t_freq)
    emb_cos = torch.cos(t_freq)
    emb = torch.cat([emb_sin, emb_cos], dim=1) # [B, emb_dim]
    return emb

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class TimeResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, time_emb_dim=128):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, 2*in_channels)
        self.silu = nn.SiLU()
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.silu(h)

        alpha, beta = self.time_emb_proj(time_emb).chunk(2, dim=1)
        h = h * (1 + alpha[:, :, None, None]) + beta[:, :, None, None]

        h = self.conv1(h)
        h = self.norm2(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, height, width = x.shape
        h = self.norm(x)
        
        q = self.q(h).reshape(b, c, height * width)
        k = self.k(h).reshape(b, c, height * width)
        v = self.v(h).reshape(b, c, height * width)
        
        w = torch.einsum('bli,blc->bic', q, k)/(self.in_channels ** 0.5)
        w = torch.nn.functional.softmax(w, dim=-1)
        
        h = torch.einsum('bcl,bil->bci', v, w).reshape(b, c, height, width)
        h = self.proj_out(h)

        return x + h

class TimeResNet(nn.Module):
    def __init__(
        self, 
        channels, 
        out_channels, 
        channels_mult, 
        num_res_blocks, 
        attn_resolutions, 
        dropout, 
        in_channels, 
        resolution, 
        resamp_with_conv
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        channels_mult = tuple(channels_mult)
        self.time_embed_dim = channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim*2),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim*2, self.time_embed_dim)
        )
        self.silu = nn.SiLU()
        
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_channels_mult = (1,) + channels_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_channels_mult[i_level]
            block_out = channels * channels_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    TimeResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        time_emb_dim=self.time_embed_dim
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down_module = nn.Module()
            down_module.block = block
            down_module.attn = attn

            if i_level != self.num_resolutions - 1:
                down_module.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down_module)

        self.mid = nn.Module()
        self.mid.block_1 = TimeResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout, time_emb_dim=self.time_embed_dim
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = TimeResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout, time_emb_dim=self.time_embed_dim
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            skip_in = channels * channels_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = channels * in_channels_mult[i_level]

                block.append(
                    TimeResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        dropout=dropout,
                        time_emb_dim=self.time_embed_dim
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up_module = nn.Module()
            up_module.block = block
            up_module.attn = attn
            if i_level != 0:
                up_module.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2

            self.up.insert(0, up_module)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        if t.dim() > 1:
            t = t.reshape(t.shape[0])
        time_emb = sinusoidal_time_emb(t, self.time_embed_dim)
        time_emb = self.time_mlp(time_emb)

        h = self.conv_in(x)
        hs = [h]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, time_emb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
                hs.append(h)

        h = self.mid.block_1(h, time_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_emb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), time_emb)

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h
