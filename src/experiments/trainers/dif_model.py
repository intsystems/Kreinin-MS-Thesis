import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DifTimeResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.silu = nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)

        h = h + self.temb_proj(self.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
            
        return x+h


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


class DiffusionModel(nn.Module):
    def __init__(
        self, 
        channels, 
        in_channels, 
        out_channels, 
        channels_mult, 
        num_res_blocks, 
        attn_resolutions, 
        dropout, 
        resolution, 
        resamp_with_conv
    ):
        super().__init__()
        self.channels = channels
        self.temb_channels = channels*4
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(channels, self.temb_channels),
            torch.nn.Linear(self.temb_channels, self.temb_channels),
        ])
        self.silu = nn.SiLU()

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        curr_res = resolution
        in_channels_mult = (1,)+tuple(channels_mult)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = channels*in_channels_mult[i_level]
            block_out = channels*channels_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(
                    DifTimeResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_channels,
                        dropout=dropout
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = DifTimeResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_channels,dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = DifTimeResnetBlock(in_channels=block_in,out_channels=block_in,temb_channels=self.temb_channels,dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = channels*channels_mult[i_level]
            skip_in = channels*channels_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = channels*in_channels_mult[i_level]
                block.append(
                    DifTimeResnetBlock(
                        in_channels=block_in+skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_channels,
                        dropout=dropout
                    )
                )
                block_in = block_out
                
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        temb = get_timestep_embedding(t, self.channels)
        temb = self.temb.dense[0](temb)
        temb = self.silu(temb)
        temb = self.temb.dense[1](temb)

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h
