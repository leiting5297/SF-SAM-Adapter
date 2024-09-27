# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn.functional as F
import torch
import torch.nn as nn

from typing import Type

from einops import rearrange, repeat


class SAAdapter(nn.Module):

    def __init__(self, dim, h, w, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.norm1 = nn.LayerNorm(768)

        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 768)
        self.l1 = nn.Linear(64, 16)
        self.l2 = nn.Linear(16, 64)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = rearrange(x, 'b h w c -> b c w h').contiguous()
        row_pooled_tensor = torch.mean(x, dim=(0, 1, 2)).unsqueeze(0).unsqueeze(0)
        col_pooled_tensor = torch.mean(x, dim=(0, 1, 3)).unsqueeze(0).unsqueeze(-1)
        row_pooled_tensor = row_pooled_tensor.view(1, -1)
        col_pooled_tensor = col_pooled_tensor.view(1, -1)
        row_pooled_tensor = self.l1(row_pooled_tensor)
        row_pooled_tensor = F.gelu(row_pooled_tensor)
        row_pooled_tensor = self.l2(row_pooled_tensor)
        col_pooled_tensor = self.l1(col_pooled_tensor)
        col_pooled_tensor = F.gelu(col_pooled_tensor)
        col_pooled_tensor = self.l2(col_pooled_tensor)
        row_expanded_tensor = F.sigmoid(row_pooled_tensor.expand(x.shape))
        col_expanded_tensor = F.sigmoid(col_pooled_tensor.expand(x.shape))

        row_expanded_tensor = row_expanded_tensor * x
        col_expanded_tensor = col_expanded_tensor * x
        x = row_expanded_tensor + col_expanded_tensor

        x = rearrange(x, 'b c w h -> b h w c').contiguous()
        x = x + res
        res = x
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        out = x + res
        return out


class FAAdapter(nn.Module):

    def __init__(self, dim, h, w, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, pe=None):
        super().__init__()
        self.skip_connect = skip_connect

        self.norm1 = nn.LayerNorm(768)

        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 768)

        self.MyFF = MyParser()

    def forward(self, x, pe):
        res = x
        
        x=self.norm1(x)
        
        x = rearrange(x, 'b h w c -> b c w h').contiguous()
        
        
        x = self.MyFF(x)

        x = rearrange(x, 'b c w h -> b h w c').contiguous()

        x = x + res

        res = x

        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        out = x + res
        return out


class MyParser(nn.Module):
    def __init__(self, dim=768, h=64, w=64):
        super().__init__()

    def forward(self, x, spatial_size=None):

        B, C, H, W = x.shape
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        x = x.to(torch.float32)

        c = torch.full((1, 1), 5, dtype=torch.float)

        c = nn.Parameter(c)

        mask = tensor = create_circle_tensor(64, c).expand(x.shape).to(x.device)

        fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)))

        fft = fft * mask

        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))

        inv = torch.fft.ifft2(fft_hires, dim=(-2, -1)).real
        inv = torch.abs(inv)

        return inv

def create_circle_tensor(size, radius):
    # Create a grid of coordinates
    x = torch.arange(size, dtype=torch.float32)
    y = torch.arange(size, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y)

    # Calculate distances from the center
    distance = torch.sqrt((xx - size // 2) ** 2 + (yy - size // 2) ** 2)

    # Create a tensor of ones
    tensor = torch.ones(size, size)

    # Set values inside the circle to 0
    tensor[distance < radius] = 0

    return tensor


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
