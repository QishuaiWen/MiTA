# source: https://drive.google.com/file/d/1tf4pD6VQlq2uTzuPHjzE1b8I5XU8mwsZ/view

import torch
import torch.nn as nn

from einops import rearrange, repeat


class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 # version=1, 
                 focusing_factor=3, n=197):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, n, dim)))
        # self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=5, groups=head_dim, padding=2)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.version = version
        self.focusing_factor = focusing_factor

    def forward(self, x):
        # print('using focused linear attention!')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        focusing_factor = self.focusing_factor
        k = k + self.positional_encoding
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])

        # Q: [b, i, c]
        # K: [b, j, c]
        # V: [b, j, d]
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k, v = q.float(), k.float(), v.float()
            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                o = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                o = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        # num = int(v.shape[1] ** 0.5)
        # num2 = num ** 2
        # feature_map = rearrange(v[:, v.shape[1] - num2:, :], "b (w h) c -> b c w h", w=num, h=num)
        # feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        # o[:, v.shape[1] - num2:, :] = o[:, v.shape[1] - num2:, :] + feature_map

        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        o = self.proj(o)
        o = self.proj_drop(o)
        return o