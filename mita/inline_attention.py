# https://github.com/LeapLabTHU/InLine/blob/master/models/inline_deit.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class InLineAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.softmax = nn.Softmax(dim=-1)
        # self.window = window

        self.residual = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),  # head-wise channel-mixing
            nn.GELU(),
            nn.Conv1d(dim, dim * 9, kernel_size=1, groups=num_heads)
        )

    def forward(self, x):
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        res_weight = self.residual(x.mean(dim=1).unsqueeze(dim=-1)).reshape(b * c, 1, 3, 3)

        # The self.scale / n = head_dim ** -0.5 / n is a scale factor used in InLine attention.
        # This factor can be equivalently achieved by scaling \phi(Q) = \phi(Q) * self.scale / n
        # Therefore, we omit it in eq. 5 of the paper for simplicity.
        kv = (k.transpose(-2, -1) * (self.scale / n) ** 0.5) @ (v * (self.scale / n) ** 0.5)
        x = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).reshape(1, b * c, h, w)
        residual = F.conv2d(v_, res_weight, None, padding=(1, 1), groups=b * c)
        x[:, 1:, :] = x[:, 1:, :] + residual.reshape(b, c, n - 1).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x