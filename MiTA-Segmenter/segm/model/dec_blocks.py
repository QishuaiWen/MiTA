"""
Adapted from https://github.com/Ma-Lab-Berkeley/CRATE
"""

import torch
from torch import nn

from einops import rearrange

from segm.model.utils import ortho

# from .mita import MiTA_Attention as Attention


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# class Attention(nn.Module):
#     def __init__(self, dim, heads, dim_head, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads

#         self.heads = heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         self.proj = nn.Parameter((dim ** -0.5) * torch.randn(inner_dim, dim))
#         self.step_size = nn.Parameter(torch.randn(1))

#     def forward(self, x, query=None):
#         proj = self.proj
#         proj = ortho(proj, self.heads, self.dim_head, operation=None)  # operation: None, ortho_trans, head_ortho, head_trans
        
#         w = rearrange(x @ proj.t(), 'b n (h d) -> b h n d', h=self.heads)
        
#         if query is not None:
#             query = rearrange(query @ proj.t(), 'b n (h d) -> b h n d', h=self.heads)
#             dots = torch.matmul(query, w.transpose(-1, -2)) * self.scale
#         else:
#             dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, w)

#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = out @ proj
#         return self.step_size * out

        
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, inner_dim, bias=False)
        
        self.step_x = nn.Parameter(torch.randn(heads, 1, 1))
        self.step_rep = nn.Parameter(torch.randn(heads, 1, 1))
        
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(10, 10))
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        
        # dwc
        # self.dwc = nn.Conv2d(in_channels=300, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
    
    def attention(self, query, key, value):        
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = attn @ value
        return out, attn

    def forward(self, x):
        b, n, c = x.shape
        h = width = int((n - 150) ** 0.5)
        # print(h, width)
        
        w = self.proj(x)

        # print(w.shape)
        rep = self.pool(w[:, :-150, :].reshape(b, h, width, 300).permute(0, 3, 1, 2)).reshape(b, 300, -1).permute(0, 2, 1)
        # print(rep.shape)

        w = w.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        rep = rep.reshape(b, 100, self.heads, self.dim_head).permute(0, 2, 1, 3)
        # rep = rep.reshape(b, 25, self.heads, self.dim_head).permute(0, 2, 1, 3)
        # rep = rep.reshape(b, 49, self.heads, self.dim_head).permute(0, 2, 1, 3)

        rep_delta, attn = self.attention(rep, w, w)
        rep = rep + self.step_rep * rep_delta
        
        x_delta, _ = self.attention(rep, rep, rep)  
        x_delta = attn.transpose(-1, -2) @ x_delta
        x_delta = self.step_x * x_delta
        
        x_delta = rearrange(x_delta, 'b h n k -> b n (h k)')
        
#         H = W = int((n-150) ** 0.5)
#         assert H * W == n - 150
#         # print(w.shape, H, W, n, c)
#         v_ = w[:, :, :-150, :].transpose(1, 2).reshape(b, H, W, 300).permute(0, 3, 1, 2).contiguous()
#         x_ = self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n - 150, 300)
        
#         x_delta = x_delta + F.pad(x_, (0, 0, 0, 150))
        
        return self.to_out(x_delta)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            # Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            # Attention(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout)
            
    def forward(self, x, query=None):
        if query is not None:
            for attn in self.layers:
                query = attn(x, query=query) + query
            return query
        else:
            for attn in self.layers:
                x = attn(x) + x
            return x