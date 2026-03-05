# paper https://arxiv.org/abs/2506.21137v2

import math
import torch
import torch.nn as nn


class NaLaLinearAttention(nn.Module):
    """
    For fair comparison, this implementation of NaLaLinearAttention removes: 1. the gate matrix; 2. the dwc module.
    """
    def __init__(self, dim, 
                 # window_size, 
                 num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        # self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.qkvg = nn.Linear(dim, dim * 4)
        self.qkv = nn.Linear(dim, dim * 3)

        self.act = nn.SiLU()
        # self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        # self.cosine_inhibit = False
        self.cosine_inhibit = True
        self.alpha = 20


    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print('using NaLaLinearAttention')
        B, N, C = x.shape
        # H = self.window_size[0]
        # W = self.window_size[1]

        head_dim = self.head_dim
        num_heads = self.num_heads

        # qkvg = self.qkvg(x).reshape(B, H * W, 4, C).permute(2,0,1,3)
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2,0,1,3)
        
        # qkvg = qkvg.float()
        qkv = qkv.float()
        # q, k, v, g = qkvg[0], qkvg[1], qkvg[2], qkvg[3]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # lepe = self.lepe(v.reshape(B, H, W, C).permute(0,3,1,2)).permute(0, 2, 3, 1).reshape(B, N, -1) # (b c h w)
        
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()

        if self.cosine_inhibit:
            q_norm = q.norm(dim=-1, p=2, keepdim=True) #B, nhead, N, 1
            k_norm = k.norm(dim=-1, p=2, keepdim=True) #B, nhead, N, 1
            q_t = q / q_norm
            dq = torch.tanh(q / q_norm * self.alpha) * math.pi / 4
            dk = torch.tanh(k / k_norm * self.alpha) * math.pi / 4

            k = torch.abs(k) ** 3
            k = torch.cat([k * torch.cos(dk),k * torch.sin(dk)],dim=-1)
            
            power = 3 / 2 * (0.5 + torch.tanh(q_norm))
            q_t = (q_t**2) ** power
            q = torch.cat([q_t * torch.cos(dq), q_t * torch.sin(dq)],dim=-1)
            

        else:
            k = k ** 3
            q1 = torch.relu(q)
            k1 = torch.relu(k)

            q2 = torch.relu(-q)
            k2 = torch.relu(-k)

            q_norm = q.norm(dim=-1, p=2, keepdim=True) #B, nhead, N, 1

            q1 = q1 / q_norm
            q2 = q2 / q_norm

            power = 3 * (0.5 + torch.tanh(q_norm))

            q1 = q1**power
            q2 = q2**power

            q = torch.cat([q1,q2],dim=-1)
            k = torch.cat([k1,k2],dim=-1)
            
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k, v = q.float(), k.float(), v.float()
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
            x = q @ kv * z

        x = x.transpose(1, 2).reshape(B, N, -1)
        # x = x + lepe

        # x = self.ln(x) * self.act(g)
        x = self.ln(x)
            
        x = self.out_proj(x)
        return x

#         x = x.reshape(B, H, W, C).permute(0,3,1,2)
#         return x.contiguous().view(B, N, C)