import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

from einops import rearrange, repeat
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


class Agent_Attention_Bias(nn.Module): # we remove the agent bias and dwc trick of agent attention, and refer to this as Agent Attention minus
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = None  # dummy
        # notice: agent attention is compatible with flash attention / SDPA even with the agent bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # averge pooling
        self.pool_size = 7
        self.num_agent = self.pool_size ** 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.pool_size, self.pool_size))
        
        # agent bias
        # there are 4 drawbacks when using agent bias
        # 1. it can not be implemented via FlashAttention
        # 2. it heavily relies static spatial prior
        # 3. agent bias must be trained from scratch even when using pretrained ViTs based on full attention
        # 4. intuitively, this design struggles to generalize to other modalities
        # it is supposed that agent bias works because 1) it diversifies the agent values, i.e., reducing the information loss caused by compression; 2) it speeds up the convergence via a limited attention span
        # additionaly, comparing full attention / mita attention without agent bias with agent attention with agent bias is unfair, as full attention can also be equipped with tricks like agent bias, e.g., masked attention (see Table 4a in Mask2Former for ablation).
        window = 14
        self.window = window
        self.an_bias = nn.Parameter(torch.zeros(num_heads, self.num_agent, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, self.num_agent, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, self.num_agent, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, self.num_agent, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, self.num_agent))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, self.num_agent))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, self.num_agent, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, self.num_agent))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        
    # def attention(self, q, k, v):
    #     q = q * self.scale
    #     attn = q @ k.transpose(-2, -1)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     x = attn @ v
    #     return x, attn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('using agent attention with agent bias!')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # [3, B ,N, C]
        q, k, v = qkv.unbind(0)  # [B, N, C]
        
        H = W = int(N ** 0.5)
        router = self.pool(q[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1) 
        assert N == 1 + self.window ** 2 and 1 + H * W 

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        router = router.reshape(B, self.num_agent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, self.num_heads, self.num_agent, -1).repeat(B, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.num_agent, -1).repeat(B, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        position_bias = torch.cat([self.ac_bias.repeat(B, 1, 1, 1), position_bias], dim=-1)
        agent_attn = F.softmax((router * self.scale) @ k.transpose(-2, -1) + position_bias, dim=-1)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.num_agent, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.num_agent).repeat(B, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        agent_bias = torch.cat([self.ca_bias.repeat(B, 1, 1, 1), agent_bias], dim=-2)
        q_attn = F.softmax((q * self.scale) @ router.transpose(-2, -1) + agent_bias, dim=-1)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(B, N, C)
        # v_ = v[:, :, 1:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x[:, 1:, :] = x[:, 1:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n - 1, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x