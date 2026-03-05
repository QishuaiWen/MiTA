import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

from einops import rearrange, repeat
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List


class Agent_Attention(nn.Module): # we remove the agent bias and dwc trick of agent attention, and refer to this as Agent Attention minus
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
        
        # agent bias (ditch agent bias to enable FlashAttention
        # agent_num = num_agent
        # window = 14
        # self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        # self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        # self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        # self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        # self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        # self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        # self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        # self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        # trunc_normal_(self.an_bias, std=.02)
        # trunc_normal_(self.na_bias, std=.02)
        # trunc_normal_(self.ah_bias, std=.02)
        # trunc_normal_(self.aw_bias, std=.02)
        # trunc_normal_(self.ha_bias, std=.02)
        # trunc_normal_(self.wa_bias, std=.02)
        # trunc_normal_(self.ac_bias, std=.02)
        # trunc_normal_(self.ca_bias, std=.02)
        
    # def attention(self, q, k, v):
    #     q = q * self.scale
    #     attn = q @ k.transpose(-2, -1)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     x = attn @ v
    #     return x, attn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('using agent attention!')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # [3, B ,N, C]
        q, k, v = qkv.unbind(0)  # [B, N, C]
        
        H = W = int(N ** 0.5)
        router = self.pool(q[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)  # notice: include the cls token, ditch the last token

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        router = router.reshape(B, self.num_agent, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # raise Exception(router.shape)
          
        # router_value, _ = self.attention(router, k, v)
        # x, _ = self.attention(q, router, router_value)
        router_value = F.scaled_dot_product_attention(router, k, v, dropout_p=self.attn_drop.p if self.training else 0.,)
        x = F.scaled_dot_product_attention(q, router, router_value, dropout_p=self.attn_drop.p if self.training else 0.,)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x