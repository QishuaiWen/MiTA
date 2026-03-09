# Adapted from MoonshotAI/MoBA/moba/moba_efficient.py
# https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_efficient.py
# Modifications by Qishuai Wen, 2026-1-5

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange, repeat
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

from .mixed_attention import MixedAttention


class MiTA_Attention(nn.Module):
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
            pool_size=5,
            kv_topk=5,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = None  # dummy

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # averge pooling
        self.pool_size = pool_size
        self.num_expert = self.pool_size ** 2
        self.kv_topk = kv_topk ** 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.pool_size, self.pool_size))
        print(f'num_expert: {self.num_expert}, kv_topk: {self.kv_topk}')
        
    def mita(self, query, key, value, router, router_topk=1, kv_topk=64):
        """
        query, key, value: [B, H, N, d]
        router: [B, H, M, d]
        """
        B, H, N, d = query.shape
        M = router.shape[2]
        scale = d ** -0.5

        # construct the dynamic experts
        router_attn_key = router @ key.transpose(-2, -1)  # [B, H, M, N]
        _, router_topk_key_idx = router_attn_key.topk(kv_topk, dim=-1, largest=True, sorted=False)  # [B, H, M, kv_topk]

        # assign queries to experts
        gate = router @ query.transpose(-2, -1)  # [B, H, M, N]
        _, gate_topk_idx = torch.topk(gate, k=router_topk, dim=-2, largest=True, sorted=False)  # [B, H, router_topk, N], i.e, [B, H, 1, N]

        # compress values via routers (i.e., agents)
        agent_value = F.scaled_dot_product_attention(router, key, value)

        # align with MoBA ...
        gate_topk_idx = gate_topk_idx + M * torch.arange(0, B, device=query.device).view(B, 1, 1, 1)  # index the expert in 0~M-1 -> index the expert in 0~B*M-1 
        gate_mask = torch.zeros((B, H, B*M, N), dtype=torch.bool, device=query.device)   #  [B, H, B*M, N] 
        gate_mask = gate_mask.scatter_(dim=-2, index=gate_topk_idx, value=True)  # [B, H, B*M, N]    
        gate_mask = gate_mask.permute(2, 1, 0, 3).reshape(B*M, H, -1)  # [B, H, B*M, N] -> [B*M, H, B, N] -> [B*M, H, B*N]    
        moba_seqlen_q = gate_mask.sum(dim=-1).flatten()  # [B*M*H]

        # combining all q index that needs moba attn
        moba_q_indices = gate_mask.reshape(B*M, -1).nonzero(as_tuple=True)[-1]  # [B*M, H, B*N] -> [B*M, H*B*N]  , index the query in 0~H*B*N-1
        moba_q = query.permute(1, 0, 2, 3).reshape(-1, d).index_select(0, moba_q_indices)  # [B, H, N, d] -> [H, B, N, d] -> [H*B*N, d] -> [?, d]
        moba_q = moba_q.unsqueeze(1)  # [?, 1, d], unsqueeze a dummy head dimension
        moba_q_sh_indices = moba_q_indices % (B*N) * H + moba_q_indices // (B*N)

        # cut off zero experts
        q_zero_mask = moba_seqlen_q == 0
        valid_expert_mask = ~q_zero_mask
        zero_expert_count = q_zero_mask.sum()    
        if zero_expert_count > 0:
            moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
        moba_cu_seqlen_q = torch.cat((torch.tensor([0], device=query.device, dtype=moba_seqlen_q.dtype), moba_seqlen_q.cumsum(dim=0)), dim=0).to(torch.int32)

        # index top-{kv_topk} key-value paris in 0~N -> ... 0~B*N*H-1
        router_topk_key_idx = (N * H) * torch.arange(0, B, device=query.device).view(B, 1, 1, 1) + torch.arange(0, H, device=query.device).view(1, H, 1, 1) + router_topk_key_idx * H
        router_topk_key_idx = router_topk_key_idx.permute(0, 2, 1, 3).reshape(-1, kv_topk)  # [B, H, M, kv_topk] -> [B, M, H, kv_topk] -> [B*M*H, kv_topk] 
        if zero_expert_count > 0:
            assert valid_expert_mask.sum() == router_topk_key_idx.shape[0] - zero_expert_count
            router_topk_key_idx = router_topk_key_idx[valid_expert_mask]
        key = key.permute(0, 2, 1, 3).reshape(-1, d)   # [B, H, N, d] -> [B, N, H, d] -> [B*N*H, d] 
        value = value.permute(0, 2, 1, 3).reshape(-1, d)
        router_topk_key = torch.take_along_dim(key.unsqueeze(0), router_topk_key_idx.unsqueeze(-1), dim=1)
        router_topk_value = torch.take_along_dim(value.unsqueeze(0), router_topk_key_idx.unsqueeze(-1), dim=1)
        moba_kv = torch.cat((router_topk_key.flatten(start_dim=0, end_dim=1).unsqueeze(1), router_topk_value.flatten(start_dim=0, end_dim=1).unsqueeze(1)), dim=1).unsqueeze(2)  
        moba_cu_seqlen_kv = (torch.arange(0, B*M*H + 1 - zero_expert_count, dtype=torch.int32, device=query.device) * kv_topk)

        # shape check
        assert moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape, f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

        query = query.permute(0, 2, 1, 3).reshape(-1, H, d)    # [B, H, N, d] -> [B, N, H, d] -> [B*N, H, d]
        router = router.permute(0, 2, 1, 3).reshape(-1, H, d)    # [B, H, M, d] -> [B, M, H, d] -> [B*M, H, d]
        agent_value = agent_value.permute(0, 2, 1, 3).reshape(-1, H, d)    # [B, H, M, d] -> [B, M, H, d] -> [B*M, H, d]

        agent_attn_cu_seqlen_q = N * torch.arange(0, B+1, dtype=torch.int32, device=moba_q.device)
        agent_attn_cu_seqlen_kv = M * torch.arange(0, B+1, dtype=torch.int32, device=moba_q.device)
        
        output = MixedAttention.apply(
            query,  # param q
            router * scale,  # param k, smooth the attention weights on router/agent_key
            agent_value,  # param v
            agent_attn_cu_seqlen_q,  # param self_attn_cu_seqlens_q
            agent_attn_cu_seqlen_kv,  # param self_attn_cu_seqlens_kv
            moba_q,  # param moba_q
            moba_kv,  # param moba_kv
            moba_cu_seqlen_q,  # param moba_cu_seqlen_q
            moba_cu_seqlen_kv,  # param moba_cu_seqlen_kv
            moba_seqlen_q.max().item(),  # max_seqlen
            M,  # param max_seqlen_agent
            router_topk * kv_topk,  # param moba_chunk_size
            moba_q_sh_indices,  # param moba_q_sh_indices
        )
        output = rearrange(output, "(B N) H d -> B H N d", B=B)
        return output


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print('using mita attention pro')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # [3, B ,N, C]
        q, k, v = qkv.unbind(0)  # [B, N, C]
        
        H = W = int(N ** 0.5)
        router = self.pool(q[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)  # notice: include the cls token, ditch the last token

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        router = router.reshape(B, self.num_expert, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        x = self.mita(q, k, v, router, router_topk=2, kv_topk=int(self.kv_topk // 2))
        x = x.transpose(1, 2).reshape(B, N, C)
   
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
