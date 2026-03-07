# https://github.com/ZacharyMeng/PolaFormer/blob/main/models/pola_swin.py
# Table 3: PolaLinearAttention - DWC - Gate: 68.1 acc on ImageNet-1K

# class PolaLinearAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  kernel_size=5, alpha=4):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.head_dim = head_dim

#         # self.qkvg = nn.Linear(dim, dim * 4, bias=qkv_bias)
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         # self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
#         #                      groups=head_dim, padding=kernel_size // 2)
        
#         self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
#         self.alpha = alpha

#         self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
#         self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))
#         print('Linear Attention window{} f{} kernel{}'.
#               format(window_size, alpha, kernel_size))


#     def forward(self, x, H, W, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B, N, C = x.shape
#         # qkv = self.qkvg(x).reshape(B, N, 4, C).permute(2, 0, 1, 3)
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         # q, k, v, g = qkv.unbind(0)
#         q, k, v = qkv.unbind(0)
#         k = k + self.positional_encoding
#         kernel_function = nn.ReLU()
        
#         scale = nn.Softplus()(self.scale)
#         power = 1 + self.alpha * nn.functional.sigmoid(self.power)

#         q = q / scale
#         k = k / scale
#         q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
#         k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
#         v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

#         q_pos = kernel_function(q) ** power 
#         q_neg = kernel_function(-q) ** power 
#         k_pos = kernel_function(k) ** power 
#         k_neg = kernel_function(-k) ** power 

#         q_sim = torch.cat([q_pos, q_neg],dim=-1)
#         q_opp = torch.cat([q_neg, q_pos],dim=-1)
#         k = torch.cat([k_pos, k_neg],dim=-1)     

#         v1,v2 = torch.chunk(v,2,dim=-1)
        
#         z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v1 * (N ** -0.5))
#         x_sim = q_sim @ kv * z
#         z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v2 * (N ** -0.5))
#         x_opp = q_opp @ kv * z
        
#         x = torch.cat([x_sim, x_opp],dim=-1)
#         x = x.transpose(1,2).reshape(B, N, C)
        
#         H = W = int(N ** 0.5)
# #         v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
# #         v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

# #         x = x + v
#         # x = x * g

#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x
#     def eval(self):
#         super().eval()
#         print('eval')

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
