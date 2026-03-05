import torch
import torch.nn as nn
from einops import rearrange

from timm.models.layers import trunc_normal_

from segm.model.dec_blocks import Transformer
from segm.model.utils import init_weights

class MaskTransformer(nn.Module):
    def __init__(
            self,
            n_cls,
            patch_size,
            n_layers,
            n_heads,
            d_model,
            dropout,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        
        self.net = Transformer(d_model, n_layers, n_heads, 100, dropout)
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size=None):
        H, W = im_size
        GS = H // self.patch_size

        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        
        x = torch.cat((x, cls_emb), 1)
        x = self.net(x)
        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, :-self.n_cls], x[:, -self.n_cls:]
            
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks