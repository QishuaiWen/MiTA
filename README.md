# MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k Activations

This repository is the official PyTorch implementation of the paper:
[MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k Activations](https://arxiv.org/abs/2602.01219v3).

MiTA Attention is a novel efficient attention mechanism that adopts a **compress-and-route strategy**, consisting of a compressed shared fast-weight expert and several routed **deformable** fast-weight experts.

<p align="center">
    <img src="figures/illustration_MiTA.png" width="650"\>
<br> <em>Overview of MiTA Attention </em>
<p align="center">

# Usage
We provide a pure implementation of MiTA Attention in package [mita](https://github.com/QishuaiWen/MiTA/tree/main/mita), which can be a plug-in module in other tasks.

For exmaple:
```
# make sure that flash-attn==2.6.3 is installed before using MiTA Attention
from mita import MiTA_Attention
attention_mita = MiTA_Attention(dim=384, num_heads=6)
x = torch.randn(1, 256, 384)
x = block(x)
```
Additionally, there are many variants of MiTA Attention as well as other classical/newest efficient attention in the mita package. 
Interested users can check them out and use them. Stay tuned for more integrated implementations in the future.

We also release the code of:
+ MiTA-DeiT [[README](https://github.com/QishuaiWen/MiTA/blob/main/MiTA-DeiT/README.md)]: DeiT models with MiTA Attention for image classification;
+ MiTA-ViT-5 [[README](https://github.com/QishuaiWen/MiTA/blob/main/MiTA-ViT-5/README.md)]: ViT-5 models with MiTA Attention for image classification;
+ MiTA-Segmenter [[README](https://github.com/QishuaiWen/MiTA/blob/main/MiTA-Segmenter/README.md)]: Segmenter models with MiTA Attention for semantic segmentation.

## 🔄 Updates
[2026/3/5] The lastest version ([v3](https://arxiv.org/abs/2602.01219v3)) of our paper includes improved experiments.

[2026/2/3] Released a pure implementation of MiTA attention at [here](https://github.com/QishuaiWen/MiTA/tree/main/MiTA%20Attention).

[2026/2/1] Our paper is now publicly available on arXiv ([v1](https://arxiv.org/abs/2602.01219v1)).

## Citation
If you find this repo helpful, please consider citing us.
```
@article{wen2026mita,
  title={MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-$k$ Activations},
  author={Wen, Qishuai and Huang, Zhiyuan and Meng, Xianghan and He, Wei and Li, Chun-Guang},
  journal={arXiv preprint arXiv:2602.01219},
  year={2026}
}
```
