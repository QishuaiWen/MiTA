# MiTA-ViT-5: ViT-5 models with MiTA Attention 

## Usage
Our implementation requires FlashAttention v2.6.3. 
We recommend directly downloading the wheel `flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl`  at [this URL](https://flashattn.dev/?utm_source=chatgpt.com).

Then, run:
```
# Install PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
pip install timm==0.4.12 numpy wandb einops

# Install FlashAttention
pip install ./flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl

# Install NVIDIA Apex (required for fused optimizers, e.g., fusedlamb)
git clone https://github.com/NVIDIA/apex
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
```

## Training & Evaluation
Train a ViT-5-Small model with our MiTA Attention (MiTA-ViT-5-Small) using:
```
torchrun --nproc_per_node 8 main.py --model vit5_small --input-size 224 --data-path imagenet --output_dir output_dir --batch 256 --accum_iter 1 --lr 4e-3 --weight-decay 0.05 --epochs 800 --opt fusedlamb --unscale-lr --mixup .8 --cutmix 1.0 --color-jitter 0.3 --drop-path 0.05 --reprob 0.0 --smoothing 0.0 --ThreeAugment --repeated-aug --bce-loss --warmup-epochs 5 --eval-crop-ratio 1.0 --dist-eval --disable_wandb
```
To evaluate the model, add the following two arguments:
```
--eval --resume output_dir/best_checkpoint.pth
```

## Acknowledgements
The code is largely adapted from the official [ViT-5](https://github.com/wangf3014/ViT-5) repository.




