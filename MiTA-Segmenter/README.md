# MiTA-Segmenter

## Usage
Our implementation requires FlashAttention v2.6.3. 
We recommend directly downloading the wheel `flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl`  at [this URL](https://flashattn.dev/?utm_source=chatgpt.com).

Then, run:
```
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
```

## Training & Evaluation
Train a Segmenter model with our MiTA Attention using:
```
python -m segm.train --log-dir mita_seg_tiny_mask --dataset ade20k --backbone vit_tiny_patch16_384 --decoder mask_transformer
```
To evaluate the model, use:
```
# single-scale evaluation:
python -m segm.eval.miou mita_seg_tiny_mask/checkpoint.pth ade20k --singlescale
# multi-scale evaluation:
python -m segm.eval.miou mita_seg_tiny_mask/checkpoint.pth ade20k --multiscale
```

## Acknowledgements
The code is largely adapted from the official [Segmenter](https://github.com/rstrudel/segmenter) repository.
