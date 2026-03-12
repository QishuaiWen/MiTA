# MiTA-DeiT: DeiT Models with MiTA Attention 

## Usage 
Our implementation requires FlashAttention v2.6.3. 
We recommend directly downloading the wheel that matches your environment at [this URL](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3).

Take `flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl` as an example.
If this wheel matches your environment, run:
```
pip install -r requirements.txt
```
Otherwise, you should modify the last entry in `requirements.txt`.

## Training & Evaluation
By default, standard attention in a DeiT will be replaced by our MiTA Attention in `main.py` #L285.
You can manually switch to other efficient attention supported by the `mita` package, or add new ones.

Train a DeiT-Tiny model with your desired attention mechanism using:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path imagenet --output_dir output_dir
```
To evaluate the model, add the following two arguments:
```
--eval --resume output_dir/best_checkpoint.pth
```

More commands can be found in the official [DeiT](https://github.com/facebookresearch/deit/tree/main) repository.
Our only modification is the replacement of the attention mechanism.

## Acknowledgements

The code is largely adapted from the great [DeiT](https://github.com/facebookresearch/deit/tree/main) project of facebookresearch.
 








