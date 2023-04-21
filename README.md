# DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification

Train

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes 1 \
    --node_rank=0 --master_port=1901 --use_env train_memory.py \
    --dataset 'WHUHi_LongKou_15_100' \
    --backbone 'vgg16' --crop-size 256 \
    --epochs 30 --lr 1e-3 --groups 128 --eval_interval 1 \
    --batch_size 4 --test_batch_size 1 --workers 2 \
    --ra_head_num 4 --ga_head_num 4 --mode 'soft'
```

Testing

```
CUDA_VISIBLE_DEVICES=0 python test_gpu.py \
    --dataset 'WHUHi_LongKou_15_100' \
    --backbone 'vgg16' --ra_head_num 4 --ga_head_num 4 \
    --scales 1  --crop-size 256 --groups 128 \
    --model_path './run/WHUHi_LongKou_15_100/vgg16_128/experiment_0/model_last.pth.tar' \
    --save_folder './run/WHUHi_LongKou_15_100/vgg16_128/experiment_0/'
```


Citation

```
@ARTICLE{2023arXiv230409915W,
       author = {{Wang}, Di and {Zhang}, Jing and {Du}, Bo and {Zhang}, Liangpei and {Tao}, Dacheng},
        title = "{DCN-T: Dual Context Network with Transformer for Hyperspectral Image Classification}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2023,
        month = apr,
          eid = {arXiv:2304.09915},
        pages = {arXiv:2304.09915},
          doi = {10.48550/arXiv.2304.09915},
archivePrefix = {arXiv},
       eprint = {2304.09915},
 primaryClass = {cs.CV},
}
```

To be continued.
