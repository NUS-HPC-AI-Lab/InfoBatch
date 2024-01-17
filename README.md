# InfoBatch 
ICLR 2024 Oral Presentation

Paper Link: https://arxiv.org/pdf/2303.04947.pdf

InfoBatch is a work aiming at lossless deep learning training acceleration. We provide our code for research studies and real-world applications.

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/pipeline.png)

## News

[2024/1/17] 🔥 New version with only 3 lines of change comes!  Note that one should use a per-sample loss (to serve as the score and total loss calculation)

[2024/1/16] 🔥 Our work got accepted to ICLR 2024 (oral)! A new version with only 3 lines of change will be updated soon. Experiments included in the paper (and beyond) will be gradually updated with detail.

[2023/8/1] 🔥 InfoBatch can now losslessly save 40.9% on CIFAR100 and ImageNet. We are updating paper content and preparing for public code.

## TODO List
- [x] Paper
- [ ] PyPI Registration
- [ ] Update experiment code


## Contents
- [Get Started](#get-started)
- [Citation](#citation)


## Get Started
To run CIFAR-100 example with Baseline, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```
To run CIFAR-100 example with mixed precision training, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --fp16 --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```
To run CIFAR-100 example with mixed precision training, distributed data parallel (e.g 1 machine with 2 gpus.), execute in command line:

```angular2html
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --use_env --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=23456 --node_rank=0 cifar_example.py --use_ddp --fp16 --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```

To run CIFAR-100 example with InfoBatch, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --ratio 0.5 --delta 0.875
```

To run CIFAR-100 example with InfoBatch, with mixed precision training, distributed data parallel (e.g 1 machine with 2 gpus.), execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --use_env --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=23456 --node_rank=0 cifar_example.py --use_ddp --use_info_batch --fp16 --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 --ratio 0.5
```

You may observe performance drop when using the Distributed Data Parallel (DDP) training approach compared to the Data Parallel (DP) approach on multiple GPUs, especially in versions prior to Pytorch 1.11. However, this is not specific to our algorithm itself.


## Citation
```angular2html
@article{qin2023infobatch,
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhaopan Xu and Zhou, Daquan and Lei Shang and Baigui Sun and Xuansong Xie and You, Yang},
  journal={arXiv preprint arXiv:2303.04947},
  year={2023}
}
```