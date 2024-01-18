<h2 align="center">InfoBatch</h2>
<p align="center"><b>ICLR 2024 Oral</b> | <a href="https://arxiv.org/pdf/2303.04947.pdf">[Paper]</a> | <a href="https://github.com/henryqin1997/InfoBatch">[Code]</a> </p>

InfoBatch is a tool for lossless deep learning training acceleration. It achieves lossless training speed-up by unbiased dynamic data pruning.

![image](https://github.com/NUS-HPC-AI-LAB/InfoBatch/blob/master/figs/pipeline.png)

## News

[2024/1/17] 🔥 New version with only 3 lines of change comes!  Note that one should use a per-sample loss (to update the score and calculate batch loss)

[2024/1/16] 🔥 Our work got accepted to ICLR 2024 (oral)! A new version with only 3 lines of change will be updated soon. Experiments included in the paper (and beyond) will be gradually updated with detail.

[2023/8/1] 🔥 InfoBatch can now losslessly save 40.9% on CIFAR100 and ImageNet. We are updating paper content and preparing for public code.

## TODO List

- [x] Plug-and-Play Implementation of InfoBatch
- [ ] PyPI Registration
- [x] Experiment: Classification on Cifar
- [ ] Experiment: Classification on ImageNet
- [ ] Experiment: Segmentation
- [ ] Experiment: Diffusion
- [ ] Experiment: Instruction Finetunning

## Contents
- [Install](#install)
- [Get Started](#get-started)
- [Experiments](#experiments)
- [Citation](#citation)

## Install

Install InfoBatch via

```bash
pip install git+https://github.com/henryqin1997/InfoBatch
```

Or you can clone this repo and install it locally.

```bash
git clone https://github.com/henryqin1997/InfoBatch
cd InfoBatch
pip install -e .
```

## Get Started
To adapt your code with InfoBatch, just download and import InfoBatch, and change the following three lines:

![image](https://github.com/NUS-HPC-AI-Lab/InfoBatch/blob/master/figs/three_line_of_code.png)

Note that one should use a **per-sample loss** to update the score and calculate batch loss; if the **learning rate scheduler**
is **epoch-based**, **adjust its steps accordingly** at beginning of each epoch.

For research studies and more flexible codes, you can refer to the code in `research`.

## Experiments

### Cifar

To run the CIFAR-100 example with baseline, run with delta=0:
```bash
python3 examples/cifar_example.py \
  --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```

To run the CIFAR-100 example with InfoBatch, run the following:
```bash
python3 examples/cifar_example.py \
  --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 --ratio 0.5
```

Our example also supports mixed precision training and distributed data parallelism with the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --use_env --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=23456 --node_rank=0 cifar_example.py \
  --use_ddp --use_info_batch --fp16 \
  --model r50 --optimizer lars --max-lr 5.2 --delta 0.875 --ratio 0.5
```

You may observe a performance drop when using the Distributed Data-Parallel (DDP) training approach compared to the Data Parallel (DP) approach on multiple GPUs, especially in versions prior to Pytorch 1.11. However, this is not specific to our algorithm itself.

## Citation
```bibtex
@inproceedings{
  qin2024infobatch,
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhaopan Xu and Zhou, Daquan and Lei Shang and Baigui Sun and Xuansong Xie and You, Yang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=C61sk5LsK6}
}
```
