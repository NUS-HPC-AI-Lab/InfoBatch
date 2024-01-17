# InfoBatch ICLR-2024 Oral Presentation
Paper Link: https://arxiv.org/pdf/2303.04947.pdf

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/motivation.png) 

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/pipeline.png)

## 2024 Jan 16 Update
Our work got accepted to ICLR 2024(Oral)! A new version with only 3 lines of change will be updated soon. Experiments included in the paper (and beyond) will be gradually updated with detail.

## 2023 Aug 1 Update
We are now able to losslessly save 40.9% on CIFAR100 and ImageNet. Updating paper content and preparing for public code.

## CIFAR Experiments
To run CIFAR-100 example with Baseline, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```

To run CIFAR-100 example with InfoBatch, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --ratio 0.5 --delta 0.875
```

## Other experiments
The code of other experiments will be released in the following months. 
Hyperparameters not included in main text will also be released in the appendix.


## Citation
```angular2html
@article{qin2023infobatch,
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhaopan Xu and Zhou, Daquan and Lei Shang and Baigui Sun and Xuansong Xie and You, Yang},
  journal={arXiv preprint arXiv:2303.04947},
  year={2023}
}
```
