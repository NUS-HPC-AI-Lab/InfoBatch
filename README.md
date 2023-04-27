# InfoBatch Code
Paper Link: https://arxiv.org/pdf/2303.04947.pdf

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/motivation.png) 

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/pipeline.png)

##CIFAR Experiments
To run CIFAR-100 example with Baseline, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```

To run CIFAR-100 example with InfoBatch, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --ratio 0.5 --delta 0.875
```

##Other experiments
Code of other experiments will be released in the following month (May 2023). 
Hyperparamets not included in main text will also be released in appendix.


## Citation
```angular2html
@article{qin2023infobatch,
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhou, Daquan and You, Yang},
  journal={arXiv preprint arXiv:2303.04947},
  year={2023}
}
```