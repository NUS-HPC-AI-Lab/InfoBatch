# InfoBatch Code Coming Soon
Paper Link: https://arxiv.org/pdf/2303.04947.pdf

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/motivation.png) 

![image](https://github.com/henryqin1997/InfoBatch/blob/master/figs/pipeline.png)

To run CIFAR-100 example with Baseline, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```

To run CIFAR-100 example with InfoBatch, execute in command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --ratio 0.5 --delta 0.875
```

## Citation
```angular2html
@article{qin2023infobatch,
  title={InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning},
  author={Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhou, Daquan and You, Yang},
  journal={arXiv preprint arXiv:2303.04947},
  year={2023}
}
```