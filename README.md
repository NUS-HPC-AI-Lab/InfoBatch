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