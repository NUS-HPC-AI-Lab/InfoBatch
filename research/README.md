## Purpose of this directory
Here we provide a code version for the research study. All steps are more straightforward.

## CIFAR Experiments
To run the CIFAR-100 example with Baseline, execute in the command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --delta 0.0
```
To run the CIFAR-100 example with InfoBatch, execute in the command line:
```angular2html
CUDA_VISIBLE_DEVICES=0 python3 cifar_example.py --model r50 --optimizer lars --max-lr 5.2 --ratio 0.5 --delta 0.875
```
