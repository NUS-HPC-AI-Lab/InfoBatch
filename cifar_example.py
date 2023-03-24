import argparse
import datetime
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time

from infobatch import InfoBatch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--optimizer',type=str,default='lars',
                    help='different optimizers')
parser.add_argument('--label-smoothing',type=float,default=0.1)
parser.add_argument('--class-balance', default = False, action='store_true')
# onecycle scheduling arguments
parser.add_argument('--max-lr',default=0.1,type=float)
parser.add_argument('--div-factor',default=25,type=float)
parser.add_argument('--final-div',default=10000,type=float)
parser.add_argument('--num-epoch',default=200,type=int)
parser.add_argument('--pct-start',default=0.3,type=float)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--ratio',default=0.5,type=float)
parser.add_argument('--delta',default=0.875,type=float)
parser.add_argument('--model',default='r50',type=str)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

trainset = torchvision.datasets.CIFAR100(root='/tmp/cifar100', train=True, transform=train_transform,
    download=True)

#1.Substitute dataset with InfoBatch dataset, optionally set r; delta and num_epoch are needed to anneal with full data
trainset = InfoBatch(trainset, args.ratio if args.ratio else None, args.num_epoch, args.delta)

#2.Substitute sampler
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=args.shuffle, sampler = trainset.pruning_sampler())

testset = torchvision.datasets.CIFAR100(
    root='/tmp/cifar100', train=False, download=True, transform=test_transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)


# Model
print('==> Building model..')

if args.model.lower()=='r18':
    net = torchvision.models.resnet18(num_classes=100)
elif args.model.lower()=='r50':
    net = torchvision.models.resnet50(num_classes=100)
elif args.model.lower()=='r101':
    net = torchvision.models.resnet101(num_classes=100)
else:
    net = torchvision.models.resnet50(num_classes=100)

net = net.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')
test_criterion = nn.CrossEntropyLoss()

if args.optimizer.lower()=='sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
elif args.optimizer.lower()=='adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'lars':#no tensorboardX
    from lars import Lars
    optimizer = Lars(net.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'lamb':
    from lamb import Lamb
    optimizer  = Lamb(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,args.max_lr,steps_per_epoch=len(trainloader),
                                                  epochs=args.num_epoch,div_factor=args.div_factor,
                                                  final_div_factor=args.final_div,pct_start=args.pct_start)

train_acc = []
valid_acc = []

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    #3.Change loading content from loader
    for batch_idx, (inputs, targets, indices, rescale_weight) in enumerate(trainloader):
        #4.Put corresponding tensor onto device, and send back the scores
        inputs, targets, rescale_weight = inputs.to(device), targets.to(device), rescale_weight.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        trainset.__setscore__(indices.detach().cpu().numpy(),loss.detach().cpu().numpy())
        loss = loss*rescale_weight
        loss = torch.mean(loss)
        #-----------Only need to adapt above code in training------------#
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('epoch:', epoch, '  Training Accuracy:', 100. * correct / total, '  Train loss:', train_loss / len(trainloader))
    train_acc.append(correct/total)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = test_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('epoch:', epoch, '  Test Accuracy:', 100. * correct / total, '  Test loss:', test_loss/len(testloader))
    valid_acc.append(correct/total)

total_time = 0

for epoch in range(args.num_epoch):
    # 5. For epoch based implementation, update corresponding learning rate schedule according to steps this epoch
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr,
                                                           steps_per_epoch=len(trainloader),
                                                           epochs=args.num_epoch, div_factor=args.div_factor,
                                                           final_div_factor=args.final_div, pct_start=args.pct_start,
                                                           last_epoch=epoch * len(trainloader) - 1)
    end = time.time()
    train(epoch)
    total_time+=time.time()-end
    test(epoch)


print('Total saved sample forwarding: ', trainset.total_save())
print('Total training time: ', total_time)

fn = '{}{}-{}-epoch{}-batchsize{}-pct{}-labelsm{}-{}_cifar100_InfoBatch_log.json'.format(
                                                                args.model,
                                                                str(args.max_lr/args.div_factor),
                                                                str(args.max_lr),
                                                                args.num_epoch,
                                                                args.batch_size,
                                                                args.pct_start,
                                                                args.label_smoothing,
                                                                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

file = open(fn,'w+')
json.dump([total_time,trainset.total_save(),args.ratio,train_acc,valid_acc],file)