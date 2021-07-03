# MoCo
A unofficial PyTorch implementation of [MoCo](https://arxiv.org/pdf/1911.05722.pdf).

There are some difficult with official implementation, only using model ResNet18 and ResNet50 in this repo and training pretrained model on one v100 GPU.

## Requirements

```
$ conda activate env
$ pip install -r requirements.txt
```

## Usage

### Train
```
$ python main_train.py --model resnet18 --cos
```


```
main_train.py [-h] [--dataset DATASET] [--epochs EPOCHS]
                     [--start-epoch START_EPOCH] [--batch-size BATCH_SIZE]
                     [--lr LR] [--schedule [SCHEDULE [SCHEDULE ...]]]
                     [--momentum MOMENTUM] [--wd WD] [--checkpoint CHECKPOINT]
                     [--workers WORKERS] [--cos] [--device DEVICE]
                     [--model MODEL] [--moco-dim MOCO_DIM] [--moco-k MOCO_K]
                     [--moco-m MOCO_M] [--moco-t MOCO_T]

Pytorch MocoV2 training

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     name for dataset, (Options: cifar, stl)
  --epochs EPOCHS       Number of epochs in training
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  --batch-size BATCH_SIZE
                        Number of batch size
  --lr LR               learning rate
  --schedule [SCHEDULE [SCHEDULE ...]]
                        learning rate schedule (drop 0.8)
  --momentum MOMENTUM   momentum for SGD
  --wd WD               weight decay
  --checkpoint CHECKPOINT
                        path to latest checkpoint
  --workers WORKERS     Number of dataloader workers
  --cos                 using cosine lr schedule
  --device DEVICE       device for training
  --model MODEL         model, (Options: resnet18, resnet50, resnet50x2d,
                        resnet50x4d)
  --moco-dim MOCO_DIM   feature dimension
  --moco-k MOCO_K       size fo queue, number of negative keys
  --moco-m MOCO_M       momentum for key encoder
  --moco-t MOCO_T       temperature in InfoNCE
```

### Linear Evaluation

```
$ python main_cls.py --model resnet18 --lr 0.3
```


```
main_cls.py [-h] [--dataset DATASET] [--epochs N] [--start-epoch N]
                   [--batch-size N] [--lr LR]
                   [--schedule [SCHEDULE [SCHEDULE ...]]] [--momentum M]
                   [--wd WD] [--checkpoint PATH] [--workers N]
                   [--device DEVICE] [--model MODEL] [--pretrained PATH]
                   [--dim N]

Pytorch MocoV2 linear classification

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     name for dataset, (Options: cifar, stl)
  --epochs N            Number of epochs in training
  --start-epoch N       manual epoch number (useful on restarts)
  --batch-size N        Number of batch size
  --lr LR               learning rate
  --schedule [SCHEDULE [SCHEDULE ...]]
                        learning rate schedule (drop ratio)
  --momentum M          momentum for SGD
  --wd WD               weight decay
  --checkpoint PATH     path to latest checkpoint
  --workers N           Number of dataloader workers
  --device DEVICE       device for training
  --model MODEL         model, (Options: resnet18, resnet50, resnet50x2d,
                        resnet50x4d)
  --pretrained PATH     path to moco pretrained checkpoint
  --dim N               number of classification
```

## Performance

We train encoder by using resnet18 and resnet50, with dataset CIFAR10 and STL10, optimizer SGD. And We freeze all parameters but fc layer of resent model to training a linear classifier evaluating our model.

This is the performance:

|  Dataset  |  Architecture  |  Queue size  |  Feature dimensions  |  Epochs  |  Top1 %  |  Top5 %  |
|  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |
| CIFAR10  | ResNet18 | 4096 | 128 | 500 | 81.06 | 99.13 |
| CIFAR10  | ResNet50 | 4096 | 128 | 500 | 84.03 | 99.40 |
| CIFAR10  | ResNet50 | 16384 | 128 | 500 | 84.57 | 99.43 |
