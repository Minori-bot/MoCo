import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datasets
from torch.utils.data import DataLoader
from model.MoCo import MoCo

DIR = '..\..\dataset'

parser = argparse.ArgumentParser(description='Pytorch MocoV2 training')
parser.add_argument('--dataset', default='cifar', help='name for dataset, (Options: cifar, stl)')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs in training')
parser.add_argument('--start-epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', type=int, default=256, help='Number of batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', default=None, help='path to latest checkpoint')
parser.add_argument('--workers', type=int, default=16, help='Number of dataloader workers')
parser.add_argument('--model', default='resnet50',
                    help='model, (Options: resnet18, resnet50, resnet50x2d, resnet50x4d)')

# moco specific configs
parser.add_argument('--moco-dim', type=int, default=128, help='feature dimension')
parser.add_argument('--moco-k', type=int, default=65536, help='size fo queue, number of negative keys')
parser.add_argument('--moco-m', type=float, default=0.999, help='momentum for key encoder')
parser.add_argument('--moco-t', type=float, default=0.07, help='temperature in InfoNCE')

def main():
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('Using device {}'.format(device))

    print("=> creating model '{}'".format(args.model))
    model = MoCo(args.model, args.dim, args.k, args.m, args.t, args.dataset=='cifar')
    print(model)
    if device == torch.device('cuda'):
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    if device == torch.device('cuda'):
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            if device == torch.device('cpu'):
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
            else:
                # just using single gpu
                checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))

    cudnn.benchmark = True

    # get dataset
    PATH = os.path.join(DIR, args.dataset)
    print("=> loading dataset in ''".format(PATH))
    dataset = datasets.ContrastiveLearningDatasets(root_folder=PATH)
    if args.dataset == 'cifar':
        train_data = dataset.get_datasets(args.dataset+'10')
    else:
        train_data = dataset.get_datasets(args.dataset+'10')
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=True)