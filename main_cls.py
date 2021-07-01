import argparse
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets

from torch.utils.data import DataLoader
from model.encoder import get_model



DIR = {
    'DATA': '..\..\dataset',
    'CHECKPOINT': '.\checkpoint',
    'LOG': '.\log',
    'WRITER': '.\writer'
}

for path in DIR.values():
    if not os.path.exists(path):
        os.mkdir(path)

parser = argparse.ArgumentParser(description='Pytorch MocoV2 linear classification')
parser.add_argument('--dataset', default='cifar', help='name for dataset, (Options: cifar, stl)')
parser.add_argument('--epochs', metavar='N', type=int, default=100, help='Number of epochs in training')
parser.add_argument('--start-epoch', metavar='N', type=int, default=0, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', metavar='N', type=int, default=256, help='Number of batch size')
parser.add_argument('--lr', metavar='LR', type=float, default=30, help='learning rate')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (drop ratio)')
parser.add_argument('--momentum', metavar='M', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--wd', metavar='WD', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', metavar='PATH', default=None, help='path to latest checkpoint')
parser.add_argument('--workers', metavar='N', type=int, default=0, help='Number of dataloader workers')
parser.add_argument('--device', default=None, help='device for training')
parser.add_argument('--model', default='resnet50',
                    help='model, (Options: resnet18, resnet50, resnet50x2d, resnet50x4d)')
parser.add_argument('--pretrained', metavar='PATH', default=None, help='path to moco pretrained checkpoint')
parser.add_argument('--dim', metavar='N', type=int, default=10, help='number of classification')


def main():
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(args.device))

    print("creating model '{}'".format(args.model))
    model = get_model(name=args.model, cifar=args.dataset == 'cifar', num_classes=args.dim)
    # freeze all layer but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                    state_dict[k[len('encoder_q.'):]] = state_dict[k]
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert msg.missing_keys == ['fc.weight', 'fc.bias']
            print(model)

            print("loaded pretrained model '{}'".format(args.pretrained))

        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

    if args.device == torch.device('cuda'):
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.device == torch.device('cuda'):
        criterion = criterion.cuda()
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(params) == 2
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            if args.device == torch.device('cpu'):
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
    PATH = os.path.join(DIR['DATA'], args.dataset)
    normlize = transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091], std=[0.24703223, 0.24348513, 0.26158784])
    train_data = None
    if args.dataset == 'cifar':
        train_data = torchvision.datasets.CIFAR10(root=PATH, transform=transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normlize
        ]), download=True, train=True)
    if args.dataset == 'stl':
        train_data = torchvision.datasets.STL10(root=PATH, transform=transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normlize
        ]), download=True, split='train')

    test_data = None
    if args.dataset == 'cifar':
        test_data = torchvision.datasets.CIFAR10(root=PATH, transform=transforms.Compose([
            transforms.ToTensor(),
            normlize
        ]), download=True, train=False)
    if args.dataset == 'stl':
        test_data = torchvision.datasets.STL10(root=PATH, transform=transforms.Compose([
            transforms.ToTensor(),
            normlize
        ]), download=True, split='test')

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True)

    # train and test
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train(model, train_loader, optimizer, criterion, args)

        test(model, test_loader, optimizer, criterion, args)

def train(model, data_lodar, optimizer, criterion, args):

    for i, (images, labels) in enumerate(data_lodar):
        if args.device == torch.device('cuda'):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    pass

def test(model, data_loader, optimizer, criterion, args):
    pass

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    main()