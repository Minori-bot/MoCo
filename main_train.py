import argparse
import math
import os
import shutil
import torch
import logging
import tensorboardX
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datasets
from torch.utils.data import DataLoader
from model.MoCo import MoCo
from utils import AverageMeter, accuracy

DIR = {
    'DATA': '..\..\dataset',
    'CHECKPOINT': '.\checkpoint',
    'LOG': '.\log',
    'WRITER': '.\writer'
}

for path in DIR.values():
    if not os.path.exists(path):
        os.mkdir(path)

parser = argparse.ArgumentParser(description='Pytorch MocoV2 training')
parser.add_argument('--dataset', default='cifar', help='name for dataset, (Options: cifar, stl)')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs in training')
parser.add_argument('--start-epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', type=int, default=256, help='Number of batch size')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
parser.add_argument('--schedule', default=[120, 160, 200, 240, 280, 320], nargs='*', type=int,
                    help='learning rate schedule (drop 0.8)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', default=None, help='path to latest checkpoint')
parser.add_argument('--workers', type=int, default=0, help='Number of dataloader workers')
parser.add_argument('--cos', action='store_true', help='using cosine lr schedule')
parser.add_argument('--device', default=None, help='device for training')
parser.add_argument('--model', default='resnet50',
                    help='model, (Options: resnet18, resnet50, resnet50x2d, resnet50x4d)')

# moco specific configs
parser.add_argument('--moco-dim', type=int, default=128, help='feature dimension')
parser.add_argument('--moco-k', type=int, default=4096, help='size fo queue, number of negative keys')
parser.add_argument('--moco-m', type=float, default=0.999, help='momentum for key encoder')
parser.add_argument('--moco-t', type=float, default=0.07, help='temperature in InfoNCE')

def main():
    args = parser.parse_args()
    writer = tensorboardX.SummaryWriter(os.path.join(DIR['WRITER'], 'train'),
                                        filename_suffix='-' + args.dataset + '-' + args.model)
    logging.basicConfig(filename=os.path.join(DIR['LOG'], 'train.log'), level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    args.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print('Using device {}'.format(args.device))

    print("=> creating model '{}'".format(args.model))
    model = MoCo(args.model, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.dataset == 'cifar')
    print(model)
    if args.device == torch.device('cuda'):
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.device == torch.device('cuda'):
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

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
    print("=> loading dataset in '{}'".format(PATH))
    dataset = datasets.ContrastiveLearningDatasets(root_folder=PATH)
    train_data = dataset.get_datasets(args.dataset+'10')
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        loss, top1, top5 = train(model, train_loader, optimizer, criterion, args)
        logging.info('[Train] epoch: {}, {}, {}, {}'.format(epoch, loss, top1, top5))
        writer.add_scalar("loss", loss.avg, global_step=epoch)
        writer.add_scalar("top1", top1.avg, global_step=epoch)
        writer.add_scalar("top5", top5.avg, global_step=epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, is_best=False, filename=os.path.join(DIR['CHECKPOINT'], 'checkpoint_{:03d}.pth.tar'.format(epoch)))

    writer.close()
    logging.info('training finished')

def train(model, train_loader, optimizer, criterion, args):
    epoch_loss = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.train()
    for i, (images, _) in enumerate(train_loader):
        if args.device == torch.device('cuda'):
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)
        output, target = model(images[0], images[1])
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        epoch_loss.maintain(loss.item(), images[0].shape[0])
        top1.maintain(acc1.item(), images[0].shape[0])
        top5.maintain(acc5.item(), images[0].shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss, top1, top5

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(DIR['CHECKPOINT'], 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.8 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    main()