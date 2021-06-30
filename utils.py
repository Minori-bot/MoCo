import random
import torch
from PIL import ImageFilter

class TwoCropsTransform:
    def __init__(self, base_transform=None):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur:
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def maintain(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    batch_size = output.shape[0]
    maxk = max(topk)
    _, pred = torch.topk(output, k=maxk, largest=True, sorted=True)
    pred = pred.t()
    correct = torch.eq(pred, target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.sum(correct[:k].any(dim=0).float())
        res.append(correct_k * (100 / batch_size))
    return res