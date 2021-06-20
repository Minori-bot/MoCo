import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(models.resnet.ResNet):
    def  __init__(self, block, layers, cifar=False, hparams=None, **kwargs):
        super(ResNetEncoder, self).__init__(block, layers, **kwargs)
        self._cifar = cifar
        self._hparams = hparams
        if cifar:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

        print('** Using avgpool **')

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self._cifar:
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

class ResNet18(ResNetEncoder):
    def __init__(self, cifar=True, **kwargs):
        super(ResNet18, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2], cifar=cifar, **kwargs)

class ResNet50(ResNetEncoder):
    def __init__(self, cifar=True, hparams=None, **kwargs):
        super(ResNet50, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar=cifar, hparams=hparams, **kwargs)

class ResNet50x2d(ResNetEncoder):
    def __init__(self, cifar=True, hparams=None, **kwargs):
        kwargs['width_per_group'] = 2 * 64
        super(ResNet50x2d, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar=cifar, hparams=hparams, **kwargs)

class ResNet50x4d(ResNetEncoder):
    def __init__(self, cifar=True, hparams=None, **kwargs):
        kwargs['width_per_group'] = 4 * 64
        super(ResNet50x4d, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar=cifar, hparams=hparams, **kwargs)