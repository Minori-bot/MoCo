import torch
import torch.nn as nn
from model.encoder import ResNet50, ResNet18, ResNet50x2d, ResNet50x4d

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, k=65536, momentum=0.999, temperature=0.07, cifar=False):
        super(MoCo, self).__init__()
        self.resnet_list = ['resnet18', 'resnet50', 'resnet50x2d', 'resnet50x4d']
        assert self._is_resnet(encoder), \
            "Invalid encoder architecture. The encoder must be resnet in [resnet18, resnet50, resnet50x2d, resnet50x4d]"

        self.k = k
        self.momentum = momentum
        self.temperature = temperature
        self.encoder_q = self._get_encoder(encoder, cifar, dim)
        self.encoder_k = self._get_encoder(encoder, cifar, dim)

        dim_mlp = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, k))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

    def _is_resnet(self, encoder):
        return encoder in self.resnet_list

    def _get_encoder(self, encoder, cifar, dim):
        if encoder == 'resnet18':
            return ResNet18(cifar=cifar, num_classes=dim)
        elif encoder == 'resnet50':
            return ResNet50(cifar=cifar, num_classes=dim)
        elif encoder == 'resnet50x2d':
            return ResNet50x2d(cifar=cifar, num_classes=dim)
        else:
            return ResNet50x4d(cifar=cifar, num_classes=dim)
