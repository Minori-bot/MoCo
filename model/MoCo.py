import torch
import torch.nn as nn
from model.encoder import ResNet50, ResNet18, ResNet50x2d, ResNet50x4d
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, k=4096, momentum=0.999, temperature=0.07, cifar=False):
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
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, parqm_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            parqm_k.data = self.momentum * parqm_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def _maintain_queue(self, x):
        ptr = int(self.ptr)
        batch_size = x.shape[0]
        assert self.k % batch_size == 0 # for simplicity

        # maintain
        self.queue[:, ptr:ptr + batch_size] = x.T
        ptr = (ptr + batch_size) % self.k
        self.ptr[0] = ptr

    def forward(self, x_q, x_k):
        """
        Input:
            x_q: a batch of query images
            x_k: a barch of key images
        Output:
            logits, targets
        """

        q = self.encoder_q(x_q)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)
        pos = torch.bmm(q.view(q.shape[0], 1, -1), k.view(k.shape[0], -1, 1))
        pos = torch.squeeze(pos, dim=-1)
        neg = torch.mm(q, self.queue.clone().detach())

        logits = torch.cat([pos, neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._maintain_queue(k)

        return logits, labels

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
