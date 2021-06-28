from model.MoCo import MoCo
import torch

net = MoCo(encoder='resnet18', k=10, cifar=True).cuda()
x = torch.randn(1, 3, 32, 32).cuda()
y = torch.randn(1, 3, 32, 32).cuda()
X = [x, y]
print(X)

Y = net(x, y)
print(Y)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print('Using device {}'.format(device))
print(device)
print(device == torch.device('cuda'))