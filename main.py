import torch
from utils import accuracy

x = torch.Tensor([[0.1, 0.2, 0.4, 0.1],
                  [0.5, 0.3, 0.2, 0.1],
                  [0.3, 0.2, 0.6, 0.1],
                  [0.5, 0.3, 0.2, 0.1]])
y = torch.tensor([0, 0, 0, 0], dtype=torch.long)
print(x.shape, y.shape)
#
# _, pred = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
# pred = pred.t()
# print(pred)
#
# correct = torch.eq(pred, y.view(1, -1).expand_as(pred))
# print(correct)

acc1, acc2 = accuracy(x, y, (1, 2))
print(acc1, acc2)