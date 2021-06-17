import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import ContrastiveLearningDatasets

root_folder = '../data/cifar'

datasets = ContrastiveLearningDatasets(root_folder)
data = datasets.get_datasets('cifar10')
loader = DataLoader(data, batch_size=1)

for img, _ in loader:
    q, k = img[0], img[1]
    print(q.shape, k.shape)
    print(q)
    print(k)
    break