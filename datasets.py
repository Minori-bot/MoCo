import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import GaussianBlur, TwoCropsTransform


class ContrastiveLearningDatasets:
    def __init__(self, root_folder):
        self.root_floder = root_folder
        self.mean = {
            'cifar10': [0.49139968, 0.48215841, 0.44653091],
            'stl10': [0.49139968, 0.48215841, 0.44653091],
        }
        self.std = {
            'cifar10': [0.24703223, 0.24348513, 0.26158784],
            'stl10': [0.24703223, 0.24348513, 0.26158784],
        }

    def transform_pipeline(self, size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        return data_transforms

    def get_datasets(self, name):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_floder, train=True,
                                                transform=TwoCropsTransform(transforms.Compose([
                                                    self.transform_pipeline(32),
                                                    transforms.Normalize(self.mean['cifar10'], self.std['cifar10'])])), download=True),
            'stl10': lambda: datasets.STL10(self.root_floder, split='unlabeled',
                                                transform=TwoCropsTransform(transforms.Compose([
                                                    self.transform_pipeline(96),
                                                    transforms.Normalize(self.mean['stl10'], self.std['stl10'])])), download=True)
        }

        try:
            datasets_fn = valid_datasets[name]
        except KeyError:
            raise('no datasets in dict.')
        else:
            return datasets_fn()
