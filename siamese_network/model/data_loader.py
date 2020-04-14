import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA


class SiameseDataset(CelebA):

    def __init__(self):
        super().__init__(root='data/')

    def __getitem__(self, index):
        print('yay overwite ')
    


sd = SiameseDataset()
print(sd.identity[0:100])
