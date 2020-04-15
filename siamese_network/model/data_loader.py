import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA

from siamese_network.utils import load_pairs


class SiameseDataset(CelebA):

    def __init__(self, max_pairs=None, split=None):
        """__init__ [summary]

        Args:
            max_pairs (int, optional): [description]. Defaults to None.
            split (str, optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
            TypeError: [description]
        """
        if split:
            self.split = split
        else:
            self.split = 'all'

        super().__init__(root='data/', split=self.split)

        self.max_pairs = max_pairs

        self.pairs = {}
        if self.split == 'all':
            self.pairs = load_pairs()
        elif self.split == 'train':
            self.pairs = load_pairs('train')
        elif self.split == 'test':
            self.pairs = load_pairs('test')
        elif self.split == 'valid':
            self.pairs = load_pairs('valid')
        else:
            raise ValueError(f"Split type {self.split} is not recognized")

        if max_pairs:
            if isinstance(max_pairs, int):
                self.pairs = random.sample(self.pairs, max_pairs)
            else:
                raise TypeError(
                    f'The value "{max_pairs}" is not a valid type for max_pairs. Must be of type int.')

        self.pairs = torch.as_tensor(self.pairs)

    def __getitem__(self, index):
        """__getitem__ is overriden
        from the CelebA dataset to 
        fetch pairs of images sequentially

        Args:
            index ([type]): [description]
        """
        print('yay overwite ')

    @classmethod
    def from_train(cls):
        return cls(split='train')

    @classmethod
    def from_validation(cls):
        return cls(split='valid')

    @classmethod
    def from_test(cls):
        return cls(split='test')


sd = SiameseDataset(max_pairs=100)
print(type(sd.pairs))
