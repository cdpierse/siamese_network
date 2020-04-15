import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
import PIL
import os

from siamese_network.utils import load_pairs


class SiameseDataset(CelebA):

    def __init__(self, root, max_pairs=None, split=None):
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

        self.root = root

        super().__init__(root=self.root, split=self.split)

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
                self.filename = []
                for pair in self.pairs:
                    self.filename.append(pair[0])
                    self.filename.append(pair[1])
            else:
                raise TypeError(
                    f'The value "{max_pairs}" is not a valid type for max_pairs. Must be of type int.')

        self.pair_idx = 0

    def __getitem__(self, index):
        if index % 2 == 0:
            img0_name, img1_name = self.pairs[self.pair_idx]
            img0 = PIL.Image.open(os.path.join(
                self.root, self.base_folder, "img_align_celeba", img0_name))
            img1 = PIL.Image.open(os.path.join(
                self.root, self.base_folder, "img_align_celeba", img1_name))

            self.pair_idx += 1
        else:
            # random acccess logic goes here
            pass

    def __len__(self):
        # this multiplication by 2 may seem odd
        # but for every pair (identity match) we will
        # also provide the network with a non pair
        # which will be randomly generated, thus the multiplication

        return len(self.pairs) * 2

    @classmethod
    def from_train(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_validation(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_test(cls, *args, **kwargs):
        return cls(*args, **kwargs)


sd = SiameseDataset.from_train('data/', max_pairs=1000)
