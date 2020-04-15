import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA
from siamese_network.utils import load_pairs


class SiameseDataset(CelebA):

    def __init__(self, max_pairs=None, split=None):
        self.split = split
        if self.split:
            super().__init__(root='data/', split=self.split)
        else:
            super().__init__(root='data/')

        self.max_pairs = max_pairs

        self.pairs = {}
        if self.split == 'train':
            self.pairs = load_pairs('train')
        elif self.split is None:
            self.pairs = load_pairs()
        elif self.split == 'test':
            self.pairs = load_pairs('test')
        elif self.split == 'valid':
            self.pairs = load_pairs('valid')
        else:
            raise ValueError(f"Split type {self.split} is not recognized")

        if max_pairs:
            # method to randomly indices of pairs
            # and choose `max_pairs` amount
            # from the dataset to be our files
            pass

    def __getitem__(self, index):
        """__getitem__ is overriden
        from the CelebA dataset to 
        fetch pairs of images sequentially

        Args:
            index ([type]): [description]
        """
        print('yay overwite ')

    @classmethod
    def shuffle(cls):
        pass

    @classmethod
    def get_test_dataset(cls):
        pass

    @classmethod
    def from_validation(cls):
        return cls(split='valid')

    @classmethod
    def from_test(cls):
        return cls(split='test')


sd = SiameseDataset(split='train')
for pair in sd.pairs:
    print(pair)
    break
