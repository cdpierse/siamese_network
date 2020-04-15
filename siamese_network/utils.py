import itertools
import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CelebA

DATA_DIR = os.path.join("data", "celeba")


def get_identities():
    return pd.read_csv(DATA_DIR + "/identity_CelebA.txt",
                       sep=' ',
                       names=['filename', 'identity_num'],
                       )


def get_pairs(df: pd.DataFrame):
    return df.identity_num.value_counts()


def get_unique_ids(identity_df) -> list:
    if isinstance(identity_df, torch.Tensor):
        return identity_df.unique().tolist()
    return identity_df.identity_num.unique()


def split_ids(ids: list, split_size=0.9) -> tuple:

    train_size = round(len(ids) * split_size)
    test_size = len(ids) - train_size
    test_ids = np.random.choice(ids, test_size)
    train_ids = list(set(ids) - set(test_ids))
    return (train_ids, list(test_ids))


def make_combinations(filenames: list):
    return list(itertools.combinations(filenames, 2))


def make_pairs(identity_df: pd.DataFrame, overwrite_cache=False, ids=None, name=None) -> pd.DataFrame:
    # create dir if it's not there
    Path("cache/").mkdir(parents=True, exist_ok=True)
    if name:
        filename = name + '_pairs_cache.pkl'
    else:
        filename = 'pairs_cache.pkl'
    if os.path.exists(os.path.join("cache", filename)) and not overwrite_cache:
        with open(os.path.join("cache", filename), "rb") as cache:
            print("using cached result from '%s'" % filename)
            return pickle.load(cache)
    if ids is not None:
        unique_ids = get_unique_ids(ids)
    else:
        unique_ids = get_unique_ids(identity_df)

    pairs = []
    for id in unique_ids:
        filenames = list(identity_df.filename[identity_df.identity_num == id])
        combinations = make_combinations(filenames)
        pairs.extend(combinations)
    pairs = set(pairs)
    with open(os.path.join("cache", filename), 'wb') as cache:
        print("saving result to cache '%s'" % filename)
        pickle.dump(pairs, cache)
    return pairs


def make_pair_split_cache_files():
    celeb_train = CelebA(root='data/', split='train')
    celeb_test = CelebA(root='data/', split='test')
    celeb_valid = CelebA(root='data/', split='valid')

    train_ids = celeb_train.identity
    test_ids = celeb_test.identity
    valid_ids = celeb_valid.identity

    id_df = get_identities()
    make_pairs(id_df, ids=train_ids, name='train', overwrite_cache=True)
    make_pairs(id_df, ids=test_ids, name='test', overwrite_cache=True)
    make_pairs(id_df, ids=valid_ids, name='valid', overwrite_cache=True)


def load_pairs(name=None):
    if name is None:
        filename = 'pairs_cache.pkl'
    else:
        filename = name + '_pairs_cache.pkl'
    if os.path.exists(os.path.join("cache", filename)):
        with open(os.path.join("cache", filename), "rb") as cache:
            print("using cached result from '%s'" % filename)
            return pickle.load(cache)
    else:
        make_pair_split_cache_files()


if __name__ == "__main__":
    vals = load_pairs('valid')
   
  

