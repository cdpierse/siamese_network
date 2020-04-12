import itertools
import os

import numpy as np
import pandas as pd
from functools import lru_cache
import pickle
from pathlib import Path
DATA_DIR = os.path.join("data", "celeba-dataset")


def get_identities():
    return pd.read_csv(DATA_DIR + "/identity_CelebA.txt",
                       sep=' ',
                       names=['filename', 'identity_num'],
                       )


def get_pairs(df: pd.DataFrame):
    return df.identity_num.value_counts()


def get_unique_ids(identity_df: pd.DataFrame) -> list:
    return identity_df.identity_num.unique()


def split_ids(ids: list, split_size=0.9) -> tuple:
    train_size = round(len(ids) * split_size)
    test_size = len(ids) - train_size
    test_ids = np.random.choice(ids, test_size)
    train_ids = list(set(ids) - set(test_ids))
    return (train_ids, list(test_ids))


def make_combinations(filenames: list):
    return list(itertools.combinations(filenames, 2))


def make_pairs(identity_df: pd.DataFrame, overwrite_cache=False) -> pd.DataFrame:
    # create dir if it's not there
    Path("cache/").mkdir(parents=True, exist_ok=True)
    filename = 'pairs_cache.pkl'
    if os.path.exists(os.path.join("cache", "pairs_cache.pkl")) and not overwrite_cache:
        with open(os.path.join("cache", "pairs_cache.pkl"), "rb") as cache:
            print("using cached result from '%s'" % filename)
            return pickle.load(cache)

    unique_ids = get_unique_ids(identity_df)
    pairs = []
    for id in unique_ids:
        filenames = list(identity_df.filename[identity_df.identity_num == id])
        combinations = make_combinations(filenames)
        pairs.extend(combinations)
    pairs = set(pairs)
    with open(os.path.join("cache", "pairs_cache.pkl"), 'wb') as cache:
        print("saving result to cache '%s'" % filename)
        pickle.dump(pairs, cache)
    return pairs
