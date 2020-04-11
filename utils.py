import os
import pandas as pd
import numpy as np

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


def split_ids(ids: list, split_size) -> tuple:
    train_size = round(len(ids) * 0.9)
    test_size = len(ids) - train_size
    test_ids = np.random.choice(ids, test_size)
    train_ids = list(set(ids) - set(test_ids))
    return train_ids, list(test_ids)
