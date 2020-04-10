import os
import pandas as pd

DATA_DIR = os.path.join("data", "celeba-dataset")


def get_identities():
    return pd.read_csv(DATA_DIR + "/identity_CelebA.txt",
                       sep=' ',
                       names=['filename', 'identity_num'],
                       )


def get_pairs(df: pd.DataFrame):
    return df.identity_num.value_counts()
