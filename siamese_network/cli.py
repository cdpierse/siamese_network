import click
from utils import get_identities, get_pairs, split_ids, get_unique_ids
import random
import numpy as np


@click.group()
def main():
    pass


@main.command()
def hello():
    click.echo('Hello, Charles')


@main.command()
@click.option('--split_size', default=0.9, help='size of the train test split')
@click.option('--total_samples', default=10000, help="""the total number of image pairs and non_pairs
              to be used. Train and test will be derived from this. Defaults to 10,000, can theorectically
              go up to ~4 million to keep class parity as there are ~2m possible pairs""")
def create_train_test_directory(split_size, total_samples):
    """
    For tomorrows' Charles - you need to do some reworking
    as you need to have a way to create a suitable method for 
    seperating train and test files. Make pairs right now
    does not store identity info and returns a cached result. 
    We may need to bundle two files into the cache train and test. 
    """
    identity_df = get_identities()
    unique_ids = get_unique_ids(identity_df)
    train_ids, test_ids = split_ids(unique_ids, split_size)
    print(len(train_ids), len(test_ids))
    print(identity_df[identity_df.identity_num == 2880])


if __name__ == "__main__":
    main()
    # df = get_identities()
    # pairs = get_pairs(df)
    # # print(len(pairs[pairs.values > 1]))
