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
def create_train_test_files(split_size):
    # Method for creating train and test index files
    # * should firstly get all the identities and according to a train/test split split them up
    # into unique non crossover groups
    # * from there I'll need to work out a way to create an even amount of pairs/non-pairs
    #
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
