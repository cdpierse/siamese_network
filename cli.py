import click
from utils import get_identities, get_pairs, split_ids
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
    unique_ids = (get_identities()).identity_num.unique()
    train_size = round(len(unique_ids) * 0.9)
    test_size = len(unique_ids) - train_size

    test_ids = np.random.choice(unique_ids, test_size)
    train_ids = list(set(unique_ids) - set(test_ids))
    print(train_ids)


if __name__ == "__main__":
    main()
    # df = get_identities()
    # pairs = get_pairs(df)
    # # print(len(pairs[pairs.values > 1]))
