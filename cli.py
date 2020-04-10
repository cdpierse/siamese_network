import click
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


@click.group()
def main():
    pass


@main.command()
def hello():
    click.echo('Hello, Charles')


@main.command()
@click.option('--size', default=1000, help='size of the pair and non pairs file')
def create_pair_file(size):
    pass


if __name__ == "__main__":
    main()
    # df = get_identities()
    # pairs = get_pairs(df)
    # # print(len(pairs[pairs.values > 1]))
    
