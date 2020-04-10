import click
from utils import get_identities, get_pairs

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
    
