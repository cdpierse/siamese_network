import os
import pandas as pd
import numpy as np
import itertools
import random


def read_identities():
    filedir = os.path.join(os.getcwd(), 'data', 'celeba-dataset')
    return pd.read_csv(os.path.join(filedir, 'identity_CelebA.txt'), delimiter=' ', names=['Filename', 'identity_no'])


def sample_faces(dataframe, sample_size):
    """Randomly sample entire dataframe for sample of size (sample_size∏
    """
    return dataframe.sample(n=sample_size, random_state=42)


def find_pairs(dataframe):

    pairs = pd.DataFrame(
        data=[], columns=['file1', 'file2', 'identity_no', 'match'])

    pair_count = 0
    for row in range(len(dataframe)):
        identity_val = dataframe['identity_no'].iloc[row]
        pair_list = dataframe.Filename[dataframe.identity_no ==
                                       identity_val].values
        if len(pair_list) >= 2:
            for f1, f2 in itertools.combinations(pair_list, r=2):
                pairs = pairs.append(
                    {'file1': f1, 'file2': f2, 'identity_no': identity_val, 'match': 'Yes'}, ignore_index=True)
                # pair_count += 1
                # if pair_count % 500 == 0:
                #     print(pair_count)

    return pairs


def construct_final_df(sample_size):
    """ Method to create dataframe made up of an equal amount of pairs
    and non pairs. Firstly reads in dataframe of a given sample size, 
    then finds all the identity pairs in that df. After that we 
    create another df that's made up of filenames where the idenity
    number is not a match.
    
    Returns:
        [dataframe] -- [dataframe of equal numbers of pairs and non-pairs]
    """
    df = sample_faces(read_identities(), sample_size)
    pairs = find_pairs(df)
    pair_length = len(pairs)

    indices = list(range(len(df)))
    random.shuffle(indices)
    for i in range(len(indices) - 1):
        if df.identity_no.iloc[indices[i]] != df.identity_no.iloc[indices[i + 1]] and len(pairs) < pair_length * 2:
            pairs = pairs.append({'file1': df.Filename.iloc[indices[i]], 'file2': df.Filename.iloc[indices[i + 1]], 'match': 'No'},
                                 ignore_index=True)

    return pairs.sample(frac=1)


def get_pair_list():
    filepath = os.path.join(os.getcwd(), 'data',
                            'celeba-dataset', 'pairs_non_pairs.csv')
    return pd.read_csv(filepath)


if __name__ == '__main__':
    df = construct_final_df(5800)
    file_path = os.path.join(
        os.getcwd(), 'data', 'celeba-dataset', 'pairs_non_pairs.csv')
    df.to_csv(file_path, index=False)
