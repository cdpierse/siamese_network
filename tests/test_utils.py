import pandas as pd
import pytest

from siamese_network.utils import (get_identities, get_unique_ids,
                                   make_non_pairs, make_pairs, split_ids,
                                   make_combinations)


@pytest.fixture
def get_identity_df():
    return get_identities()


@pytest.fixture
def _get_unique_ids(get_identity_df):
    return get_unique_ids(get_identity_df)


def test_split_ids_are_lists(_get_unique_ids):
    train_ids, test_ids = split_ids(_get_unique_ids, 0.9)
    assert type(train_ids) is list and type(test_ids) is list


def test_split_ids_are_distinct(_get_unique_ids):
    train_ids, test_ids = split_ids(_get_unique_ids, 0.9)
    train_set = set(train_ids)
    test_set = set(test_ids)
    assert len(train_set.intersection(test_set)) == 0


@pytest.fixture
def get_dummy_combination_data():
    return ['00', '01', '02', '03', '04', '05']


def test_make_combinations(get_dummy_combination_data):
    combinations = make_combinations(get_dummy_combination_data)
    # check return type is a list...
    assert type(combinations) is list

    # that contains tuples
    for element in combinations:
        assert type(element) is tuple

    # quick way to check uniqueness of all tuples
    assert len(list(set(combinations))) == len(combinations)


def test_make_pairs(get_identity_df):
    pair_df = make_pairs(get_identity_df)
    assert isinstance(pair_df, pd.DataFrame)
    column_names = ['filename1', 'filename2', 'identity_num']
    assert column_names == list(pair_df.columns)


def test_make_non_pairs(get_identity_df):
    pass

