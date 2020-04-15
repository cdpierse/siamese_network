import pandas as pd
import pytest

from siamese_network.utils import (get_identities, get_unique_ids,
                                   make_pairs, split_ids,
                                   make_combinations, load_pairs)


@pytest.fixture
def get_identity_df():
    yield get_identities()


@pytest.fixture
def _get_unique_ids(get_identity_df):
    yield get_unique_ids(get_identity_df)


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
    yield ['00', '01', '02', '03', '04', '05']


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
    pairs = make_pairs(get_identity_df)
    assert type(pairs) is set
    for val in pairs:
        assert type(val) is tuple


def test_load_pairs():
    all_pairs = load_pairs()
    train_pairs = load_pairs(name='train')
    test_pairs = load_pairs(name='test')
    valid_pairs = load_pairs(name='valid')

    # assert sets are non empty
    assert(all_pairs)
    assert(train_pairs)
    assert(test_pairs)
    assert(valid_pairs)

    # assert they are in fact sets
    assert type(all_pairs) is set
    assert type(train_pairs) is set
    assert type(test_pairs) is set
    assert type(valid_pairs) is set
