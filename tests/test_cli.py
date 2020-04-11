import pytest
from utils import split_ids, get_identities, get_unique_ids


@pytest.fixture
def get_identity_df():
    return get_identities()


@pytest.fixture
def _get_unique_ids(get_identity_df):
    return get_unique_ids(get_identity_df)


def test_ids_are_list(_get_unique_ids):
    train_ids, test_ids = split_ids(_get_unique_ids, 0.9)
    assert type(train_ids) is list and type(test_ids) is list


def test_trainIds_not_in_testIds(_get_unique_ids):
    train_ids, test_ids = split_ids(_get_unique_ids, 0.9)
    train_set = set(train_ids)
    test_set = set(test_ids)
    assert len(train_set.intersection(test_set)) == 0
