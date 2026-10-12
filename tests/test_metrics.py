import numpy as np
import pytest

from vectorforge.metrics import recall_at_k


def test_perfect_recall():
    ids = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(ids, ids, 3) == pytest.approx(1.0)


def test_order_does_not_matter():
    found = np.array([[3, 1, 2]])
    true = np.array([[1, 2, 3]])
    assert recall_at_k(found, true, 3) == pytest.approx(1.0)


def test_partial_recall_example_from_docstring():
    found = np.array([[7, 2, 4]])
    true = np.array([[4, 7, 9]])
    assert recall_at_k(found, true, 3) == pytest.approx(2 / 3)


def test_mean_over_queries():
    found = np.array([[1, 2], [9, 8]])
    true = np.array([[1, 2], [1, 2]])
    assert recall_at_k(found, true, 2) == pytest.approx(0.5)


def test_uses_only_first_k_columns():
    # ground truth files store 100 neighbours per query; recall@1 must look at column 0 only
    found = np.array([[5, 1]])
    true = np.array([[1, 5]])
    assert recall_at_k(found, true, 1) == pytest.approx(0.0)
    assert recall_at_k(found, true, 2) == pytest.approx(1.0)
