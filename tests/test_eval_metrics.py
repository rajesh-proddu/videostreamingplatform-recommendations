"""Tests for the eval ranking metrics."""

import math

import pytest

from evals.metrics import ndcg_at_k, precision_at_k, recall_at_k

REL = {"a": 3, "b": 1, "c": 0}


def test_precision_counts_graded_hits_over_k():
    assert precision_at_k(["a", "c", "b", "x"], REL, 4) == 0.5
    assert precision_at_k(["a"], REL, 2) == 0.5  # short list is penalised


def test_recall():
    assert recall_at_k(["a", "x"], REL, 10) == 0.5
    assert recall_at_k(["x"], {}, 10) == 1.0


def test_ndcg_perfect_and_reversed():
    assert ndcg_at_k(["a", "b"], REL, 10) == pytest.approx(1.0)
    reversed_score = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], REL, 10) == pytest.approx(reversed_score)


def test_ndcg_truncates_at_k():
    assert ndcg_at_k(["x", "a"], REL, 1) == 0.0


def test_ndcg_nothing_relevant():
    assert ndcg_at_k([], {}, 10) == 1.0
    assert ndcg_at_k(["x"], {"x": 0}, 10) == 0.0
