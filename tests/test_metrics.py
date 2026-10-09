import numpy as np

from src.metrics import recall_at_precision


def test_perfect_ranking_gives_full_recall():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert recall_at_precision(y, scores, min_p=0.8) == 1.0


def test_returns_zero_when_precision_unreachable():
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.7])
    assert recall_at_precision(y, scores, min_p=0.8) == 0.0