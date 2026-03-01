import numpy as np

from src.credit_fraud.modeling import find_best_threshold


def test_find_best_threshold_f1_returns_valid_probability():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.20, 0.35, 0.55, 0.85, 0.99])

    result = find_best_threshold(y_true, y_proba, objective="f1")

    assert 0.0 <= result.threshold <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_find_best_threshold_recall_at_precision_prefers_high_precision():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.4, 0.6, 0.61, 0.7, 0.9, 0.95])

    result = find_best_threshold(
        y_true,
        y_proba,
        objective="recall@precision",
        min_precision=0.9,
    )

    assert result.precision >= 0.9
