from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        ),
    }


def tune_random_forest(x_train, y_train, random_state: int = 42):
    model = RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": [150, 250, 350],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.8],
    }
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=12,
        scoring="f1",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_


def find_best_threshold(
    y_true,
    y_proba,
    objective: str = "f1",
    min_precision: float = 0.9,
) -> ThresholdResult:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)

    if objective == "recall@precision":
        valid = np.where(precision >= min_precision)[0]
        if len(valid) == 0:
            idx = int(np.nanargmax((2 * precision * recall) / (precision + recall + 1e-12)))
        else:
            idx = valid[np.argmax(recall[valid])]
    else:
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
        idx = int(np.nanargmax(f1_scores))

    t = float(np.clip(thresholds[idx], 0.0, 1.0))
    y_pred = (y_proba >= t).astype(int)
    f1 = f1_score(y_true, y_pred)
    return ThresholdResult(t, float(precision[idx]), float(recall[idx]), float(f1))
