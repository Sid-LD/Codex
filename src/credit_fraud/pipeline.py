from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

from .config import TrainingConfig
from .data import load_credit_data, split_data
from .evaluate import (
    compute_metrics,
    generate_classification_report,
    save_confusion_matrix,
    save_report,
)
from .modeling import build_models, find_best_threshold, tune_random_forest


def run_training_pipeline(config: TrainingConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_credit_data(config.data_path)
    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column=config.target_column,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    if config.use_smote:
        sampler = SMOTE(random_state=config.random_state)
        x_train, y_train = sampler.fit_resample(x_train, y_train)

    models = build_models(config.random_state)

    trained = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]
        threshold = find_best_threshold(
            y_test,
            proba,
            objective=config.threshold_objective,
            min_precision=config.min_precision,
        )
        y_pred = (proba >= threshold.threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, proba)
        trained[name] = {
            "model": model,
            "metrics": metrics,
            "threshold": threshold,
            "y_pred": y_pred,
            "y_proba": proba,
        }

    best_name = max(trained, key=lambda n: trained[n]["metrics"]["f1"])

    if config.tune_model:
        tuned_model, best_params = tune_random_forest(
            x_train,
            y_train,
            random_state=config.random_state,
        )
        tuned_proba = tuned_model.predict_proba(x_test)[:, 1]
        tuned_threshold = find_best_threshold(
            y_test,
            tuned_proba,
            objective=config.threshold_objective,
            min_precision=config.min_precision,
        )
        tuned_pred = (tuned_proba >= tuned_threshold.threshold).astype(int)
        tuned_metrics = compute_metrics(y_test, tuned_pred, tuned_proba)
        trained["random_forest_tuned"] = {
            "model": tuned_model,
            "metrics": tuned_metrics,
            "threshold": tuned_threshold,
            "y_pred": tuned_pred,
            "y_proba": tuned_proba,
            "best_params": best_params,
        }
        best_name = max(trained, key=lambda n: trained[n]["metrics"]["f1"])

    best = trained[best_name]
    model_path = config.output_dir / "best_model.joblib"
    metadata_path = config.output_dir / "model_metadata.joblib"
    report_path = config.output_dir / "evaluation_report.json"
    confusion_matrix_path = config.output_dir / "confusion_matrix.png"
    leaderboard_path = config.output_dir / "model_leaderboard.csv"

    joblib.dump(best["model"], model_path)
    joblib.dump(
        {
            "best_model_name": best_name,
            "threshold": best["threshold"].threshold,
            "metrics": best["metrics"],
        },
        metadata_path,
    )

    report = generate_classification_report(y_test, best["y_pred"])
    save_report(best["metrics"], report, report_path)
    save_confusion_matrix(y_test, best["y_pred"], confusion_matrix_path)

    leaderboard = pd.DataFrame(
        [
            {
                "model": name,
                **payload["metrics"],
                "decision_threshold": payload["threshold"].threshold,
            }
            for name, payload in trained.items()
        ]
    ).sort_values(by="f1", ascending=False)
    leaderboard.to_csv(leaderboard_path, index=False)

    return {
        "best_model": best_name,
        "best_metrics": best["metrics"],
        "best_threshold": best["threshold"].threshold,
        "artifacts": {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "report": str(report_path),
            "confusion_matrix": str(confusion_matrix_path),
            "leaderboard": str(leaderboard_path),
        },
    }
