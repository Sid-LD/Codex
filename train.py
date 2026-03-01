from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.credit_fraud.config import TrainingConfig
from src.credit_fraud.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit card fraud detection model")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument(
        "--threshold-objective",
        choices=["f1", "recall@precision"],
        default="f1",
    )
    parser.add_argument("--min-precision", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        use_smote=args.use_smote,
        tune_model=not args.no_tuning,
        threshold_objective=args.threshold_objective,
        min_precision=args.min_precision,
    )
    result = run_training_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
