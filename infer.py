from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run fraud inference with saved model")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/best_model.joblib"))
    parser.add_argument("--metadata-path", type=Path, default=Path("outputs/model_metadata.joblib"))
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/predictions.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model_path)
    metadata = joblib.load(args.metadata_path)
    threshold = metadata["threshold"]

    df = pd.read_csv(args.input_csv)
    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = probabilities
    out["predicted_class"] = predictions
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(json.dumps({"output_csv": str(args.output_csv), "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
