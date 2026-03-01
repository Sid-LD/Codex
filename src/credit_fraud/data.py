from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]


def load_credit_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
):
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
