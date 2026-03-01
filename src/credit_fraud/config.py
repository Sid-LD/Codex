from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path
    output_dir: Path = Path("outputs")
    target_column: str = "Class"
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = False
    tune_model: bool = True
    threshold_objective: str = "f1"  # choose from f1, recall@precision
    min_precision: float = 0.90
