# Credit Card Fraud Detection Project (Internship-Ready)

This project upgrades the baseline GeeksforGeeks workflow into a production-style ML project with:

- **Imbalance-aware training** (class weights + optional SMOTE).
- **Model leaderboard** (Random Forest, Logistic Regression, Tuned Random Forest).
- **Threshold optimization** for better fraud catch-rate/precision tradeoff.
- **Rich evaluation artifacts** (JSON report, confusion matrix image, leaderboard CSV).
- **Reusable inference script** for batch scoring new transactions.

## Project Structure

```text
.
├── infer.py
├── train.py
├── requirements.txt
├── README.md
├── src/
│   └── credit_fraud/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── modeling.py
│       └── pipeline.py
└── tests/
    └── test_modeling.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Use the `creditcard.csv` dataset with columns:
`Time, V1...V28, Amount, Class`.

Example location:

```bash
mkdir -p data
# place creditcard.csv at data/creditcard.csv
```

## Train

### Standard training

```bash
python train.py --data-path data/creditcard.csv
```

### More aggressive fraud detection (high recall with minimum precision)

```bash
python train.py \
  --data-path data/creditcard.csv \
  --use-smote \
  --threshold-objective recall@precision \
  --min-precision 0.95
```

## Inference

Score new transactions (must have same feature columns used in training, without `Class`):

```bash
python infer.py --input-csv data/new_transactions.csv
```

## Output Artifacts

Training writes into `outputs/`:

- `best_model.joblib`
- `model_metadata.joblib`
- `evaluation_report.json`
- `confusion_matrix.png`
- `model_leaderboard.csv`

## Extra Features You Can Showcase in Internship Applications

1. **Imbalanced-learning strategy** with optional oversampling.
2. **Model selection pipeline** instead of single-model training.
3. **Threshold-aware decisions** aligned to business goals.
4. **Portable inference utility** for practical deployment.
5. **Reproducibility** through configuration-driven training.

## Suggested Next Upgrades

- Add experiment tracking (MLflow / Weights & Biases).
- Build FastAPI endpoint + Docker deployment.
- Add SHAP explainability dashboard for flagged transactions.
- Add drift monitoring jobs for real-time model health.
