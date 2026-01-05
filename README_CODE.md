# IEEE-CIS Fraud Detection Implementation

This project implements the experiments described in the "Revisiting the State of the Art" research proposal. It compares an "Academic Baseline" (Stacking + SMOTE) against a "Proposed SOTA" (Client Identification + Single Model).

## Directory Structure

```
ieee-cis_fraud_detection/
├── dataset/                # Place CSV files here (train_transaction.csv, etc.)
├── doc/                    # Documentation and Analysis
├── src/
│   ├── data_loader.py      # Memory reduction and loading
│   ├── feature_engineering.py # "Magic" features (UID) and Baseline preprocessing
│   ├── train.py            # LightGBM and XGBoost training logic
│   └── validation.py       # Time-Series Split strategy
├── main.py                 # Entry point
├── plot_auc.py             # ROC Curve visualization
├── plot_shap.py            # SHAP feature importance
├── requirements.txt        # Dependencies
└── run_experiments.sh      # Helper script to run full pipeline
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure data is in `dataset/`:
   - `train_transaction.csv`
   - `train_identity.csv`
   - `test_transaction.csv`
   - `test_identity.csv`

## Running Experiments

You can run the full pipeline using the provided shell script:

```bash
# Run both baseline and SOTA experiments
./run_experiments.sh

# Run in quick mode (10,000 rows only) for testing
./run_experiments.sh quick
```

### Manual Execution

**1. Run Baseline (Academic Approach)**
Standard preprocessing + SMOTE + XGBoost:
```bash
python main.py --mode baseline --output results/baseline_metrics.json
```

**2. Run SOTA (Proposed Approach)**
UID Feature Engineering + LightGBM (No SMOTE):
```bash
python main.py --mode sota --output results/sota_metrics.json
```

## Visualization

After running the experiments, generate plots:

**ROC Comparison**:
```bash
python plot_auc.py
```
Output: `results/roc_comparison.png`

**SHAP Feature Importance (for SOTA)**:
```bash
python plot_shap.py
```
Output: `results/shap_summary.png`

## Key Implementation Details

- **`src/feature_engineering.py`**: Contains the `make_uid_features` function which constructs the User ID (`card1`+`addr1`+`D1`) and simulates the "Magic Features" (aggregations by UID).
- **`src/validation.py`**: Implements strict Time-Series splitting to prevent leakage, unlike standard K-Fold.
