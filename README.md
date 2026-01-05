# IEEE-CIS Fraud Detection: Time-Invariant Client Identification

This repository contains the code and experiments for the paper:
**"Time-Invariant Client Identification for Fraud Detection: A Feature Engineering Approach"**

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (place in dataset/ folder)
# Download from: https://www.kaggle.com/c/ieee-fraud-detection/data
# Required files: train_transaction.csv, train_identity.csv

# 4. Run all experiments
./run_experiments.sh
```

## 📁 Project Structure

```
ieee-cis_fraud_detection/
├── src/                          # Core library code
│   ├── data_loader.py           # Data loading and memory optimization
│   ├── feature_engineering.py   # Magic Features & UID construction
│   ├── train.py                 # Model training (Baseline & SOTA)
│   └── validation.py            # Time-series cross-validation
├── scripts/                      # Utility scripts
│   ├── plot_config.py           # Matplotlib styling for papers
│   ├── plot_auc.py              # ROC curve comparison
│   ├── plot_results.py          # Feature importance & EDA
│   ├── plot_shap.py             # SHAP analysis
│   ├── plot_d1_transformation.py # D1 vs D1_inv visualization
│   └── run_additional_experiments.py  # Robustness analysis
├── manuscript/                   # LaTeX paper source
│   ├── paper.tex                # Main paper
│   ├── references.bib           # Bibliography
│   └── build.sh                 # Compilation script
├── results/                      # Experiment outputs (auto-generated)
├── dataset/                      # Dataset (not in git)
├── main.py                       # Main experiment entry point
├── run_experiments.sh            # Full reproduction script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔬 Reproducing Experiments

### Full Experiment (590K samples, ~15 min)
```bash
./run_experiments.sh
```

### Quick Test (10K samples, ~2 min)
```bash
./run_experiments.sh quick
```

### Individual Experiments
```bash
# Baseline (SMOTE + XGBoost)
python main.py --mode baseline

# SOTA (Magic Features + LightGBM)
python main.py --mode sota

# Ablation (SOTA + SMOTE)
python main.py --mode ablation
```

## 📊 Expected Results

| Method | AUC | AP | Training Time |
|--------|-----|-----|---------------|
| Baseline (SMOTE+XGB) | 0.908 | 0.543 | 81s |
| **SOTA (Magic Features)** | **0.932** | 0.588 | 67s |
| Ablation (SOTA+SMOTE) | 0.935 | 0.621 | 328s |

## 📈 Generating Figures

```bash
cd scripts
python plot_results.py       # Feature importance + EDA
python plot_auc.py           # ROC curves
python plot_shap.py          # SHAP summary
python plot_d1_transformation.py  # Core insight figure
python run_additional_experiments.py  # Robustness + Sensitivity

# Compile paper
cd ../manuscript && ./build.sh
```

## 🔑 Key Innovation

The core contribution is the **Time-Invariant UID Transformation**:

```python
# D1 is "days since registration" (time-variant)
# D1_inv is "registration day" (time-invariant)
D1_inv = Day(transaction_time) - D1
```

This simple transformation enables stable user identification across transactions.

## 📜 License

MIT License
