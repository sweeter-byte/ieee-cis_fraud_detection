"""
Main Experiment Runner for IEEE-CIS Fraud Detection
"""
import argparse
import json
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.train import train_baseline_model, train_sota_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(mode: str, data_path: str, output_path: str, quick_run: bool = False, use_smote: bool = False):
    """
    Run the specified experiment mode.
    
    Args:
        mode: 'baseline' or 'sota'
        data_path: Path to dataset
        output_path: Path to save results JSON
        quick_run: If True, use a small sample for testing code
        use_smote: Force usage of SMOTE (mainly for ablation in SOTA mode)
    """
    logger.info(f"Starting Experiment: {mode.upper()} (SMOTE={use_smote})")
    
    # 1. Load Data
    sample_size = 10000 if quick_run else None
    if quick_run:
        logger.warning("Quick run mode enabled: Loading only 10,000 rows.")
        
    df = load_data(data_path, sample_size=sample_size, is_train=True)
    
    # 2. Feature Engineering
    if mode == 'baseline':
        # Baseline typically uses SMOTE by default, but we can respect the flag or enforce it
        # Here we keep baseline behavior as-is (SMOTE+XGB) unless we rethink architecture
        # But for this function, if mode is baseline, we assume the specific baseline function handles it
        df = engineer_features(df, use_magic=False)
        model_features = [c for c in df.columns if c not in ['isFraud', 'TransactionID', 'TransactionDT']]
    elif mode == 'sota':
        df = engineer_features(df, use_magic=True)
        # SOTA typically uses raw features + magic features, but excludes UID raw strings
        exclude = ['isFraud', 'TransactionID', 'TransactionDT', 'uid', 'uid_encoded']
        model_features = [c for c in df.columns if c not in exclude]
        
    logger.info(f"Feature Engineering Complete. Feature count: {len(model_features)}")
    
    # 3. Train
    if mode == 'baseline':
        # Baseline uses SMOTE + XGBoost
        # We don't change this existing function for now to preserve replication accuracy
        result = train_baseline_model(df, model_features)
    elif mode == 'sota':
        # SOTA uses Magic Features + LightGBM
        # Now supports optional SMOTE for ablation
        result = train_sota_model(df, model_features, use_smote=use_smote)
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    # 4. Save Results
    metrics = result['metrics']
    metrics['training_time'] = result['training_time']
    metrics['mode'] = mode
    
    # Convert numpy types to native python for JSON serialization
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            metrics[k] = float(v)
            
    logger.info(f"Experiment Complete. Metrics: {metrics}")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Results saved to {output_path}")
    
    # Optionally save predictions for visualization
    # In a real pipeline, we'd save the model pickle too
    preds = pd.DataFrame({
        'y_true': result['y_val'],
        'y_pred_prob': result['y_pred_prob']
    })
    preds_path = str(Path(output_path).with_suffix('.csv'))
    preds.to_csv(preds_path, index=False)
    logger.info(f"Predictions saved to {preds_path}")

    # Save Model
    model_path = str(Path(output_path).with_suffix('.pkl'))
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IEEE-CIS Fraud Detection Experiment')
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'sota'],
                        help='Experiment mode: baseline (Academic) or sota (Proposed)')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='results/metrics.json',
                        help='Path to save output metrics')
    parser.add_argument('--quick', action='store_true',
                        help='Run fast with small sample')
    parser.add_argument('--smote', action='store_true',
                        help='Enable SMOTE for SOTA mode (Ablation)')
                        
    args = parser.parse_args()
    
    run_experiment(args.mode, args.data_path, args.output, args.quick, args.smote)
