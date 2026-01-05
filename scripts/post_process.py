"""
Post-processing script to boost AUC.
Technique: UID Consistency
Logic: If a user (UID) is fraudulent, most of their transactions are likely fraud.
       Averaging predictions per UID smoothens variance and inputs 'future' information 
       (if using future test data) or just reinforces the signal.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from src.data_loader import load_data
from src.feature_engineering import engineer_features

def apply_post_process(pred_csv: str, data_path: str):
    print(f"Loading predictions from {pred_csv}...")
    preds = pd.read_csv(pred_csv)
    
    # We need the UIDs to group by. 
    # Since predictions csv doesn't have UIDs, we need to regenerate them from raw data.
    # This assumes the preds correspond exactly to the validation set of the last run.
    # To do this correctly without reloading everything, we should have saved UIDs with preds.
    
    # FOR DEMONSTRATION: We will reload the data and regenerate UIDs.
    # NOTE: This only works if we know WHICH rows were validation. 
    # The 'get_single_time_split' uses the last 25% by default.
    
    print("Regenerating UIDs for validation set (Last 25%)...")
    df = load_data(data_path, is_train=True)
    
    # Get Validation indices
    n = len(df)
    train_ratio = 0.75
    split_idx = int(n * train_ratio)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # Ensure lengths match
    if len(val_df) != len(preds):
        print(f"Error: Length mismatch. Val dataset: {len(val_df)}, Preds: {len(preds)}")
        # Check if maybe quick run was used? 
        # Actually validation.py uses simple split.
        # If lengths differ, we can't easily map.
        return

    # Generate UIDs
    # Re-use the Exact logic from feature_engineering
    val_df['day'] = (val_df['TransactionDT'] // (24*60*60))
    val_df['D1n'] = val_df['day'] - val_df['D1']
    c1 = val_df['card1'].astype(str)
    a1 = val_df['addr1'].astype(str).fillna('nan')
    d1n = val_df['D1n'].apply(lambda x: f"{x:.0f}").replace('nan', 'nan')
    email = val_df['P_emaildomain'].astype(str).fillna('nan')
    val_df['uid'] = c1 + '_' + a1 + '_' + d1n + '_' + email
    
    # Attach predictions
    val_df['y_pred_prob'] = preds['y_pred_prob']
    val_df['y_true'] = preds['y_true'] # Or use val_df['isFraud']
    
    print("Applying UID prediction averaging...")
    val_df['y_pred_avg'] = val_df.groupby('uid')['y_pred_prob'].transform('mean')
    
    # Calculate Scores
    original_auc = roc_auc_score(val_df['y_true'], val_df['y_pred_prob'])
    new_auc = roc_auc_score(val_df['y_true'], val_df['y_pred_avg'])
    
    print(f"Original AUC: {original_auc:.4f}")
    print(f"Post-Processed AUC: {new_auc:.4f}")
    print(f"Lift: {new_auc - original_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default='../results/sota_metrics_improved.csv')
    parser.add_argument('--data', default='dataset')
    
    args = parser.parse_args()
    apply_post_process(args.preds, args.data)
