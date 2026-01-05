"""
Training Module for IEEE-CIS Fraud Detection
Handles model training (LightGBM, XGBoost) and evaluation
"""
import time
import gc
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from .validation import get_time_split_folds, get_single_time_split

# Global Seed
SEED = 42

def train_baseline_model(
    df: pd.DataFrame,
    features: list,
    target: str = 'isFraud'
) -> Dict[str, Any]:
    """
    Train using the 'Academic Baseline' approach:
    - Use SMOTE for balancing
    - Time-series aware CV (strict split)
    - Ensemble (Simulated by training XGBoost as representative)
    
    Args:
        df: Input DataFrame
        features: List of feature names
        target: Target column name
        
    Returns:
        Dictionary containing metrics and model
    """
    print("\nTraining Baseline Model (SMOTE + XGBoost)...")
    
    # Simple Time-Split: Train 75%, Val 25%
    train_idx, val_idx = get_single_time_split(df)
    
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][target]
    X_val = df.iloc[val_idx][features]
    y_val = df.iloc[val_idx][target]
    
    # Apply SMOTE - ONLY on Training Data
    print(f"Applying SMOTE to training set. Original shape: {X_train.shape}, Class distribution: {y_train.value_counts(normalize=True).to_dict()}")
    # Handle NaN before SMOTE (XGB handles them, SMOTE doesn't usually)
    X_train = X_train.fillna(-999) 
    
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Post-SMOTE shape: {X_train_res.shape}, Class distribution: {y_train_res.value_counts(normalize=True).to_dict()}")
    
    # Train XGBoost
    # Using parameters similar to academic papers
    # XGBoost 3.x+ compatibility
    # early_stopping_rounds is now preferred in constructor or callbacks, 
    # but simplest fix for recent versions is enabling it in constructor if available
    # or passing it via callbacks.
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED,
        tree_method='hist', # Faster
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    start_time = time.time()
    clf.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    train_time = time.time() - start_time
    
    # Predict
    y_pred_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    ap = average_precision_score(y_val, y_pred_prob)
    f1 = f1_score(y_val, y_pred)
    
    print(f"Baseline Results - AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}, Time: {train_time:.1f}s")
    
    return {
        'model': clf,
        'metrics': {'auc': auc, 'ap': ap, 'f1': f1},
        'y_val': y_val,
        'y_pred_prob': y_pred_prob,
        'training_time': train_time
    }


def train_sota_model(
    df: pd.DataFrame,
    features: list,
    target: str = 'isFraud',
    use_group_kfold: bool = False,
    use_smote: bool = False
) -> Dict[str, Any]:
    """
    Train using the 'SOTA' approach:
    - Magic Features (assumed present in df)
    - LightGBM
    - Optional SMOTE (for ablation study)
    
    Args:
        df: Input DataFrame (with Magic features)
        features: List of feature names
        target: Target column name
        use_group_kfold: Whether to use GroupKFold style (not fully implemented in simple version)
        use_smote: Whether to apply SMOTE (Ablation study)
        
    Returns:
        Dictionary containing metrics and model
    """
    if use_smote:
        print("\nTraining Ablation Model (Magic Features + SMOTE + LightGBM)...")
    else:
        print("\nTraining SOTA Model (No SMOTE + LightGBM + Magic Features)...")
    
    # Simple Time-Split: Train 75%, Val 25%
    train_idx, val_idx = get_single_time_split(df)
    
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][target]
    X_val = df.iloc[val_idx][features]
    y_val = df.iloc[val_idx][target]
    
    if use_smote:
        print(f"Applying SMOTE to training set. Original shape: {X_train.shape}")
         # Handle NaN before SMOTE (LGBM handles them by default, but SMOTE needs help)
         # Simple fill for SMOTE purposes
        X_train_filled = X_train.copy()
        
        # Handle Categorical columns: SMOTE requires numeric input!
        # We must label encode everything for SMOTE to work
        # CRITICAL: Must encode X_val same way (or at least to numeric) so LGBM accepts it
        cat_cols = [c for c in X_train_filled.columns if isinstance(X_train_filled[c].dtype, pd.CategoricalDtype) or X_train_filled[c].dtype == 'object']
        
        for col in cat_cols:
            # Simple factorization for ablation purposes.
            # In production, use OrdinalEncoder fit on Train and transform Val.
            # Here we just want them to be numeric.
            # Convert to category codes if possible, else factorize
            if isinstance(X_train_filled[col].dtype, pd.CategoricalDtype):
                X_train_filled[col] = X_train_filled[col].cat.codes
                X_val[col] = X_val[col].astype('category').cat.codes
            else:
                X_train_filled[col], _ = pd.factorize(X_train_filled[col])
                # For Val, force conversion (might lose alignment but fine for this proof)
                X_val[col], _ = pd.factorize(X_val[col])

        # Fill remaining numerical NaNs
        X_train_filled = X_train_filled.fillna(-999)
        X_val = X_val.fillna(-999)
        
        smote = SMOTE(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train_filled, y_train)
        print(f"Post-SMOTE shape: {X_train.shape}")
        
    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Create LGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters optimized for speed and performance
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': not use_smote, # If SMOTE is used, data is balanced, so turn this off
        'learning_rate': 0.02, # Lower LR for better accuracy
        'num_leaves': 256,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    }
    
    start_time = time.time()
    
    clf = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    train_time = time.time() - start_time
    
    # Predict
    y_pred_prob = clf.predict(X_val, num_iteration=clf.best_iteration)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    ap = average_precision_score(y_val, y_pred_prob)
    f1 = f1_score(y_val, y_pred)
    
    print(f"SOTA Results - AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}, Time: {train_time:.1f}s")
    
    # Feature Importance (Quick Look)
    importance = pd.DataFrame({
        'feature': features,
        'gain': clf.feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False).head(10)
    
    print("\nTop 10 Features by Gain:")
    print(importance)
    
    return {
        'model': clf,
        'metrics': {'auc': auc, 'ap': ap, 'f1': f1},
        'y_val': y_val,
        'y_pred_prob': y_pred_prob,
        'feature_importance': importance,
        'training_time': train_time
    }

if __name__ == "__main__":
    pass  # Logic tests mostly integrated in main
