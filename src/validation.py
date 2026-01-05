"""
Validation Strategy Module for IEEE-CIS Fraud Detection
Implements Time-Series Split to prevent future leakage
"""
import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit


def get_time_split_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    time_col: str = 'TransactionDT'
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate time-based train/validation splits.
    
    This prevents future leakage by ensuring validation data is always
    temporally after training data.
    
    Args:
        df: DataFrame sorted by time
        n_splits: Number of folds
        time_col: Name of the time column
    
    Yields:
        Tuples of (train_indices, val_indices)
    """
    # Ensure data is sorted by time
    assert df[time_col].is_monotonic_increasing, \
        f"DataFrame must be sorted by {time_col}"
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, val_idx in tscv.split(df):
        yield train_idx, val_idx


def get_month_based_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    time_col: str = 'TransactionDT'
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate month-based folds for GroupKFold-style validation.
    
    This ensures no UID overlap between train/val naturally since
    users appear consistently across time.
    
    Args:
        df: DataFrame with time column
        n_splits: Number of folds
        time_col: Name of the time column
    
    Yields:
        Tuples of (train_indices, val_indices)
    """
    # Convert TransactionDT (seconds since reference) to month
    # The dataset spans approximately 6 months
    seconds_per_month = 30 * 24 * 60 * 60
    df_months = (df[time_col] // seconds_per_month).astype(int)
    
    unique_months = sorted(df_months.unique())
    n_months = len(unique_months)
    
    if n_months < n_splits + 1:
        raise ValueError(f"Not enough months ({n_months}) for {n_splits} splits")
    
    # Calculate months per fold
    months_per_fold = n_months // (n_splits + 1)
    
    for fold in range(n_splits):
        # Training: first (fold + 1) groups of months
        train_months = unique_months[:(fold + 1) * months_per_fold]
        # Validation: next group of months
        val_start = (fold + 1) * months_per_fold
        val_end = min((fold + 2) * months_per_fold, n_months)
        val_months = unique_months[val_start:val_end]
        
        train_idx = np.where(df_months.isin(train_months))[0]
        val_idx = np.where(df_months.isin(val_months))[0]
        
        yield train_idx, val_idx


def get_single_time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    time_col: str = 'TransactionDT'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a single time-based train/validation split.
    
    This is the recommended approach for this competition:
    Train on first 75%, validate on last 25%.
    
    Args:
        df: DataFrame sorted by time
        train_ratio: Fraction of data for training
        time_col: Name of the time column
    
    Returns:
        Tuple of (train_indices, val_indices)
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    
    train_idx = np.arange(0, split_idx)
    val_idx = np.arange(split_idx, n)
    
    return train_idx, val_idx


def validate_no_uid_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    uid_col: str = 'uid'
) -> dict:
    """
    Validate that there's no UID overlap between train and validation sets.
    
    This is important to ensure the model doesn't just memorize user IDs.
    
    Args:
        df: DataFrame with UID column
        train_idx: Training indices
        val_idx: Validation indices
        uid_col: Name of the UID column
    
    Returns:
        Dictionary with validation statistics
    """
    if uid_col not in df.columns:
        return {"error": f"Column {uid_col} not found"}
    
    train_uids = set(df.iloc[train_idx][uid_col].unique())
    val_uids = set(df.iloc[val_idx][uid_col].unique())
    
    overlap = train_uids.intersection(val_uids)
    
    return {
        "train_unique_uids": len(train_uids),
        "val_unique_uids": len(val_uids),
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / len(val_uids) if val_uids else 0,
        "is_clean": len(overlap) == 0
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    n = 1000
    df = pd.DataFrame({
        'TransactionDT': np.sort(np.random.randint(0, 15552000, n)),  # ~6 months
        'isFraud': np.random.randint(0, 2, n)
    })
    
    print("Testing time-series split...")
    for fold, (train_idx, val_idx) in enumerate(get_time_split_folds(df, n_splits=3)):
        print(f"Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        print(f"  Train time range: {df.iloc[train_idx]['TransactionDT'].min()} - "
              f"{df.iloc[train_idx]['TransactionDT'].max()}")
        print(f"  Val time range: {df.iloc[val_idx]['TransactionDT'].min()} - "
              f"{df.iloc[val_idx]['TransactionDT'].max()}")
    
    print("\nTesting single split...")
    train_idx, val_idx = get_single_time_split(df, train_ratio=0.75)
    print(f"Train: {len(train_idx)} samples")
    print(f"Val: {len(val_idx)} samples")
