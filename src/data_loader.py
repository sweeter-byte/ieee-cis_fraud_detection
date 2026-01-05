"""
Data Loading Module for IEEE-CIS Fraud Detection
Handles memory optimization and data merging
"""
import gc
import pandas as pd
import numpy as np
from pathlib import Path


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        verbose: Whether to print memory reduction info
    
    Returns:
        DataFrame with optimized memory usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # float16 can cause issues
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Convert object columns to category if unique values are limited
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def load_data(data_path: str, sample_size: int = None, is_train: bool = True) -> pd.DataFrame:
    """
    Load and merge transaction and identity data.
    
    Args:
        data_path: Path to the data directory containing CSV files
        sample_size: Optional number of rows to sample (for testing)
        is_train: Whether to load training or test data
    
    Returns:
        Merged DataFrame sorted by TransactionDT
    """
    data_path = Path(data_path)
    
    prefix = 'train' if is_train else 'test'
    
    print(f"Loading {prefix}_transaction.csv...")
    transaction = pd.read_csv(
        data_path / f'{prefix}_transaction.csv',
        nrows=sample_size
    )
    
    print(f"Loading {prefix}_identity.csv...")
    identity = pd.read_csv(data_path / f'{prefix}_identity.csv')
    
    print("Reducing memory usage for transaction data...")
    transaction = reduce_mem_usage(transaction)
    
    print("Reducing memory usage for identity data...")
    identity = reduce_mem_usage(identity)
    
    print("Merging transaction and identity data...")
    df = transaction.merge(identity, on='TransactionID', how='left')
    
    # Clean up
    del transaction, identity
    gc.collect()
    
    # Sort by time - CRITICAL for time-series validation
    print("Sorting by TransactionDT...")
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    print(f"Final dataset shape: {df.shape}")
    
    return df


def load_test_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load test data for submission.
    
    Args:
        data_path: Path to the data directory
        sample_size: Optional number of rows to sample
    
    Returns:
        Merged test DataFrame
    """
    return load_data(data_path, sample_size=sample_size, is_train=False)


if __name__ == "__main__":
    # Quick test
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    df = load_data(data_path, sample_size=1000)
    print(f"\nLoaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns[:10])}...")
