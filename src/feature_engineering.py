"""
Feature Engineering Module for IEEE-CIS Fraud Detection
Implements Client Identification (UID) and Magic Features
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def preprocessing_preprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implement the preprocessing steps from the academic baseline papers.
    
    Steps:
    1. Filter missing values (>95%)
    2. Impute with median (numerical) or 'missing' (categorical)
    3. Label Encoding for categoricals
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with baseline features preprocessed
    """
    print("Running baseline preprocessing (Academic Approach)...")
    
    # 1. Drop columns with >95% missing values
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.95].index
    print(f"Dropping {len(cols_to_drop)} columns with >95% missing values")
    df = df.drop(columns=cols_to_drop)
    
    # Identify column types
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype) == 'category']
    num_cols = [c for c in df.columns if c not in cat_cols and c not in ['TransactionID', 'isFraud', 'TransactionDT']]
    
    # 2. Imputation
    # Note: In a real production scenario, imputers should be fit on train and transform test
    # Here we do simple imputation for the baseline reproduction
    print("Imputing numerical values with median...")
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    print("Imputing categorical values...")
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].astype(str).fillna('missing')
            
    # 3. Label Encoding
    print("Label encoding categorical features...")
    for col in cat_cols:
        # Simple factorize/LabelEncode
        df[col], _ = pd.factorize(df[col])
        
    return df


def make_uid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implement the 'Magic' features based on Client Identification.
    
    Strategy:
    1. Create UID = card1 + addr1 + D1
    2. Group Aggregations based on UID
    3. Frequency Encoding
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new magic features
    """
    print("Generating UID features (SOTA Approach)...")
    
    # Ensure required columns exist
    required_cols = ['card1', 'addr1', 'D1', 'TransactionAmt', 'TransactionDT']
    for col in required_cols:
        if col not in df.columns:
            # Handle missing columns if partial data loaded (e.g. for testing)
            if col == 'D1' and 'D2' in df.columns:
                print(f"Column {col} missing, using D2 as proxy")
                df[col] = df['D2']
            elif col == 'addr1':
                print(f"Column {col} missing, using default")
                df[col] = -1
            elif col == 'D1':
                print(f"Column {col} missing, using 0")
                df[col] = 0
    
    # 1. Create UID
    # 'card1' is coarse. 'card1'+'addr1' is better. 
    # 'card1'+'addr1'+'D1' is very strong as D1 is "days since client began" (roughly)
    
    # CRITICAL FIX for SOTA Performance:
    # D1 increases as time moves forward. We need a "Time Invariant" D1.
    # D1n = Day - D1 (approximate Start Day of the card)
    print("Creating Time-Invariant UID...")
    
    # Ensure 'day' column exists (it is created in engineer_features but we might run this independently)
    if 'day' not in df.columns:
        df['day'] = (df['TransactionDT'] // (24*60*60))
        
    # Calculate D1n = Day - D1
    # We clip to positive to handle potential noise
    df['D1n'] = df['day'] - df['D1']
    
    c1 = df['card1'].astype(str)
    a1 = df['addr1'].astype(str).fillna('nan')
    # Use the invariant D1n instead of raw D1
    d1n = df['D1n'].apply(lambda x: f"{x:.0f}").replace('nan', 'nan')
    
    # Adding P_emaildomain significantly boosts UID uniqueness
    # Format: card1_addr1_D1n_email
    email = df['P_emaildomain'].astype(str).fillna('nan')
    
    df['uid'] = c1 + '_' + a1 + '_' + d1n + '_' + email
    
    # 2. Group Aggregations
    # Calculate stats for the user
    # Note: Using transform allows us to keep the original shape
    print("Calculating Group Aggregations...")
    
    cols_to_agg = ['TransactionAmt']
    # Add other columns if they exist. C features are very important for aggregation
    potential_cols = ['id_02', 'D15', 'C13', 'C1', 'C14', 'dist1']
    for c in potential_cols:
        if c in df.columns:
            cols_to_agg.append(c)
            
    for col in cols_to_agg:
        # Mean and Std of amount/feature for this user
        df[f'{col}_mean_by_uid'] = df.groupby('uid')[col].transform('mean')
        df[f'{col}_std_by_uid'] = df.groupby('uid')[col].transform('std')
        
    # 3. Frequency Encoding (Count Encoding)
    # "How many times has this user appeared?"
    print("Calculating Frequency Encodings...")
    # Global count (Train+Test would be ideal, here we use available data)
    df['uid_count'] = df.groupby('uid')['TransactionDT'].transform('count')
    
    # 4. Normalize Aggregations
    # e.g. How big is this transaction compared to user's average?
    df['TransactionAmt_div_mean_uid'] = df['TransactionAmt'] / (df['TransactionAmt_mean_by_uid'] + 1e-5)
    
    # 5. UID consistency check feature (Simulated)
    # Count unique IP addresses (id_30/id_31/DeviceType) per UID if available
    if 'DeviceType' in df.columns:
        df['uid_unique_device_types'] = df.groupby('uid')['DeviceType'].transform('nunique')
        
    # Convert UID to numeric for model consumption if needed (Label Encode)
    # Or drop it if we only want the aggregates
    df['uid_encoded'], _ = pd.factorize(df['uid'])
    
    # Drop raw string UID to save memory
    df = df.drop(columns=['uid'])
    
    return df


def engineer_features(df: pd.DataFrame, use_magic: bool = True) -> pd.DataFrame:
    """
    Main entry point for feature engineering.
    
    Args:
        df: Input DataFrame
        use_magic: Whether to use User Identification features (SOTA) or Baseline
        
    Returns:
        Engineered DataFrame
    """
    # Common Time Features
    print("Generating common time features...")
    # TransactionDT is seconds. 
    # Day count
    df['day'] = (df['TransactionDT'] // (24*60*60))
    # Hour of day
    df['hour'] = (df['TransactionDT'] // (60*60)) % 24
    
    if use_magic:
        print("--- Mode: SOTA (Magic Features) ---")
        # Apply Magic Features
        df = make_uid_features(df)
        
        # Simple processing for categorical columns for LightGBM
        # LightGBM handles categories directly if type is 'category'
        cat_cols = [c for c in df.columns if df[c].dtype == 'object']
        for col in cat_cols:
            df[col] = df[col].astype('category')
            
    else:
        print("--- Mode: Baseline ---")
        # Apply Baseline paper preprocessing
        df = preprocessing_preprint(df)
        
    return df


if __name__ == "__main__":
    # Test with synthetic data
    df_test = pd.DataFrame({
        'TransactionID': range(10),
        'TransactionDT': range(0, 10000, 1000),
        'TransactionAmt': np.random.rand(10) * 100,
        'card1': [100, 100, 100, 200, 200, 300, 100, 100, 400, 200],
        'addr1': [10, 10, 10, 20, 20, 30, 10, 10, 40, 20],
        'D1': [0, 0, 0, 5, 5, 0, 0, 0, 10, 5],
        'isFraud': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        'dist1': np.random.rand(10) * 50
    })
    
    print("Original shape:", df_test.shape)
    
    # Test Magic
    df_magic = engineer_features(df_test.copy(), use_magic=True)
    print("\nMagic Features created:", [c for c in df_magic.columns if '_uid' in c])
    print("Magic shape:", df_magic.shape)
    print(df_magic[['card1', 'addr1', 'D1', 'uid_count', 'TransactionAmt_mean_by_uid']])
    
    # Test Baseline
    # Add some nulls to test imputation
    df_test.loc[0, 'dist1'] = np.nan
    df_baseline = engineer_features(df_test.copy(), use_magic=False)
    print("\nBaseline shape:", df_baseline.shape)
    print("Missing values after baseline:", df_baseline.isnull().sum().sum())
