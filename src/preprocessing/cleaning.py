import pandas as pd
import numpy as np

def handle_sentinels(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identifies -1 placeholders, creates missingness flags, and imputes medians.

    Args:
        df_train: The training feature matrix.
        df_test: The testing feature matrix.

    Returns:
        A tuple of processed (train, test) DataFrames.
    """
    # Define columns with hidden -1 values identified during EDA
    sentinel_cols = [
        'prev_address_months_count', 'bank_months_count', 
        'current_address_months_count', 'session_length_in_minutes', 
        'credit_risk_score', 'device_distinct_emails_8w'
    ]

    for col in sentinel_cols:
        if col in df_train.columns:
            # Create binary flags to capture the behavioral signal of missing data
            df_train[f'{col}_is_missing'] = (df_train[col] == -1).astype(int)
            df_test[f'{col}_is_missing'] = (df_test[col] == -1).astype(int)
            
            # Replace sentinels with NaN for proper median calculation
            df_train[col] = df_train[col].replace(-1, np.nan)
            df_test[col] = df_test[col].replace(-1, np.nan)
            
            # Calculate median strictly from training set to prevent leakage
            train_median = df_train[col].median()
            
            # Impute missing values
            df_train[col] = df_train[col].fillna(train_median)
            df_test[col] = df_test[col].fillna(train_median)
            
    return df_train, df_test

def apply_skewness_fix(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies Signed Pseudo-Log transformation to stabilize heavy-tailed features.

    Args:
        df_train: The training feature matrix.
        df_test: The testing feature matrix.

    Returns:
        A tuple of transformed (train, test) DataFrames.
    """
    # Columns identified by the Automated Skewness Scanner as having |skew| > 1.0
    highly_skewed_cols = [
        'days_since_request', 'prev_address_months_count', 
        'session_length_in_minutes', 'bank_branch_count_8w', 
        'intended_balcon_amount', 'device_distinct_emails_8w', 
        'zip_count_4w', 'current_address_months_count', 
        'proposed_credit_limit'
    ]

    for col in highly_skewed_cols:
        if col in df_train.columns:
            # Apply Signed Pseudo-Log to handle right-skew and negative overdrafts safely
            # y = sign(x) * log(1 + |x|)
            df_train[f'{col}_log'] = np.sign(df_train[col]) * np.log1p(np.abs(df_train[col]))
            df_test[f'{col}_log'] = np.sign(df_test[col]) * np.log1p(np.abs(df_test[col]))
            
            # Drop raw columns to eliminate multicollinearity
            df_train = df_train.drop(columns=[col])
            df_test = df_test.drop(columns=[col])
            
    return df_train, df_test