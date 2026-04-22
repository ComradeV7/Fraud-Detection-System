import pandas as pd
from sklearn.preprocessing import TargetEncoder

def add_domain_ratios(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injects custom financial risk ratios derived from log-transformed features.

    Args:
        df_train: The training feature matrix (post-cleaning).
        df_test: The testing feature matrix (post-cleaning).

    Returns:
        A tuple of (train, test) DataFrames with new ratio features.
    """
    def _calculate(df: pd.DataFrame) -> pd.DataFrame:
        # 1. Credit Utilization Proxy: Captures relationship between requested balance and limits
        if 'intended_balcon_amount_log' in df.columns and 'proposed_credit_limit_log' in df.columns:
            # Adding 0.001 prevents division by zero in the PyTorch training loop
            df['credit_utilization_proxy'] = (
                df['intended_balcon_amount_log'] / (df['proposed_credit_limit_log'] + 0.001)
            )
        
        # 2. Suspicious Identity Velocity: High email usage vs low name/email similarity
        if 'device_distinct_emails_8w_log' in df.columns and 'name_email_similarity' in df.columns:
            df['suspicious_identity_velocity'] = (
                (df['device_distinct_emails_8w_log'] + 1) / (df['name_email_similarity'] + 0.001)
            )
            # Drop the raw log feature as its signal is now mathematically absorbed into the ratio
            df = df.drop(columns=['device_distinct_emails_8w_log'])
            
        return df

    return _calculate(df_train), _calculate(df_test)

def apply_target_encoding(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converts categorical strings into dense numerical fraud probabilities.

    Args:
        X_train: Training feature DataFrame.
        X_test: Testing feature DataFrame.
        y_train: Training labels used to fit the encoder.

    Returns:
        A tuple of (train, test) DataFrames with encoded categoricals.
    """
    # Identify all remaining text/category columns (e.g., housing_status, payment_type)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        # 'smooth=auto' prevents overfitting on categories with very few samples
        encoder = TargetEncoder(smooth='auto')
        
        # Fit strictly on training labels to maintain the Leakage Barrier
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])
        
    return X_train, X_test

def save_preprocessors(
    encoder, 
    scaler, 
    selector, 
    output_dir: str = "artifacts"
) -> None:
    """Saves fitted preprocessing objects to disk for production inference.

    Args:
        encoder: The fitted TargetEncoder object.
        scaler: The fitted StandardScaler object.
        pca: The fitted PCA object.
        output_dir: The directory to save the .pkl files.
    """
    # Create the artifacts directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Serialize and save the objects
        joblib.dump(encoder, os.path.join(output_dir, "target_encoder.pkl"))
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(selector, os.path.join(output_dir, "selector.pkl"))
        
        print(f"Preprocessors successfully saved to {output_dir}/")
    except Exception as e:
        print(f"Failed to save preprocessors: {str(e)}")
        raise