import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(filepath: str) -> pd.DataFrame:
    """Reads the BAF CSV file and validates the presence of core fraud features.

    Args:
        filepath: The local path to the NeurIPS BAF dataset CSV.

    Returns:
        A pandas DataFrame containing the raw bank account data.

    Raises:
        ValueError: If 'fraud_bool', 'income', or 'customer_age' are missing.
    """
    # Load the raw dataset
    df = pd.read_csv(filepath)
    
    # Check for core columns identified in bivariate analysis
    required_cols = {'fraud_bool', 'income', 'customer_age'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required BAF columns: {missing}")
        
    return df

def explore(df: pd.DataFrame) -> dict:
    """Generates a summary of BAF dataset statistics and feature availability.

    Args:
        df: The loaded BAF DataFrame.

    Returns:
        A dictionary containing row counts, fraud distribution, and feature list.
    """
    # Extract fraud label distribution
    fraud_count = int(df['fraud_bool'].sum())
    total_rows = len(df)
    
    # Construct exploration dictionary
    stats = {
        "total_rows": total_rows,
        "fraud_count": fraud_count,
        "clean_count": total_rows - fraud_count,
        "fraud_pct": (fraud_count / total_rows) * 100,
        "null_counts": df.isnull().sum().to_dict(),
        "feature_names": df.columns.tolist()
    }
    
    return stats

def split(df: pd.DataFrame, test_size=0.2, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Isolates features from 'fraud_bool' and performs a stratified split.

    Args:
        df: The raw BAF DataFrame.
        test_size: Proportion of rows for the test set (default 20%).
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    # Separate features from target
    X = df.drop(columns=['fraud_bool'])
    y = df['fraud_bool']
    
    # Stratified split ensures the 1.1% fraud ratio is preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test