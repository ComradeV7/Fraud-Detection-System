import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    sampling_strategy: float = 0.3,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Applies SMOTE to synthetically balance the minority fraud class.

    Args:
        X_train: The PCA-compressed training feature matrix.
        y_train: The training target labels.
        sampling_strategy: The desired ratio of the minority class to the 
            majority class after resampling. Defaults to 0.3 (30%).
        random_state: Seed for reproducible synthetic generation.

    Returns:
        A tuple containing the resampled (X_train, y_train) arrays.
    """
    # Print the class distribution before SMOTE
    print(f"Before SMOTE: {Counter(y_train)}")
    
    # Initialize and apply the SMOTE algorithm strictly to the training data
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Print the class distribution after SMOTE to verify the new ratio
    print(f"After SMOTE:  {Counter(y_resampled)}")
    
    return X_resampled, y_resampled

def validate_balance(y_original: np.ndarray, y_resampled: np.ndarray) -> dict:
    """Validates that the SMOTE transformation mathematically succeeded.

    Args:
        y_original: The target array before balancing.
        y_resampled: The target array after balancing.

    Returns:
        A dictionary containing the original and resampled metrics.

    Raises:
        ValueError: If the resampled fraud percentage is lower than the original.
    """
    # Calculate absolute row counts
    original_total = len(y_original)
    resampled_total = len(y_resampled)
    
    # Calculate the percentage of the minority class (fraud = 1)
    original_fraud_pct = (np.sum(y_original == 1) / original_total) * 100
    resampled_fraud_pct = (np.sum(y_resampled == 1) / resampled_total) * 100
    
    # Defensive check: Ensure SMOTE actually increased the minority presence
    if resampled_fraud_pct < original_fraud_pct:
        raise ValueError(
            f"SMOTE validation failed: Resampled fraud pct ({resampled_fraud_pct:.2f}%) "
            f"is lower than original ({original_fraud_pct:.2f}%)."
        )
        
    # Construct the validation report
    metrics = {
        "original_fraud_pct": original_fraud_pct,
        "resampled_fraud_pct": resampled_fraud_pct,
        "original_total": original_total,
        "resampled_total": resampled_total
    }
    
    return metrics