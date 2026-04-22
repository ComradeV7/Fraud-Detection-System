import os
import time
import torch
import joblib
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, 
    f1_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Suppress OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)
os.makedirs("artifacts/evaluation", exist_ok=True)

def load_anfis_results():
    """Loads ANFIS model and its specific test data."""
    model = torch.load("artifacts/anfis_model.pt", map_location="cpu")
    model.eval()
    X_test = pd.read_parquet("artifacts/X_test_processed.parquet").values
    y_test = pd.read_parquet("artifacts/y_test_processed.parquet").values.ravel()
    return model, X_test, y_test

def load_xgboost_results():
    """Loads XGBoost model and recreates its raw test split."""
    model = xgb.XGBClassifier()
    model.load_model("artifacts/xgboost_unchained.json")
    
    # Recreate the raw split used in the training script
    df = pd.read_csv("data/Base.csv", engine="pyarrow")
    X = df.drop(columns=['fraud_bool'])
    y = df['fraud_bool']
    
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].astype('category')
        
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return model, X_test, y_test

def benchmark_latency(model, data, model_type="torch"):
    """Measures average inference latency per transaction."""
    iterations = 1000
    sample = data[:iterations]
    
    if model_type == "torch":
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_tensor)
        end = time.perf_counter()
    else:
        start = time.perf_counter()
        _ = model.predict_proba(sample)
        end = time.perf_counter()
        
    return (end - start) / iterations * 1000  # Convert to milliseconds

def main():
    logger.info("Starting Gold Standard Model Comparison...")

    # 1. Load Systems
    logger.info("Loading models and test datasets...")
    anfis, X_test_anfis, y_test_anfis = load_anfis_results()
    xgboost, X_test_xgb, y_test_xgb = load_xgboost_results()

    # 2. Generate Predictions (PyTorch requires Batched Inference to prevent OOM)
    logger.info("Generating ANFIS predictions (Batched)...")
    batch_size = 2048
    anfis_probs_list = []
    
    with torch.no_grad():
        tensor_X = torch.tensor(X_test_anfis, dtype=torch.float32)
        for i in range(0, len(tensor_X), batch_size):
            batch = tensor_X[i:i + batch_size]
            probs = anfis(batch).numpy()
            anfis_probs_list.append(probs)
            
    # Concatenate batches and flatten to 1D array
    anfis_probs = np.concatenate(anfis_probs_list, axis=0).ravel()
    anfis_preds = (anfis_probs >= 0.5).astype(int)

    logger.info("Generating XGBoost predictions...")
    xgb_probs = xgboost.predict_proba(X_test_xgb)[:, 1]
    xgb_preds = xgboost.predict(X_test_xgb)

    # 3. Calculate Performance Metrics
    logger.info("Calculating metrics and generating charts...")
    results = []
    for name, targets, probs, preds in [
        ("PyTorch ANFIS", y_test_anfis, anfis_probs, anfis_preds),
        ("XGBoost (Raw)", y_test_xgb, xgb_probs, xgb_preds)
    ]:
        results.append({
            "Model": name,
            "AUC": roc_auc_score(targets, probs),
            "F1-Score": f1_score(targets, preds),
            "Precision": precision_score(targets, preds),
            "Recall": recall_score(targets, preds),
            "Latency (ms)": benchmark_latency(
                anfis if "ANFIS" in name else xgboost, 
                X_test_anfis if "ANFIS" in name else X_test_xgb,
                "torch" if "ANFIS" in name else "xgb"
            )
        })

    df_res = pd.DataFrame(results)
    print("\n--- PERFORMANCE SUMMARY TABLE ---")
    print(df_res.to_string(index=False))
    print("---------------------------------\n")
    df_res.to_csv("artifacts/evaluation/model_comparison.csv", index=False)

    # 4. Visual Comparison: ROC Curves
    plt.figure(figsize=(10, 7))
    for name, targets, probs in [
        ("PyTorch ANFIS", y_test_anfis, anfis_probs),
        ("XGBoost Challenger", y_test_xgb, xgb_probs)
    ]:
        fpr, tpr, _ = roc_curve(targets, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(targets, probs):.4f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Champion vs. Challenger: ROC Curve Comparison', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/evaluation/comparison_roc.png", dpi=300)
    plt.close()

    # 5. Visual Comparison: Metric Delta
    df_melted = df_res.melt(id_vars="Model", value_vars=["AUC", "F1-Score", "Precision", "Recall"])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="variable", y="value", hue="Model", palette="viridis")
    plt.title("Statistical Performance Divergence", fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.legend(title="Model Architecture")
    plt.tight_layout()
    plt.savefig("artifacts/evaluation/comparison_metrics.png", dpi=300)
    plt.close()

    logger.info("Evaluation complete. Charts saved in artifacts/evaluation/")

if __name__ == "__main__":
    main()