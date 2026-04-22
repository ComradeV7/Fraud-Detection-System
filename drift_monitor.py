import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler

def generate_drift_report():
    print("=== INITIALIZING CUSTOM K-S DRIFT MONITOR ===")
    
    # 1. Load Baseline Data
    print("Step 1: Loading Baseline Training Distribution...")
    try:
        baseline_data = pd.read_parquet("artifacts/X_train_processed.parquet")
    except FileNotFoundError:
        print("ERROR: Baseline data not found.")
        return

    # 2. Extract Live Logs from Production Database
    print("Step 2: Extracting Live Logs from Database...")
    if not os.path.exists("fraud_logs.db"):
        print("ERROR: Database 'fraud_logs.db' not found.")
        return

    conn = sqlite3.connect("fraud_logs.db")
    live_logs = pd.read_sql_query("""
        SELECT amount, fraud_score 
        FROM prediction_log 
        ORDER BY timestamp DESC 
        LIMIT 500
    """, conn)
    conn.close()

    if len(live_logs) < 10:
        print("WARNING: Insufficient live logs.")
        return

    print("Step 3: Scaling Features and Executing Statistical Drift Tests...")
    
    # Extract the raw arrays
    baseline_raw = baseline_data['income'].sample(n=len(live_logs), replace=True, random_state=42).values
    live_raw = live_logs['amount'].values

    # THE FIX: Standardize both arrays so they share the exact same scale for the visualization
    scaler = StandardScaler()
    baseline_scaled = scaler.fit_transform(baseline_raw.reshape(-1, 1)).flatten()
    live_scaled = scaler.fit_transform(live_raw.reshape(-1, 1)).flatten()

    # Run the K-S Test on the scaled shapes
    ks_stat, p_value = ks_2samp(baseline_scaled, live_scaled)
    
    drift_detected = "YES (Distribution Shifted)" if p_value < 0.05 else "NO (Stable)"
    status_color = "red" if p_value < 0.05 else "green"

    # 4. Generate the Professional MLOps Dashboard Image
    print("Step 4: Generating Visual Drift Report...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot overlapping distributions using the SCALED data
    sns.kdeplot(baseline_scaled, fill=True, color="blue", alpha=0.4, label="Baseline (Historical Data)")
    sns.kdeplot(live_scaled, fill=True, color=status_color, alpha=0.4, label=f"Live Production ({drift_detected})")
    
    plt.title("Concept Drift Monitor: Financial Transaction Variance", fontsize=14, fontweight='bold', pad=15)
    plt.suptitle(f"K-S Statistic: {ks_stat:.4f} | P-Value: {p_value:.4e} | Drift Detected: {drift_detected}", 
                 fontsize=11, color='gray', y=0.92)
    
    plt.xlabel("Standardized Feature Variance (Z-Score)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 5. Save the Output
    os.makedirs("artifacts/evaluation", exist_ok=True)
    save_path = "artifacts/evaluation/custom_drift_report.png"
    plt.savefig(save_path, dpi=300)
    
    print(f"\nSUCCESS! High-resolution report saved to: '{save_path}'")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    generate_drift_report()