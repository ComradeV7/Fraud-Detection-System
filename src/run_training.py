import os
# This must be the very first line before importing torch or numpy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
import mlflow
import matplotlib.pyplot as plt

# Import custom modules
from src.model.anfis import ANFIS, initialise_centres
from src.model.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STARTING OPTIMIZED ANFIS PIPELINE ===")
    
    # 1. Load Processed Data
    logger.info("Step 1: Loading Processed Data...")
    X_train_df = pd.read_parquet("artifacts/X_train_processed.parquet")
    X_test_df = pd.read_parquet("artifacts/X_test_processed.parquet")
    y_train_df = pd.read_parquet("artifacts/y_train_processed.parquet")
    y_test_df = pd.read_parquet("artifacts/y_test_processed.parquet")
    
    X_train_all = X_train_df.values
    X_test_all = X_test_df.values
    y_train_all = y_train_df.values.ravel()
    y_test_all = y_test_df.values.ravel()

    # Calculate Class Weight (Majority / Minority)
    pos_weight = float((len(y_train_all) - y_train_all.sum()) / y_train_all.sum())
    logger.info(f"Calculated Fraud Class Weight (pos_weight): {pos_weight:.2f}")

    # 2. Create Subsamples for Optuna Tuning
    SAMPLE_SIZE = 100000 
    indices = np.random.choice(len(X_train_all), SAMPLE_SIZE, replace=False)
    X_train_sub = X_train_all[indices]
    y_train_sub = y_train_all[indices]
    
    # 3. Setup Model Parameters
    rule_path = Path("artifacts/fuzzy_rules.json")
    with open(rule_path, 'r') as f:
        rules = json.load(f)
    
    n_rules = len(rules)
    n_inputs = X_train_all.shape[1]
    centres = np.array([list(r["premise"].values()) for r in rules])

    # 4. Bayesian Optimization
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        model = ANFIS(n_inputs=n_inputs, n_rules=n_rules)
        initialise_centres(model, centres)
        
        # Pass pos_weight to the training function
        _, best_auc, _ = train(
            model=model,
            X_train=X_train_sub, 
            y_train=y_train_sub,
            X_test=X_test_all[:20000], 
            y_test=y_test_all[:20000],
            epochs=10, 
            lr=lr,
            pos_weight=pos_weight
        )
        return best_auc

    logger.info("Step 2: Finding Optimal Learning Rate (Optuna)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    
    # 5. Final Production Training
    logger.info(f"Step 3: Training Final Model on FULL {len(X_train_all):,} rows...")
    final_lr = study.best_params["lr"]
    
    final_model = ANFIS(n_inputs=n_inputs, n_rules=n_rules)
    initialise_centres(final_model, centres)
    
    # Pass pos_weight to the final production training
    final_run_id, final_auc, history = train(
        model=final_model,
        X_train=X_train_all, 
        y_train=y_train_all,
        X_test=X_test_all,
        y_test=y_test_all,
        epochs=30, 
        lr=final_lr,
        pos_weight=pos_weight
    )
    
    # 6. Save Final Artifacts and Diagnostic Plots
    os.makedirs("artifacts", exist_ok=True)
    torch.save(final_model, "artifacts/anfis_model.pt")
    logger.info(f"SAVED: artifacts/anfis_model.pt | Final AUC: {final_auc:.4f}")

    # --- Plotting Convergence Curves ---
    # RESTORED: This line is required to prevent plot crashing
    epochs_range = range(1, len(history['loss']) + 1)
      
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs_range, history['loss'], color='tab:red', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot AUC on secondary axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation AUC', color='tab:blue')
    ax2.plot(epochs_range, history['auc'], color='tab:blue', label='Val AUC')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('ANFIS Convergence: Loss and AUC Evolution')
    fig.tight_layout()
    plt.savefig("artifacts/training_convergence.png")
    plt.close()
    logger.info("SAVED: artifacts/training_convergence.png")

if __name__ == "__main__":
    main()