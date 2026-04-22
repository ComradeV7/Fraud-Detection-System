import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import optuna
import os
import logging  # Added missing import
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("STARTING UNCHAINED XGBOOST PIPELINE")
    df = pd.read_csv("data/Base.csv", engine="pyarrow")
    target_col = 'fraud_bool'
    
    X, y = df.drop(columns=[target_col]), df[target_col]
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = X[col].astype('category')
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "scale_pos_weight": pos_weight,
            "enable_categorical": True,
            "tree_method": "hist",  # <--- THE FIX: Required for XGBoost 1.x categoricals
            "random_state": 42
        }
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) 
    
    final_params = study.best_params
    final_params.update({
        "scale_pos_weight": pos_weight, 
        "enable_categorical": True, 
        "tree_method": "hist",  # <--- THE FIX: Required here as well
        "random_state": 42
    })
    
    final_model = xgb.XGBClassifier(**final_params)
    with mlflow.start_run(run_name='xgboost_unchained'):
        final_model.fit(X_train, y_train)
        final_model.get_booster().save_model("artifacts/xgboost_unchained.json")

        preds_proba = final_model.predict_proba(X_test)[:, 1]
        metrics = {"auc": float(roc_auc_score(y_test, preds_proba))}
        mlflow.log_metrics(metrics)
        
        # SHAP Summary Plot
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test[:2000])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:2000], show=False)
        plt.savefig("artifacts/xgboost_shap.png")
        plt.close()

if __name__ == "__main__":
    main()