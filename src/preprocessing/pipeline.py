import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# Import custom modularized logic
from src.preprocessing import data_loader, cleaning, engineering

os.makedirs("artifacts", exist_ok=True)

def run_pipeline(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Executes the end-to-end BAF data processing lifecycle.
    """

    # Ingestion and Leakage Barrier
    print("\nLoading data and establishing Leakage Barrier...")
    df_raw = data_loader.load_raw(filepath)
    X_train, X_test, y_train, y_test = data_loader.split(df_raw)
    print(f"Stratified Split Complete. Training rows: {len(X_train):,}")

    # Surgical Cleaning
    print("\nExecuting Surgical Cleaning...")
    X_train, X_test = cleaning.handle_sentinels(X_train, X_test)
    X_train, X_test = cleaning.apply_skewness_fix(X_train, X_test)
    print("Sentinels imputed and Signed Pseudo-Log applied.")

    # Domain Engineering and Encoding
    print("\nInjecting Domain Knowledge and Scaling Matrix...")
    X_train, X_test = engineering.add_domain_ratios(X_train, X_test)

    from category_encoders import TargetEncoder
    encoder = TargetEncoder(cols=['payment_type', 'employment_status', 'housing_status', 'source', 'device_os'])
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    joblib.dump(encoder, "artifacts/target_encoder.pkl")
    
    # Standardize variance
    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_test_scaled_arr = scaler.transform(X_test)
    
    joblib.dump(scaler, "artifacts/scaler.pkl")
    
    # Maintain feature traceability
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=X_test.columns, index=X_test.index)
    print(f"Matrix scaled. Dense Feature Count: {X_train_scaled.shape[1]}")

    # Feature Selection
    print("\nSelecting Top 15 Raw Features for Native Explainability...")
    selector = SelectKBest(score_func=f_classif, k=15)
    
    X_train_sel_arr = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel_arr = selector.transform(X_test_scaled)
    joblib.dump(selector, "artifacts/selector.pkl")
    
    winning_features = selector.get_feature_names_out(X_train.columns)
    X_train_sel = pd.DataFrame(X_train_sel_arr, columns=winning_features, index=X_train.index)
    X_test_sel = pd.DataFrame(X_test_sel_arr, columns=winning_features, index=X_test.index)
    print(f"Feature Selection complete. Top features retained: {list(winning_features)}")

    # Visualizing Feature Importance (ANOVA Scores)
    print("Generating Feature Importance chart...")
    scores = pd.Series(selector.scores_, index=X_train.columns)
    scores = scores.sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    scores.plot(kind='barh', color='skyblue')
    plt.title('Top 15 Features by ANOVA F-Value')
    plt.xlabel('F-Value')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("artifacts/feature_importance.png")
    plt.close()

    # Visualizing Post-Selection Correlation
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train_sel.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Post-Selection Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("artifacts/correlation_heatmap.png")
    plt.close()

    # Class Balancing (SMOTE)
    print("\nBalancing Minority Class (Training Data ONLY)...")
    smote = SMOTE(sampling_strategy=0.25, random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_sel, y_train)
    
    X_test_final = X_test_sel
    y_test_final = y_test
    print(f"SMOTE complete. New training rows: {len(X_train_final):,}")

    print("\nPIPELINE EXECUTION SUCCESSFUL.")

    X_train_final.to_parquet("artifacts/X_train_processed.parquet")
    y_train_final.to_frame().to_parquet("artifacts/y_train_processed.parquet")
    X_test_final.to_parquet("artifacts/X_test_processed.parquet")
    y_test_final.to_frame().to_parquet("artifacts/y_test_processed.parquet")

    return X_train_final, X_test_final, y_train_final, y_test_final

if __name__ == "__main__":
    run_pipeline("data/Base.csv")