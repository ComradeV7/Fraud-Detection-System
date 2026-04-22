import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_transformation_plot(df, feature='income'):
    """Generates a high-inference comparison of raw vs. transformed data."""
    # 1. Apply Signed Pseudo-Log Transformation
    # Formula: sign(x) * log1p(abs(x))
    transformed_feature = np.sign(df[feature]) * np.log1p(np.abs(df[feature]))
    
    # 2. Setup Plotting Environment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_style("whitegrid")
    
    # Plot Raw Distribution
    sns.histplot(df[feature], bins=50, kde=True, ax=ax1, color='#e74c3c')
    ax1.set_title(f'Raw Distribution: {feature}', fontweight='bold')
    ax1.set_xlabel('Original Value')
    
    # Plot Transformed Distribution
    sns.histplot(transformed_feature, bins=50, kde=True, ax=ax2, color='#2ecc71')
    ax2.set_title(f'Signed Pseudo-Log Transformed: {feature}', fontweight='bold')
    ax2.set_xlabel('Normalized Value (Log Scale)')
    
    plt.tight_layout()
    plt.savefig("artifacts/feature_transformation_comparison.png", dpi=300)
    plt.show()

# Usage:
raw_df = pd.read_csv("data/Base.csv")
generate_transformation_plot(raw_df, 'income')