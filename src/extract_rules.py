import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.model import clustering

def main():
    # 1. Load Data
    X_train_final = pd.read_parquet("artifacts/X_train_processed.parquet")
    feature_names = X_train_final.columns.tolist()
    
    # Subsample 2000 rows for a cleaner density plot
    X_sample_df = X_train_final.sample(2000, random_state=42)
    X_sample = X_sample_df.values

    # 2. Execute Subtractive Clustering with a LARGER RADIUS
    # radius=1.5 to 2.0 usually results in 4-8 very strong, interpretable rules
    logger_info = "Extracting strategic behavioral prototypes..."
    print(logger_info)
    centres = clustering.subtractive_cluster(X_sample, radius=1.8) 
    
    # 3. Professional Visualization (JointGrid for absolute control)
    plt.figure(figsize=(12, 10))
    sns.set_style("white")

    # Initialize the JointGrid
    g = sns.JointGrid(data=X_sample_df, x=feature_names[0], y=feature_names[1])

    # A. The "Behavioral Landscape" - Smooth Density Contours
    g.plot_joint(sns.kdeplot, fill=True, thresh=0.05, cmap="Blues", alpha=0.8, levels=10)
    
    # B. The "Fuzzy Anchors" - Prominent Red Stars
    g.ax_joint.scatter(
        centres[:, 0], centres[:, 1], 
        s=500, marker='*', c='#ff3333', 
        edgecolor='black', linewidth=1.5,
        label='Fuzzy Prototypes (Rule Anchors)',
        zorder=10
    )

    # C. Marginal Distributions (Top and Right bars)
    g.plot_marginals(sns.histplot, color="#3498db", fill=True, bins=20)

    # D. Final Styling
    g.fig.suptitle("Clean Behavioral Landscape & Fuzzy Rule Anchors", fontweight='bold', y=1.03, fontsize=16)
    g.set_axis_labels(f"{feature_names[0].replace('_', ' ').title()}", 
                      f"{feature_names[1].replace('_', ' ').title()}", 
                      fontweight='bold', fontsize=12)
    
    # Save for the M.Tech Report
    output_plot = "artifacts/fuzzy_clusters_clean.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save the actual rules for the ANFIS model
    rules = clustering.clusters_to_rules(centres, feature_names)
    clustering.save_rules(rules, "artifacts/fuzzy_rules.json")
    print(f"SUCCESS: Found {len(rules)} distinct rules. Plot saved to {output_plot}")

if __name__ == "__main__":
    main()