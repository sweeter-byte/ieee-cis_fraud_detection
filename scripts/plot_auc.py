"""
Professional ROC Curve Comparison Plot

Generates publication-quality ROC curves comparing:
- Baseline (SMOTE + XGBoost)
- Proposed SOTA (Magic Features + LightGBM)
- Ablation (SOTA + SMOTE)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import argparse

# Import professional styling
import plot_config
plot_config.setup()


def plot_roc_comparison(baseline_csv: str, sota_csv: str, ablation_csv: str, output_path: str):
    """
    Plot ROC curves with academic styling.
    
    Args:
        baseline_csv: Path to baseline predictions CSV
        sota_csv: Path to SOTA predictions CSV
        ablation_csv: Path to ablation predictions CSV
        output_path: Output path (without extension)
    """
    # Create figure with square aspect (common for ROC)
    fig, ax = plt.subplots(figsize=plot_config.get_fig_size('single', aspect=1.0))
    
    # Define line styles for each method
    styles = {
        'baseline': {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9},
        'sota': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 1.0},
        'ablation': {'color': '#2ca02c', 'linestyle': ':', 'linewidth': 1.8, 'alpha': 0.9},
    }
    
    legend_handles = []
    
    # Process Baseline
    if Path(baseline_csv).exists():
        df_base = pd.read_csv(baseline_csv)
        fpr, tpr, _ = roc_curve(df_base['y_true'], df_base['y_pred_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Baseline (AUC = {roc_auc:.3f})', **styles['baseline'])
        print(f"Baseline AUC: {roc_auc:.4f}")
    else:
        print(f"Warning: {baseline_csv} not found.")

    # Process SOTA (Proposed)
    if Path(sota_csv).exists():
        df_sota = pd.read_csv(sota_csv)
        fpr, tpr, _ = roc_curve(df_sota['y_true'], df_sota['y_pred_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Proposed (AUC = {roc_auc:.3f})', **styles['sota'])
        print(f"Proposed AUC: {roc_auc:.4f}")
    else:
        print(f"Warning: {sota_csv} not found.")

    # Process Ablation
    if Path(ablation_csv).exists():
        df_abl = pd.read_csv(ablation_csv)
        fpr, tpr, _ = roc_curve(df_abl['y_true'], df_abl['y_pred_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Ablation (AUC = {roc_auc:.3f})', **styles['ablation'])
        print(f"Ablation AUC: {roc_auc:.4f}")
    else:
        print(f"Warning: {ablation_csv} not found.")

    # Random classifier reference line
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.0, 
            alpha=0.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve Comparison')
    
    # Legend
    ax.legend(loc='lower right', framealpha=0.95)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Ensure square aspect
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save
    output_base = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    plt.savefig(f'{output_base}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_base}.png and {output_base}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ROC Curve Comparison')
    parser.add_argument('--baseline', default='../results/baseline_metrics.csv',
                        help='Path to baseline predictions CSV')
    parser.add_argument('--sota', default='../results/sota_metrics_improved.csv',
                        help='Path to SOTA predictions CSV')
    parser.add_argument('--ablation', default='../results/ablation_smote_metrics.csv',
                        help='Path to ablation predictions CSV')
    parser.add_argument('--output', default='../results/roc_comparison_final',
                        help='Output path (without extension)')
    args = parser.parse_args()
    
    plot_roc_comparison(args.baseline, args.sota, args.ablation, args.output)
