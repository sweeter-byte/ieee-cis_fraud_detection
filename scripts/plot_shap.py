"""
Professional SHAP Analysis Plot

Generates publication-quality SHAP summary plot for model interpretation.
"""
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Import professional styling
import plot_config
plot_config.setup()

from src.data_loader import load_data
from src.feature_engineering import engineer_features


def plot_shap_summary(model_path: str, data_path: str, output_path: str, quick: bool = False,max_display: int = 15):
    """
    Generate SHAP summary plot with academic styling.
    
    Args:
        model_path: Path to saved model pickle
        data_path: Path to dataset directory
        output_path: Output path (without extension)
        quick: If True, use fewer samples for faster computation
        max_display: Number of features to display
    """
    print(f"Loading model from {model_path}...")
    
    if not Path(model_path).exists():
        print("Model not found. Run SOTA experiment first.")
        return
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load and process data
    print("Loading data for SHAP analysis...")
    sample_size = 1000 if quick else 10000
    df = load_data(data_path, sample_size=sample_size, is_train=True)
    df = engineer_features(df, use_magic=True)
    
    # Filter features to match model
    exclude = ['isFraud', 'TransactionID', 'TransactionDT', 'uid', 'uid_encoded']
    X = df[[c for c in df.columns if c not in exclude]]
    
    print(f"Computing SHAP values for {X.shape[0]} samples...")
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle LightGBM binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (fraud)
    
    # Apply professional styling
    # Note: SHAP's summary_plot uses its own styling, so we override some params
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(plot_config.SINGLE_COL_WIDTH * 1.2, 
                              plot_config.SINGLE_COL_WIDTH * 1.4))
    
    # Create SHAP summary plot
    shap.summary_plot(shap_values, X, 
                      max_display=max_display, 
                      show=False,
                      plot_size=None,
                      color_bar_label='Feature Value')
    
    # Get current axes and apply styling
    ax = plt.gca()
    ax.set_xlabel('SHAP Value (Impact on Fraud Prediction)', fontsize=10)
    ax.set_title('Feature Impact Analysis (SHAP)', fontsize=11)
    
    # Clean up feature names in y-axis
    yticks = ax.get_yticklabels()
    new_labels = []
    for label in yticks:
        text = label.get_text()
        # Shorten long feature names for readability
        text = text.replace('_', ' ').replace('by uid', '(UID)')
        if len(text) > 25:
            text = text[:22] + '...'
        new_labels.append(text)
    ax.set_yticklabels(new_labels, fontsize=8)
    
    plt.tight_layout()
    
    # Save
    output_base = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    plt.savefig(f'{output_base}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.close()
    
    # Reset to default config
    plot_config.setup()
    
    print(f"Saved: {output_base}.png and {output_base}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SHAP Summary Plot')
    parser.add_argument('--model', default='../results/sota_metrics.pkl',
                        help='Path to saved model')
    parser.add_argument('--data', default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output', default='../results/shap_summary',
                        help='Output path (without extension)')
    parser.add_argument('--quick', action='store_true',
                        help='Use fewer samples for faster computation')
    args = parser.parse_args()
    
    plot_shap_summary(args.model, args.data, args.output, args.quick)
