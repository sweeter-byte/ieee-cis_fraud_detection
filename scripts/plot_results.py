"""
Professional Plotting Script for Feature Importance and EDA

Generates publication-quality figures for:
- Feature Importance (LightGBM Gain)
- Transaction Distribution Over Time (EDA)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Import professional styling
import plot_config
plot_config.setup()


def plot_feature_importance(model_path='../results/sota_quick.pkl', 
                            output_path='../results/feature_importance',
                            top_n=15):
    """
    Plot feature importance with academic styling.
    
    Args:
        model_path: Path to saved LightGBM model
        output_path: Output path (without extension)
        top_n: Number of top features to display
    """
    print("Plotting Feature Importance...")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        df_imp = pd.DataFrame({
            'feature': feature_names,
            'gain': importance
        }).sort_values('gain', ascending=True).tail(top_n)
        
        # Normalize importance for better visualization
        df_imp['gain_norm'] = df_imp['gain'] / df_imp['gain'].max()
        
        # Create figure with academic dimensions
        fig, ax = plt.subplots(figsize=plot_config.get_fig_size('single', aspect=1.2))
        
        # Color gradient based on importance
        colors = plt.cm.Blues(0.4 + 0.5 * df_imp['gain_norm'].values)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(df_imp)), df_imp['gain'], color=colors, 
                       edgecolor='#2d5986', linewidth=0.5, height=0.7)
        
        # Add value labels
        max_val = df_imp['gain'].max()
        for i, (bar, val) in enumerate(zip(bars, df_imp['gain'])):
            ax.text(val + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', ha='left', fontsize=8, color='#333333')
        
        # Clean up feature names for display
        feature_labels = [f.replace('_', ' ').replace('by uid', '(UID)') 
                         for f in df_imp['feature']]
        
        ax.set_yticks(range(len(df_imp)))
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Information Gain')
        ax.set_title('Feature Importance (Top 15)')
        
        # Extend x-axis for labels
        ax.set_xlim(0, max_val * 1.15)
        
        # Remove top and right spines (already done by config, but ensure)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save in both PNG and PDF
        plt.savefig(f'{output_path}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}.png and {output_path}.pdf")
        
    except Exception as e:
        print(f"Failed to plot feature importance: {e}")
        import traceback
        traceback.print_exc()


def plot_eda_transaction_dt(data_path='dataset', 
                            output_path='../results/eda_transaction_dt',
                            nrows=50000):
    """
    Plot transaction distribution over time with academic styling.
    
    Args:
        data_path: Path to dataset directory
        output_path: Output path (without extension)
        nrows: Number of rows to sample
    """
    print("Plotting EDA (Transaction Distribution)...")
    
    try:
        # Load data
        df = pd.read_csv(f"{data_path}/train_transaction.csv", 
                         usecols=['TransactionDT', 'isFraud', 'TransactionAmt'],
                         nrows=nrows)
        
        # Convert TransactionDT to days for better readability
        df['Days'] = df['TransactionDT'] / (24 * 3600)
        
        # Create figure
        fig, ax = plt.subplots(figsize=plot_config.get_fig_size('single', aspect=0.7))
        
        # Create histogram with KDE overlay
        counts, bins, patches = ax.hist(df['Days'], bins=60, 
                                         color='#4a90d9', alpha=0.7,
                                         edgecolor='#2d5986', linewidth=0.5,
                                         label='Transaction Count')
        
        # Add KDE curve
        from scipy import stats
        kde_x = np.linspace(df['Days'].min(), df['Days'].max(), 200)
        kde = stats.gaussian_kde(df['Days'])
        kde_y = kde(kde_x) * len(df) * (bins[1] - bins[0])
        ax.plot(kde_x, kde_y, color='#c73e1d', linewidth=1.5, 
                label='Density Estimate')
        
        ax.set_xlabel('Time (Days from Dataset Start)')
        ax.set_ylabel('Transaction Count')
        ax.set_title('Transaction Volume Over Time')
        
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(f'{output_path}.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}.png and {output_path}.pdf")
        
    except Exception as e:
        print(f"Failed to plot EDA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    plot_feature_importance()
    plot_eda_transaction_dt()
