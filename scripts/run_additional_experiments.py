"""
Additional Experiments: Robustness and Sensitivity Analysis

Generates publication-quality figures for:
- Temporal Robustness (AUC decay over time)
- Hyperparameter Sensitivity (num_leaves vs AUC)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pickle
import json

# Import professional styling
import plot_config
plot_config.setup()


def run_robustness_analysis():
    """
    Analyze model robustness over time (concept drift).
    Splits validation set into chronological chunks and measures AUC decay.
    """
    print("Running Robustness Analysis (Temporal Stability)...")
    
    try:
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from src.validation import get_single_time_split
        
        # Load and process data
        df = load_data('dataset', sample_size=10000, is_train=True)
        df = engineer_features(df, use_magic=True)
        
        train_idx, val_idx = get_single_time_split(df)
        val_df = df.iloc[val_idx].copy()
        
        # Load model
        with open('../results/sota_quick.pkl', 'rb') as f:
            model = pickle.load(f)
            
        exclude = ['isFraud', 'TransactionID', 'TransactionDT', 'uid', 'uid_encoded']
        model_features = [c for c in df.columns if c not in exclude]
        
        X_val = val_df[model_features]
        val_df['pred'] = model.predict(X_val)
        
        # Split into 3 chronological chunks
        val_df = val_df.sort_values('TransactionDT')
        n_chunks = 3
        chunk_size = len(val_df) // n_chunks
        
        aucs = []
        std_errors = []  # For error bars
        labels = ['Period 1\n(Early)', 'Period 2\n(Middle)', 'Period 3\n(Late)']
        
        print("\nTemporal Stability Results:")
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < n_chunks - 1 else len(val_df)
            chunk = val_df.iloc[start:end]
            
            if len(chunk['isFraud'].unique()) < 2:
                print(f"Chunk {i+1} has single class. Using 0.5.")
                aucs.append(0.5)
                std_errors.append(0.0)
                continue
                
            auc = roc_auc_score(chunk['isFraud'], chunk['pred'])
            aucs.append(auc)
            
            # Bootstrap for standard error
            n_boot = 100
            boot_aucs = []
            for _ in range(n_boot):
                sample = chunk.sample(frac=1.0, replace=True)
                if len(sample['isFraud'].unique()) >= 2:
                    boot_aucs.append(roc_auc_score(sample['isFraud'], sample['pred']))
            std_errors.append(np.std(boot_aucs) if boot_aucs else 0.02)
            
            print(f"  {labels[i].replace(chr(10), ' ')}: AUC = {auc:.4f} ± {std_errors[-1]:.4f}")
        
        # Create professional plot
        fig, ax = plt.subplots(figsize=plot_config.get_fig_size('single', aspect=0.8))
        
        x_pos = np.arange(len(labels))
        
        # Bar plot with error bars
        bars = ax.bar(x_pos, aucs, yerr=std_errors, 
                      color=['#4a90d9', '#6aa84f', '#e69138'],
                      edgecolor='#333333', linewidth=0.8,
                      capsize=4, error_kw={'linewidth': 1.2, 'capthick': 1.2},
                      width=0.6, alpha=0.85)
        
        # Add value labels on bars
        for bar, auc_val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{auc_val:.3f}', ha='center', va='bottom', fontsize=9, 
                    fontweight='medium')
        
        # Connect with trend line
        ax.plot(x_pos, aucs, 'o--', color='#c73e1d', linewidth=1.5, 
                markersize=6, markerfacecolor='white', markeredgewidth=1.5,
                zorder=10)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel('AUC-ROC')
        ax.set_title('Temporal Robustness Analysis')
        ax.set_ylim(0.5, 1.0)
        
        # Add horizontal reference line
        ax.axhline(y=0.8, color='#888888', linestyle=':', linewidth=1, alpha=0.7)
        ax.text(2.35, 0.805, 'Threshold', fontsize=8, color='#666666')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        plt.savefig('../results/robustness_temporal.png', dpi=600, bbox_inches='tight')
        plt.savefig('../results/robustness_temporal.pdf', bbox_inches='tight')
        plt.close()
        
        print("Saved: results/robustness_temporal.png and .pdf")
        
    except Exception as e:
        print(f"Robustness analysis failed: {e}")
        import traceback
        traceback.print_exc()


def run_sensitivity_analysis():
    """
    Analyze model sensitivity to hyperparameter (num_leaves).
    """
    print("\nRunning Hyperparameter Sensitivity Analysis...")
    
    try:
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from src.validation import get_single_time_split
        
        # Load smaller sample for speed
        df = load_data('dataset', sample_size=5000, is_train=True)
        df = engineer_features(df, use_magic=True)
        
        exclude = ['isFraud', 'TransactionID', 'TransactionDT', 'uid', 'uid_encoded']
        features = [c for c in df.columns if c not in exclude]
        
        train_idx, val_idx = get_single_time_split(df)
        X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]['isFraud']
        X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx]['isFraud']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Test different complexities
        leaves_list = [16, 32, 64, 128, 256, 512]
        aucs = []
        
        for leaves in leaves_list:
            print(f"  Testing num_leaves={leaves}...")
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.05,
                'num_leaves': leaves,
                'is_unbalance': True,
                'verbose': -1,
                'seed': 42
            }
            
            clf = lgb.train(params, train_data, num_boost_round=100, 
                           valid_sets=[val_data],
                           callbacks=[lgb.early_stopping(10, verbose=False)])
            aucs.append(clf.best_score['valid_0']['auc'])
        
        print(f"Sensitivity Results: {dict(zip(leaves_list, aucs))}")
        
        # Create professional plot
        fig, ax = plt.subplots(figsize=plot_config.get_fig_size('single', aspect=0.8))
        
        # Main line plot
        ax.plot(leaves_list, aucs, 'o-', color='#2e86ab', linewidth=2, 
                markersize=8, markerfacecolor='white', markeredgewidth=2,
                label='Validation AUC')
        
        # Highlight optimal point
        best_idx = np.argmax(aucs)
        ax.scatter([leaves_list[best_idx]], [aucs[best_idx]], 
                   color='#c73e1d', s=100, zorder=10, 
                   marker='*', label=f'Optimal ({leaves_list[best_idx]} leaves)')
        
        # Add value annotations
        for i, (x, y) in enumerate(zip(leaves_list, aucs)):
            offset = 0.012 if i != best_idx else 0.018
            ax.annotate(f'{y:.3f}', (x, y + offset), ha='center', fontsize=8)
        
        ax.set_xlabel('Number of Leaves (Model Complexity)')
        ax.set_ylabel('Validation AUC')
        ax.set_title('Hyperparameter Sensitivity Analysis')
        ax.set_xscale('log', base=2)
        ax.set_xticks(leaves_list)
        ax.set_xticklabels([str(l) for l in leaves_list])
        
        # Set y-axis limits with some padding
        y_min = min(aucs) - 0.02
        y_max = max(aucs) + 0.03
        ax.set_ylim(y_min, y_max)
        
        ax.legend(loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        plt.savefig('../results/sensitivity_leaves.png', dpi=600, bbox_inches='tight')
        plt.savefig('../results/sensitivity_leaves.pdf', bbox_inches='tight')
        plt.close()
        
        print("Saved: results/sensitivity_leaves.png and .pdf")
        
    except Exception as e:
        print(f"Sensitivity analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_robustness_analysis()
    run_sensitivity_analysis()
