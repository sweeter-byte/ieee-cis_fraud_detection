"""
Data Efficiency Visualization
Creates a professional plot showing AUC vs training data size
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plot_config

# Setup academic styling
plot_config.setup()

# Load data
df = pd.read_csv('../results/data_efficiency.csv')

# Create figure
fig, ax1 = plt.subplots(figsize=plot_config.get_fig_size('single', aspect=0.8))

# Primary axis: AUC
color1 = '#2e86ab'  # Primary blue
ax1.plot(df['n_samples']/1000, df['auc'], 'o-', color=color1, 
         linewidth=2, markersize=8, label='AUC-ROC')
ax1.set_xlabel('Training Data Size (×1000 samples)')
ax1.set_ylabel('AUC-ROC', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.82, 0.95)

# Add horizontal reference line at 0.90
ax1.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(550, 0.902, 'AUC=0.90', fontsize=9, color='gray')

# Secondary axis: Training time
ax2 = ax1.twinx()
color2 = '#c73e1d'  # Accent red
ax2.plot(df['n_samples']/1000, df['training_time'], 's--', color=color2,
         linewidth=2, markersize=6, alpha=0.7, label='Training Time')
ax2.set_ylabel('Training Time (seconds)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# Annotations for key points
ax1.annotate(f"AUC={df.iloc[-1]['auc']:.3f}", 
             xy=(df.iloc[-1]['n_samples']/1000, df.iloc[-1]['auc']),
             xytext=(450, 0.92),
             fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray', lw=1))

# Grid and legend
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xscale('log')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', framealpha=0.9)

ax1.set_title('Data Efficiency Analysis')

plt.tight_layout()
plt.savefig('../results/data_efficiency.png', dpi=600, bbox_inches='tight')
plt.savefig('../results/data_efficiency.pdf', bbox_inches='tight')
print('Saved: results/data_efficiency.png and results/data_efficiency.pdf')
plt.close()
