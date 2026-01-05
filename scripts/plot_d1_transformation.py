"""
D1 Transformation Visualization
Creates an 'Aha Moment' figure showing identity confusion with raw D1 vs clarity with D1_inv
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plot_config

# Setup academic styling
plot_config.setup()

# Create figure with two subplots - more vertical space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_config.DOUBLE_COL_WIDTH, 2.8))

# Colors for users
colors = {'A': '#2e86ab', 'B': '#c73e1d', 'C': '#28a745'}

# --- Left Panel: Raw D1 (Time-Variant) ---
tx_days = [105, 110, 115, 120]

# User A: registered Day 100 -> D1 values are 5, 10, 15, 20
user_a_d1 = [5, 10, 15, 20]
# User B: registered Day 50 -> D1 values are 55, 60, 65, 70
user_b_d1 = [55, 60, 65, 70]  
# User C: registered Day 100 -> D1 = same as A, but slightly offset for visibility
user_c_d1 = [6, 11, 16, 21]  # Offset slightly

ax1.scatter(tx_days, user_a_d1, c=colors['A'], s=70, marker='o', label='User A', zorder=5)
ax1.scatter(tx_days, user_c_d1, c=colors['C'], s=70, marker='s', label='User C', zorder=5)
ax1.scatter(tx_days, user_b_d1, c=colors['B'], s=70, marker='^', label='User B', zorder=5)

# Add lines
ax1.plot(tx_days, user_a_d1, c=colors['A'], linestyle='--', alpha=0.5, linewidth=1.5)
ax1.plot(tx_days, user_c_d1, c=colors['C'], linestyle='--', alpha=0.5, linewidth=1.5)
ax1.plot(tx_days, user_b_d1, c=colors['B'], linestyle='--', alpha=0.5, linewidth=1.5)

# Highlight overlap region - positioned lower
ax1.axhspan(0, 30, alpha=0.12, color='red')
ax1.text(107.5, 35, 'A & C overlap!\n(Identity Confusion)', fontsize=8, ha='center', 
         color='#c73e1d', fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='#c73e1d', alpha=0.9))

ax1.set_xlabel('Transaction Day')
ax1.set_ylabel('$D_1$ (Days Since Registration)')
ax1.set_title('(a) Raw $D_1$: Time-Variant', fontweight='bold', fontsize=10)
ax1.legend(loc='upper left', fontsize=7, framealpha=0.95)
ax1.set_ylim(-5, 85)
ax1.set_xlim(102, 123)
ax1.grid(True, alpha=0.3)

# --- Right Panel: D1_inv (Time-Invariant) ---
# D1_inv = transaction_day - D1 = registration_day

user_a_d1inv = [100, 100, 100, 100]
user_b_d1inv = [50, 50, 50, 50]
user_c_d1inv = [100, 100, 100, 100]

# Offset User C slightly for visibility
tx_days_c = [x + 1 for x in tx_days]

ax2.scatter(tx_days, user_a_d1inv, c=colors['A'], s=70, marker='o', label='User A', zorder=5)
ax2.scatter(tx_days_c, user_c_d1inv, c=colors['C'], s=70, marker='s', label='User C', zorder=5)
ax2.scatter(tx_days, user_b_d1inv, c=colors['B'], s=70, marker='^', label='User B', zorder=5)

# Add horizontal lines showing stable identity
ax2.axhline(y=100, color=colors['A'], linestyle='-', alpha=0.5, linewidth=2)
ax2.axhline(y=50, color=colors['B'], linestyle='-', alpha=0.5, linewidth=2)

# Annotations - positioned outside plot area
ax2.text(122, 100, 'A & C\n(Same)', fontsize=8, ha='left', va='center', color=colors['A'],
         fontweight='bold')
ax2.text(122, 50, 'B\n(Diff.)', fontsize=8, ha='left', va='center', color=colors['B'],
         fontweight='bold')

ax2.set_xlabel('Transaction Day')
ax2.set_ylabel('$D_1^{(\\mathrm{inv})}$ (Registration Day)')
ax2.set_title('(b) $D_1^{(\\mathrm{inv})}$: Time-Invariant', fontweight='bold', fontsize=10)
ax2.legend(loc='center right', fontsize=7, framealpha=0.95)
ax2.set_ylim(30, 120)
ax2.set_xlim(102, 130)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)

plt.savefig('../results/d1_transformation.png', dpi=600, bbox_inches='tight')
plt.savefig('../results/d1_transformation.pdf', bbox_inches='tight')
print('Saved: results/d1_transformation.png and results/d1_transformation.pdf')
plt.close()
