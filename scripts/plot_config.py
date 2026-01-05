"""
Professional Plotting Configuration for Academic Papers

This module provides a unified matplotlib configuration that ensures all figures
meet academic publication standards (IEEE, ACM, etc.).

Usage:
    import plot_config
    plot_config.setup()
    # Then create your plots as usual
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# Academic Color Palettes
# A carefully curated palette for academic publications
# - High contrast for print and colorblind accessibility
ACADEMIC_COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Magenta/Purple
    'tertiary': '#F18F01',     # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#4A4A4A',      # Dark gray
    'light': '#E8E8E8',        # Light gray
}

# Tab10-based but with better print qualities
PAPER_PALETTE = [
    '#1f77b4',  # Blue
    '#d62728',  # Red
    '#2ca02c',  # Green
    '#ff7f0e',  # Orange
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]

# Figure Dimensions (IEEE Standard)
# Single column: 3.5 inches
# Double column: 7.16 inches
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.16
GOLDEN_RATIO = 1.618

def get_fig_size(width='single', aspect=None):
    """
    Get figure size for academic papers.
    
    Args:
        width: 'single' for single-column, 'double' for double-column
        aspect: height/width ratio (default: 1/golden_ratio)
    
    Returns:
        tuple: (width, height) in inches
    """
    w = SINGLE_COL_WIDTH if width == 'single' else DOUBLE_COL_WIDTH
    if aspect is None:
        aspect = 1.0 / GOLDEN_RATIO
    return (w, w * aspect)

# Main Setup Function
def setup(font_scale=1.0, use_latex=False):
    """
    Apply academic publication styling to matplotlib.
    
    Args:
        font_scale: Scale factor for all fonts (default: 1.0)
        use_latex: Whether to use LaTeX for text rendering (requires LaTeX installation)
    """
    # Base font sizes
    SMALL_SIZE = 9 * font_scale
    MEDIUM_SIZE = 10 * font_scale
    LARGE_SIZE = 11 * font_scale
    TITLE_SIZE = 12 * font_scale
    
    # Font family - use serif for IEEE papers
    if use_latex:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        })
    else:
        plt.rcParams.update({
            'text.usetex': False,
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        })
    
    # Font sizes
    plt.rcParams.update({
        'font.size': MEDIUM_SIZE,
        'axes.titlesize': LARGE_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': TITLE_SIZE,
    })
    
    # Figure defaults
    plt.rcParams.update({
        'figure.figsize': get_fig_size('single'),
        'figure.dpi': 150,  # For display
        'savefig.dpi': 600,  # For publication
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    # Axes styling
    plt.rcParams.update({
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.labelpad': 4,
        'axes.titlepad': 8,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': cycler('color', PAPER_PALETTE),
    })
    
    # Tick styling
    plt.rcParams.update({
        'xtick.major.size': 4,
        'xtick.major.width': 0.8,
        'xtick.minor.size': 2,
        'xtick.minor.width': 0.6,
        'xtick.direction': 'out',
        'xtick.color': '#333333',
        'ytick.major.size': 4,
        'ytick.major.width': 0.8,
        'ytick.minor.size': 2,
        'ytick.minor.width': 0.6,
        'ytick.direction': 'out',
        'ytick.color': '#333333',
    })
    
    # Line styling
    plt.rcParams.update({
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })
    
    # Legend styling
    plt.rcParams.update({
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.3,
        'legend.handlelength': 1.5,
    })
    
    # Grid styling (when enabled)
    plt.rcParams.update({
        'grid.color': '#e0e0e0',
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
    })

def reset():
    """Reset matplotlib to default settings."""
    mpl.rcdefaults()

def create_figure(width='single', aspect=None, **kwargs):
    """
    Create a figure with academic paper dimensions.
    
    Args:
        width: 'single' or 'double' column
        aspect: height/width ratio
        **kwargs: Additional arguments for plt.figure()
    
    Returns:
        matplotlib.figure.Figure
    """
    figsize = get_fig_size(width, aspect)
    return plt.figure(figsize=figsize, **kwargs)

def create_subplots(nrows=1, ncols=1, width='single', aspect=None, **kwargs):
    """
    Create subplots with academic paper dimensions.
    
    Args:
        nrows, ncols: Number of subplot rows/columns
        width: 'single' or 'double' column
        aspect: height/width ratio per subplot
        **kwargs: Additional arguments for plt.subplots()
    
    Returns:
        tuple: (figure, axes)
    """
    base_w, base_h = get_fig_size(width, aspect)
    figsize = (base_w, base_h * nrows / ncols) if aspect is None else (base_w, base_h * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

def save_figure(fig, filename, formats=None):
    """
    Save figure in multiple formats for publication.
    
    Args:
        fig: matplotlib figure
        filename: Base filename (without extension)
        formats: List of formats (default: ['pdf', 'png'])
    """
    if formats is None:
        formats = ['pdf', 'png']
    
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', format=fmt, dpi=600, bbox_inches='tight')
        print(f"Saved: {filename}.{fmt}")

# Auto-setup when imported
setup()

if __name__ == "__main__":
    # Demo plot
    import numpy as np
    
    setup()
    
    fig, ax = plt.subplots(figsize=get_fig_size('single'))
    
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='Sine')
    ax.plot(x, np.cos(x), label='Cosine')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Demo: Academic Plot Style')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../results/demo_style.png', dpi=300)
    print("Demo plot saved to results/demo_style.png")
