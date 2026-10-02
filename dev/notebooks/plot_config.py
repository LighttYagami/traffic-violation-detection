"""
plot_config.py
===============
Shared plot style configuration for all analysis notebooks.
Import in every notebook: from plot_config import *
"""

import matplotlib.pyplot as plt

# ─── Class Colors ──
COLORS = {
    "bike": "#2196F3",           # blue
    "helmet": "#4CAF50",         # green
    "no-helmet": "#F44336",      # red
    "number-plate": "#FF9800",   # orange
}

# ─── Figure Sizes ──
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_MULTI = (15, 5)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)

# ─── Font Sizes ──
FONT_LABEL = 12
FONT_TITLE = 14
FONT_TICK = 10

# ─── Grid ──
GRID_ALPHA = 0.3

# ─── Save Settings ──
SAVE_DPI = 150


def apply_style():
    """Apply consistent plot style."""
    plt.rcParams.update({
        "figure.figsize": FIGSIZE_SINGLE,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "axes.grid": True,
        "grid.alpha": GRID_ALPHA,
        "figure.dpi": 100,
        "savefig.dpi": SAVE_DPI,
    })


def save_fig(fig, name: str, exp_dir: str):
    """Save figure as both PNG and PDF."""
    import os
    plots_dir = os.path.join(exp_dir, "plots", "custom")
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(os.path.join(plots_dir, f"{name}.pdf"), bbox_inches="tight")
    print(f"Saved: {name}.png / .pdf")
