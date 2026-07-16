"""Shared chart style: validated categorical palette + recessive chrome."""

import matplotlib as mpl

# Validated categorical order (adjacent-pair CVD-safe); magenta/yellow are
# sub-3:1 on the light surface, so any series using them must be direct-labeled.
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
SERIES = [BLUE, GREEN, MAGENTA, YELLOW]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "figure.dpi": 150,
    })
