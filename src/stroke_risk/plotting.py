"""Editorial plotting theme.

A single, reusable style so every chart in the project looks consistent and
publication-clean: muted context, one accent colour to carry the story, no
chartjunk, direct labelling over legends.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette --------------------------------------------------------------------
INK = "#1A1A1A"      # primary text
SUBTLE = "#6B7280"   # secondary text / axes
GRID = "#EAECEF"     # gridlines
MUTED = "#C7CFD8"    # context / "no stroke"
ACCENT = "#E4572E"   # the story / "stroke"
PRIMARY = "#1B3A5B"  # secondary highlight

# Map the binary target to colours used throughout.
STROKE_COLORS = {0: MUTED, 1: ACCENT}
STROKE_LABELS = {0: "No stroke", 1: "Stroke"}


def set_theme() -> None:
    """Apply the project-wide matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8.5, 5),
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SUBTLE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelcolor": SUBTLE,
            "axes.labelsize": 11,
            "xtick.color": SUBTLE,
            "ytick.color": SUBTLE,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "text.color": INK,
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def despine(ax: plt.Axes, *, left: bool = True, bottom: bool = False) -> None:
    """Hide chart spines for a cleaner look."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def add_titles(ax: plt.Axes, headline: str, subtitle: str | None = None) -> None:
    """Add an editorial headline (the takeaway) plus a lighter descriptive subtitle."""
    y = 1.14 if subtitle else 1.06
    ax.text(
        0, y, headline, transform=ax.transAxes,
        fontsize=15, fontweight="bold", color=INK, va="bottom",
    )
    if subtitle:
        ax.text(
            0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10.5, color=SUBTLE, va="bottom",
        )
