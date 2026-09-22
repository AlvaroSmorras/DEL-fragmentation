"""Shared plotting style and helpers for the result notebooks.

The palette is the validated reference set: categorical slots are taken in fixed
order and never cycled, and only the first three are used where every pair of
series can appear together (scatter, overlaid distributions), which is the set
that clears the colour-vision-deficiency separation floors on all pairs.
Magnitude is carried by a single-hue blue ramp, never a rainbow.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- palette ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua - fixed order
GOOD, CRITICAL = "#0ca30c", "#d03b3b"        # status, never used as a series

# Single hue, light -> dark, for continuous magnitude (density, counts).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "brics_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def use_report_style() -> None:
    """Recessive axes, thin marks, ink-coloured text."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SOFT,
        "lines.linewidth": 2.0,
        "figure.dpi": 110,
        "font.size": 10,
    })


def strip_spines(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def caption(ax, text: str) -> None:
    """One line under the axes saying what the reader should take away."""
    ax.annotate(text, xy=(0, -0.22), xycoords="axes fraction",
                fontsize=9, color=INK_SOFT, va="top")


def stat_tiles(pairs, ncol: int = 4, width: float = 12.0):
    """Headline numbers as plain tiles - no chart, because there is no comparison."""
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 1.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, value) in zip(axes, pairs):
        ax.axis("off")
        ax.text(0, 0.62, value, fontsize=19, color=INK, fontweight="semibold",
                ha="left", va="center")
        ax.text(0, 0.18, label, fontsize=9, color=INK_SOFT, ha="left", va="center")
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def add_binder_ratio(frame):
    """`binder_per_nonbinder`, computed here if the table predates the column."""
    if "binder_per_nonbinder" not in frame.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            frame = frame.assign(
                binder_per_nonbinder=frame["n_binder"] / frame["n_nonbinder"]
            )
    return frame


def finite(series) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def draw_grid(mols, legends, mols_per_row: int = 4, sub_img_size=(300, 230)):
    """Structure grid that does not need an RDKit built with Cairo.

    The PNG renderer requires Cairo, which many conda RDKit builds lack; the SVG
    one is always present and scales better in a notebook anyway.
    """
    from rdkit.Chem import Draw

    if not mols:
        return None
    drawing = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row, subImgSize=sub_img_size,
        legends=legends, useSVG=True,
    )
    if isinstance(drawing, str):  # outside IPython it comes back as raw markup
        from IPython.display import SVG

        return SVG(drawing)
    return drawing
