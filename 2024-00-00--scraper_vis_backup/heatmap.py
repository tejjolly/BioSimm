#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

# =============================================================================
# GLOBAL STYLE / TUNABLES
# =============================================================================
FIGSIZE = (5, 4)

FONT_TITLE = 14
FONT_AXES_LABEL = 12
FONT_TICKS = 10
FONT_CBAR_LABEL = 12

ANNOTATE_VALUES = True
ANNOT_FONTSIZE = 8
AUTO_ANNOT_TEXTCOLOR = True
ANNOT_TEXTCOLOR_FALLBACK = "black"

CMAP_NAME = "BuPu"
BAD_COLOR = "lightgray"
UNDER_COLOR = None
OVER_COLOR  = None

# --- NEW: internal gridlines ---
SHOW_CELL_BORDERS = True
BORDER_COLOR = (0.65, 0.65, 0.65)   # gray-ish
BORDER_LINEWIDTH = 0.6

# =============================================================================
# DATA / FILTERS
# =============================================================================
DATA_PATH = "../data/data_arnav.csv"
SOURCE_KEEP = "arnav"
CONDITION_KEEP = None
LOCATION_KEEP  = None

# =============================================================================
# BINNING FOR AXES
# =============================================================================
STENOSIS_BIN = 0.10
LENGTH_BIN = 0.10
AGGFUNC = "mean"

# =============================================================================
# DISCRETE COLOR BINS
# =============================================================================
FFR_STEP = 0.10
FFR_VMIN = 0.50
FFR_VMAX = 1.00

# CFR: bounds 2.75–3.25, step 0.05 (you currently had 0.10)
CFR_STEP = 0.5
CFR_VMIN = 1
CFR_VMAX = 3.5

# =============================================================================
# OUTPUT
# =============================================================================
SAVE_FFR_PATH = "ffr_heatmap.svg"
SAVE_CFR_PATH = "cfr_heatmap.svg"
DPI = 600


# =============================================================================
# HELPERS
# =============================================================================
def parse_stenosis_to_fraction(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    has_pct = raw.str.contains("%", na=False)
    x = pd.to_numeric(raw.str.replace("%", "", regex=False), errors="coerce")
    x.loc[has_pct] = x.loc[has_pct] / 100.0

    arr = x.to_numpy()
    if np.isfinite(arr).any() and np.nanmax(arr) > 1.5:
        x = x / 100.0
    return x

def round_to_nearest_step_half_up(x: pd.Series, step: float) -> pd.Series:
    return np.floor((x + step/2) / step) * step

def bin_continuous(x: pd.Series, step: float) -> pd.Series:
    return np.floor((x + step/2) / step) * step

def rel_luminance(rgba):
    r, g, b = rgba[:3]
    return 0.299*r + 0.587*g + 0.114*b

def choose_text_color_from_rgba(rgba, threshold=0.55):
    return "black" if rel_luminance(rgba) > threshold else "white"

def make_heat_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (df.pivot_table(index="StenosisBin", columns="LengthBin", values=value_col, aggfunc=AGGFUNC)
              .sort_index(axis=0).sort_index(axis=1))

def make_edges(vmin: float, vmax: float, step: float) -> np.ndarray:
    n = int(np.round((vmax - vmin) / step))
    edges = vmin + step * np.arange(n + 1)
    edges = np.round(edges, 10)
    edges[-1] = vmax
    return edges

def add_cell_borders(ax, nrows, ncols):
    # minor ticks at cell boundaries: -0.5, 0.5, 1.5, ...
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=BORDER_COLOR, linewidth=BORDER_LINEWIDTH)
    ax.tick_params(which="minor", bottom=False, left=False)

def plot_heatmap_discrete(
    heat: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    annot_fmt: str,
    vmin: float,
    vmax: float,
    step: float,
    save_path: str | None = None
):
    Z = heat.to_numpy(dtype=float)
    Zmask = np.ma.masked_invalid(Z)

    x_vals = heat.columns.to_numpy()
    y_vals = heat.index.to_numpy()

    edges = make_edges(vmin, vmax, step)
    n_bins = len(edges) - 1

    cmap = plt.get_cmap(CMAP_NAME, n_bins).copy()
    cmap.set_bad(BAD_COLOR)
    if UNDER_COLOR is not None:
        cmap.set_under(UNDER_COLOR)
    if OVER_COLOR is not None:
        cmap.set_over(OVER_COLOR)

    norm = colors.BoundaryNorm(boundaries=edges, ncolors=n_bins, clip=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    im = ax.imshow(
        Zmask,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # --- NEW: internal borders ---
    if SHOW_CELL_BORDERS:
        add_cell_borders(ax, nrows=Z.shape[0], ncols=Z.shape[1])

    # ticks/labels
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=45, ha="right", fontsize=FONT_TICKS)

    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{100*v:.0f}%" for v in y_vals], fontsize=FONT_TICKS)

    ax.set_xlabel("Length [cm]" if LENGTH_BIN is not None else "Length", fontsize=FONT_AXES_LABEL)
    ax.set_ylabel(f"Stenosis [%]", fontsize=FONT_AXES_LABEL)
    # ax.set_title(title, fontsize=FONT_TITLE)

    # --- FIX: no triangular endcaps ---
    # extend='neither' (or omit extend entirely)
    cbar = fig.colorbar(im, ax=ax, extend="neither")
    cbar.set_label(cbar_label, fontsize=FONT_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICKS)
    cbar.set_ticks(edges)
    cbar.set_ticklabels([f"{e:g}" for e in edges])

    # annotate
    if ANNOTATE_VALUES:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if np.isfinite(val):
                    if AUTO_ANNOT_TEXTCOLOR:
                        rgba = im.cmap(im.norm(val))
                        txt_color = choose_text_color_from_rgba(rgba)
                    else:
                        txt_color = ANNOT_TEXTCOLOR_FALLBACK
                    ax.text(j, i, annot_fmt.format(val),
                            ha="center", va="center",
                            fontsize=ANNOT_FONTSIZE, color=txt_color)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", transparent=True)
    plt.show()


# =============================================================================
# LOAD + FILTER + CLEAN
# =============================================================================
df = pd.read_csv(DATA_PATH, sep=None, engine="python")

if "source" not in df.columns:
    raise KeyError("Expected a column named 'source' in the CSV.")

df = df[df["source"].astype(str).str.lower().eq(SOURCE_KEEP.lower())].copy()

if CONDITION_KEEP is not None and "Condition" in df.columns:
    df = df[df["Condition"].astype(str) == CONDITION_KEEP].copy()

if LOCATION_KEEP is not None and "Location" in df.columns:
    df = df[df["Location"].astype(str) == LOCATION_KEEP].copy()

needed = ["Stenosis Percentage", "Length", "P_d/P_a", "CFR"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

df["StenosisFrac"] = parse_stenosis_to_fraction(df["Stenosis Percentage"])
df["LengthVal"]    = pd.to_numeric(df["Length"], errors="coerce")
df["FFR"]          = pd.to_numeric(df["P_d/P_a"], errors="coerce")
df["CFR_val"]      = pd.to_numeric(df["CFR"], errors="coerce")

df = df.dropna(subset=["StenosisFrac", "LengthVal"]).copy()
if df.empty:
    raise ValueError(f"No rows left after filtering (source='{SOURCE_KEEP}', condition={CONDITION_KEEP}, location={LOCATION_KEEP}).")

# =============================================================================
# BIN AXES
# =============================================================================
df["StenosisBin"] = round_to_nearest_step_half_up(df["StenosisFrac"], STENOSIS_BIN)

if LENGTH_BIN is None:
    df["LengthBin"] = df["LengthVal"]
else:
    df["LengthBin"] = bin_continuous(df["LengthVal"], float(LENGTH_BIN))

# =============================================================================
# BUILD HEAT GRIDS
# =============================================================================
heat_ffr = make_heat_pivot(df.dropna(subset=["FFR"]), value_col="FFR")
heat_cfr = make_heat_pivot(df.dropna(subset=["CFR_val"]), value_col="CFR_val")

# =============================================================================
# PLOT
# =============================================================================
plot_heatmap_discrete(
    heat_ffr,
    title=f"FFR Heat Map (P_d/P_a) — source='{SOURCE_KEEP}'",
    cbar_label="FFR",
    annot_fmt="{:.2f}",
    vmin=FFR_VMIN, vmax=FFR_VMAX, step=FFR_STEP,
    save_path=SAVE_FFR_PATH
)

plot_heatmap_discrete(
    heat_cfr,
    title=f"CFR Heat Map — source='{SOURCE_KEEP}'",
    cbar_label="CFR",
    annot_fmt="{:.2f}",
    vmin=CFR_VMIN, vmax=CFR_VMAX, step=CFR_STEP,
    save_path=SAVE_CFR_PATH
)
