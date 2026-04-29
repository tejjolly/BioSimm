#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 01:18:34 2025

@author: tejjolly
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines  # <-- used to create custom legend handles

# Toggle if you want to see the PairGrid or not
pairgrid_switch = True

# ---------------------------------------
# 1) Load and Clean Data
# ---------------------------------------
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary2.csv')
# df_full = df_full[df_full['Location'] != 'LCX']

n_bins = 5
cmap = plt.get_cmap('Reds', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

df_full['Condition'] = df_full['Condition'].astype(str)
df_full['iFR'] = np.where(df_full['Condition'] == 'Non-hyperemic', df_full['P_d/P_a'], np.nan)
df_full['FFR'] = np.where(df_full['Condition'] == 'Hyperemic', df_full['P_d/P_a'], np.nan)
df_full.drop(columns=['P_d/P_a'], inplace=True)
df_full.drop(columns=['Condition', 'Geometry Number'], inplace=True, errors='ignore')

df_full.rename(columns={
    'Stenosis Percentage': 'Stenosis',
    'Average Flow': 'Flow',
    'Rtotal_cor Value': 'R. mult.',
    'WSS_Area_Bifur_min': 'WSS_Bif_Area_min'
}, inplace=True)

columns_of_interest = [
    'WSS_LMB','WSS_min','WSS_TE_min','WSS_LE_min',
    'WSS_Avg_Area_min','WSS_TE_Area_min','WSS_LE_Area_min','WSS_Bif_Area_min',
    'Flow','HMR','BMR/HMR','iFR','FFR','CFR','CFR/FFR','HSR'
]

# Subset to these columns if they exist
existing_cols = [col for col in columns_of_interest if col in df_full.columns]
df_subset = df_full[existing_cols].copy()

df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
df_subset["WSS_LMB"] = df_subset["WSS_LMB"].replace(0, np.nan)
df_subset.dropna(subset=["WSS_LMB"], inplace=True)

# # Normalize selected WSS columns by WSS_LMB
# for col in ["WSS_min", "WSS_TE_min", "WSS_LE_min"]:
#     if col in df_subset.columns:
#         df_subset[col] = df_subset[col] / df_subset["WSS_LMB"]

vars_to_plot = [
    'WSS_LE_min','WSS_TE_min','WSS_min','WSS_Bif_Area_min',
    'WSS_LE_Area_min','WSS_TE_Area_min','WSS_Avg_Area_min','Flow',
    'HMR','BMR/HMR','FFR','CFR','CFR/FFR'
]

# Focus only on rows where HSR is not NaN
df_plot = df_subset.dropna(subset=["HSR"])

# ---------------------------------------
# 2) Custom PairGrid Scatter Function
# ---------------------------------------
def scatter_cmap_marker(x, y, data=None, c=None, cmap=None, norm=None,
                        edgecolor="k", s=20, **kwargs):
    """
    For PairGrid: create 4 categories of markers based on data['CFR'] & data['FFR']:
       cat1: CFR>2, FFR>0.8 -> 'o'
       cat2: CFR>2, FFR<0.8 -> '^'
       cat3: CFR<2, FFR>0.8 -> 'v'
       cat4: CFR<2, FFR<0.8 -> 'x'

    - Plot is colored by 'c' (HSR).
    - Markers in the plot reflect the color scale.
    - The legend can be made with black markers if desired.
    """
    if "color" in kwargs:
        kwargs.pop("color")  # remove seaborn default

    if data is None or ("CFR" not in data.columns) or ("FFR" not in data.columns):
        # fallback: single scatter
        plt.scatter(x, y, c=c, cmap=cmap, norm=norm, edgecolors=edgecolor, s=s, **kwargs)
        return
    
    cat_definitions = {
        "cat1": {
            "mask": (data['CFR'] > 2) & (data['FFR'] > 0.8),
            "marker": "o",
            "label": "CFR>2, FFR>0.8"
        },
        "cat2": {
            "mask": (data['CFR'] > 2) & (data['FFR'] < 0.8),
            "marker": "o",
            "label": "CFR>2, FFR<0.8"
        },
        "cat3": {
            "mask": (data['CFR'] < 2) & (data['FFR'] > 0.8),
            "marker": "v",
            "label": "CFR<2, FFR>0.8"
        },
        "cat4": {
            "mask": (data['CFR'] < 2) & (data['FFR'] < 0.8),
            "marker": "s",
            "label": "CFR<2, FFR<0.8"
        },
    }

    # (A) Plot with color-coded markers but don't auto-legend them
    for cat_key, cat_info in cat_definitions.items():
        mask = cat_info["mask"].values
        plt.scatter(
            x[mask],
            y[mask],
            c=None if c is None else c[mask],
            cmap=cmap,
            norm=norm,
            marker=cat_info["marker"],
            edgecolors=edgecolor,
            s=s,
            label=None,  # <-- no automatic label in the scatter
            **kwargs
        )
    # (B) If you want a black-marker legend in the PairGrid,
    #     you could create custom handles below and call plt.legend(...).
    #     Usually in PairGrids we skip the marker-based legend,
    #     but you can do something like:
    #
    # cat_handles = []
    # for cat_key, cat_info in cat_definitions.items():
    #     cat_marker = cat_info["marker"]
    #     cat_label = cat_info["label"]
    #     handle = mlines.Line2D([], [], marker=cat_marker,
    #                            color="black", markerfacecolor="black",
    #                            markeredgecolor="black", linestyle='None',
    #                            label=cat_label)
    #     cat_handles.append(handle)
    # plt.legend(handles=cat_handles, loc="best", frameon=True)


# ---------------------------------------
# 3) Create the PairGrid (if desired)
# ---------------------------------------
if pairgrid_switch:
    g = sns.PairGrid(df_plot, vars=vars_to_plot, diag_sharey=False)
    
    # Bins for HSR color
    min_val = df_plot["HSR"].min()
    max_val = df_plot["HSR"].max()
    boundaries = np.linspace(min_val, max_val, 6)
    cmap_pg = plt.get_cmap("RdYlGn_r", 5)
    norm_pg = mcolors.BoundaryNorm(boundaries, ncolors=cmap_pg.N, clip=True)
    
    # Map the custom scatter to the lower triangle
    g.map_lower(
        scatter_cmap_marker,
        data=df_plot,
        c=df_plot["HSR"],   # color array
        cmap=cmap_pg,
        norm=norm_pg,
        edgecolor="k",
        s=20
    )
    # Diagonal hist
    # g.map_diag(sns.histplot, fill=True)
    # g.map_diag(sns.histplot, fill=True, binrange=(df_plot.min().min(), df_plot.max().max()))
    
    # Add a discrete colorbar (once)
    fig = g.fig
    cax = fig.add_axes([.975, 0.3, 0.02, 0.4])
    sm = plt.cm.ScalarMappable(norm=norm_pg, cmap=cmap_pg)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels([f"{v:.1f}" for v in boundaries])
    
    scale_cbar = 4
    cbar.ax.tick_params(labelsize=12 * scale_cbar)
    cbar.set_label("HSR [mmHg/cm/s]", fontsize=14 * scale_cbar)
    
    # for ax in g.axes.flatten():  # Iterate through all subplots
    #     if ax is not None:
    #         ax.set_xlim(df_plot[vars_to_plot].min().min(), df_plot[vars_to_plot].max().max())
    #         ax.set_ylim(df_plot[vars_to_plot].min().min(), df_plot[vars_to_plot].max().max())

    
    plt.show()

# ---------------------------------------
# 4) Individual Plot Function
# ---------------------------------------
def discrete_color_scatter(
    df,
    x_col,
    y_col,
    cbar="HSR",
    bins=5,
    cmap_name="RdYlGn_r",
    cbar_label="HSR [mmHg/cm/s]",
    marker="o",
    edgecolor="k",
    alpha=0.8,
    figsize=(5,4),
    text_size=9,
    cbar_text_size=9,
    h_line=0,
    v_line=0,
    split_cfr_ffr_markers=False
):
    """
    Make a scatter plot of df[x_col] vs df[y_col], colored by df[cbar].
    If split_cfr_ffr_markers=True, each point's marker depends on:
      - CFR>2 vs <2
      - FFR>0.8 vs <0.8
    The legend markers are shown in a single color (black).
    """
    req_cols = [x_col, y_col, cbar]
    if split_cfr_ffr_markers:
        req_cols.extend(["CFR","FFR"])  # needed for classification
    
    df_plot_local = df.dropna(subset=req_cols).copy()
    if df_plot_local.empty:
        print(f"No valid data for {x_col}, {y_col}, {cbar}.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Discretize cbar
    cmin = df_plot_local[cbar].min()
    cmax = df_plot_local[cbar].max()
    boundaries = np.linspace(cmin, cmax, bins + 1)
    cmap = plt.get_cmap(cmap_name, bins)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

    if not split_cfr_ffr_markers:
        # One scatter of all data
        sc = ax.scatter(
            df_plot_local[x_col],
            df_plot_local[y_col],
            c=df_plot_local[cbar],
            cmap=cmap,
            norm=norm,
            marker=marker,
            edgecolors=edgecolor,
            alpha=alpha
        )
    else:
        # (A) Plot color-coded markers but suppress auto-legend
        cat_definitions = {
            "cat1": {
                "mask": (df_plot_local['CFR'] > 2) & (df_plot_local['FFR'] > 0.8),
                "marker": "o",
                "label": "CFR>2, FFR>0.8"
            },
            "cat2": {
                "mask": (df_plot_local['CFR'] > 2) & (df_plot_local['FFR'] < 0.8),
                "marker": "o",
                "label": "CFR>2, FFR<0.8"
            },
            "cat3": {
                "mask": (df_plot_local['CFR'] < 2) & (df_plot_local['FFR'] > 0.8),
                "marker": "v",
                "label": "CFR<2, FFR>0.8"
            },
            "cat4": {
                "mask": (df_plot_local['CFR'] < 2) & (df_plot_local['FFR'] < 0.8),
                "marker": "s",
                "label": "CFR<2, FFR<0.8"
            },
        }
        sc = None
        cat_handles = []  # for a custom legend

        for cat_key, cat_info in cat_definitions.items():
            mask = cat_info["mask"]
            subset = df_plot_local[mask]
            if subset.empty:
                continue

            # Plot points with color scale
            sc = ax.scatter(
                subset[x_col],
                subset[y_col],
                c=subset[cbar],
                cmap=cmap,
                norm=norm,
                marker=cat_info["marker"],
                edgecolors=edgecolor,
                alpha=alpha,
                label=None  # suppress auto-legend
            )

            # (B) Create a black-marker legend handle
            handle = mlines.Line2D(
                [],
                [],
                marker=cat_info["marker"],
                color="black",         # edgecolor
                markerfacecolor="black",
                markeredgecolor="black",
                linestyle='None',
                label=cat_info["label"]
            )
            cat_handles.append(handle)

        # Show the custom legend with black markers
        ax.legend(handles=cat_handles, fontsize=text_size, frameon=True)

    ax.set_xlabel(x_col, fontsize=text_size)
    ax.set_ylabel(y_col, fontsize=text_size)

    # Optional lines
    if h_line:
        ax.axhline(h_line, color='gray', linestyle='dotted', linewidth=1)
    if v_line:
        ax.axvline(v_line, color='gray', linestyle='dotted', linewidth=1)

    # Discrete colorbar
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, spacing="proportional")
        cb.set_label(cbar_label, fontsize=cbar_text_size)
        cb.set_ticks(boundaries)
        cb.set_ticklabels([f"{v:.1f}" for v in boundaries])
        cb.ax.tick_params(labelsize=cbar_text_size)

    plt.tight_layout()
    plt.show()

# ---------------------------------------
# 5) Example Usage: Double For Loop
# ---------------------------------------
# ys = ['CFR','CFR/FFR', 'FFR']
# cols = ['HSR','HMR','BMR/HMR']

# for y in ys:
#     for col in cols:
#         discrete_color_scatter(
#             df_plot,
#             x_col="WSS_Bif_Area_min",
#             y_col=y,
#             cbar=col,
#             cbar_label=f"{col}{' [mmHg/cm/s]' if col in ['HSR','HMR'] else ''}",
#             h_line = 2 if y in ['CFR', 'CFR/FFR'] else (0.8 if y == 'FFR' else 0),
#             figsize=(6,4),
#             split_cfr_ffr_markers=True
#         )
