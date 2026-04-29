#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal scatter‑plot helper (no grouping) + sample‑count print
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# ── Matplotlib defaults ──────────────────────────────
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 8, 'figure.dpi': 600
})

# ── 1) READ & prep data ──────────────────────────────
df = pd.read_csv('../../data/data.csv')

numeric_cols = [
    'CFR', 'P_d/P_a', 'BMR/HMR', 'R_total',
    'Stenosis Percentage', 'Length', 'HMR', 'HSR', 'P_Loss_Coeff'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

if 'BMR' not in df.columns:
    df['BMR'] = df['BMR/HMR'] * df['HMR']

df['iFR'] = np.where(df['Condition'] == 'Non-hyperemic', df['P_d/P_a'], np.nan)
df['FFR'] = np.where(df['Condition'] == 'Hyperemic',    df['P_d/P_a'], np.nan)


def stenosis_group(val, decimal_places=2, tolerance=0.02):
    """Round to `decimal_places`; keep value only if within ±tolerance."""
    if pd.isna(val):
        return np.nan
    rounded = round(val, decimal_places)
    return rounded if abs(val - rounded) <= tolerance else np.nan

if 'Stenosis Group' not in df.columns:
    df['Stenosis Group'] = df['Stenosis Percentage'].apply(stenosis_group)

def make_smart_scatter(
    data, x_col, y_col, *,
    color_col=None,
    x_label='', y_label='', title='',
    cmap_name='RdYlGn',
    custom_boundaries=None,
    color_label='',
    group=True,
    connect_groups=False,
    show_singletons=True,        # ── NEW FLAG ──
    add_threshold=None,
    alpha_scatter=0.7,
    s_scatter=60,
    scatter_color='#5E9096'
):
    """
    Scatter with optional colour‑map and grouping.

    Parameters
    ----------
    show_singletons : bool, default True
        • True  – plot groups that have only one point
        • False – skip those groups (they won’t appear on the figure)
    """

    # ── marker / linestyle dictionaries (per current data) ────────────
    sten_vals = sorted(data['Stenosis Group'].dropna().unique())
    len_vals  = sorted(data['Length'].dropna().unique())
    markers_by_sten     = {s: 'o' for s in sten_vals}
    linestyle_by_length = {l: 'solid' for l in len_vals}

    # ── column subset & NaN drop ──────────────────────────────────────
    keep_cols = [x_col, y_col] + ([color_col] if color_col else [])
    if group:
        keep_cols += ['Stenosis Group', 'Length']
    df_plot = data[keep_cols].dropna()

    # if group=True & show_singletons=False  → drop groups of size 1
    if group and not show_singletons:
        counts = df_plot.groupby(['Stenosis Group', 'Length']).size()
        valid_groups = counts[counts > 1].index
        df_plot = df_plot.set_index(['Stenosis Group', 'Length']
                     ).loc[valid_groups].reset_index()

    n_samples = len(df_plot)
    print(f"{title or f'{y_col} vs {x_col}'}: {n_samples} points")
    if n_samples == 0:
        print("Nothing to plot – aborting.")
        return

    # ── colour map (if requested) ─────────────────────────────────────
    if color_col:
        if custom_boundaries is None:
            custom_boundaries = np.linspace(df_plot[color_col].min(),
                                            df_plot[color_col].max(), 6)
        norm = mcolors.BoundaryNorm(custom_boundaries, 256)
        cmap = plt.get_cmap(cmap_name)

    # ── plotting ──────────────────────────────────────────────────────
    plt.figure(figsize=(6, 4))
    first_scatter = None

    if group:
        for (sten_val, length_val), gdf in df_plot.groupby(
                ['Stenosis Group', 'Length']):
            marker = markers_by_sten.get(sten_val, 'o')
            ls     = linestyle_by_length.get(length_val, 'solid')

            # scatter points
            if color_col:
                sc = plt.scatter(gdf[x_col], gdf[y_col],
                                 c=gdf[color_col], cmap=cmap, norm=norm,
                                 edgecolor='k', alpha=alpha_scatter,
                                 s=s_scatter, marker=marker)
                if first_scatter is None:
                    first_scatter = sc
            else:
                plt.scatter(gdf[x_col], gdf[y_col],
                            color=scatter_color, edgecolor='k',
                            alpha=alpha_scatter, s=s_scatter, marker=marker)

            # optional connecting line
            if connect_groups and len(gdf) > 1:
                gdf_sorted = gdf.sort_values(x_col)
                plt.plot(gdf_sorted[x_col], gdf_sorted[y_col],
                         linestyle=ls, color='gray', alpha=0.5)

    else:  # not grouped
        if color_col:
            first_scatter = plt.scatter(df_plot[x_col], df_plot[y_col],
                                        c=df_plot[color_col], cmap=cmap,
                                        norm=norm, edgecolor='k',
                                        alpha=alpha_scatter, s=s_scatter)
        else:
            plt.scatter(df_plot[x_col], df_plot[y_col],
                        color=scatter_color, edgecolor='k',
                        alpha=alpha_scatter, s=s_scatter)

    # ── colour‑bar & threshold lines ─────────────────────────────────
    if color_col and first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label)

    if add_threshold:
        for th in add_threshold:
            axis = th.get('axis', 'y')
            val  = th.get('value', 0.0)
            style = th.get('style', '--')
            col   = th.get('color', 'gray')
            w     = th.get('width', 0.8)
            if axis == 'y':
                plt.axhline(val, color=col, linestyle=style, linewidth=w)
            if axis == 'x':
                plt.axvline(val, color=col, linestyle=style, linewidth=w)

    plt.xlabel(x_label); plt.ylabel(y_label); plt.title(title)
    plt.grid(False); plt.tight_layout()
    # plt.legend()  # enable if you want a legend
    plt.show()



# ── 3) examples ─────────────────────────────────────
# CFR vs HMR (no colour‑bar)
# make_smart_scatter(
#     data=df[(df['CFR'].notna()) & (df['HMR'].notna()) & (df['source'] == 'mine')],
#     x_col='HMR', y_col='CFR',
#     x_label='HMR [mmHg/cm/s]', y_label='CFR',
#     add_threshold=[{'axis': 'y', 'value': 2.0}]
# )
#
# # FFR vs HMR, limited to CFR<1.5 (still no colour‑bar)
# make_smart_scatter(
#     data=df[(df['FFR'].notna()) & (df['HMR'].notna()) & (df['source'] == 'mine')],
#     x_col='HMR', y_col='FFR',
#     x_label='HMR [mmHg/cm/s]', y_label='FFR',
#     add_threshold=[{'axis': 'y', 'value': 0.8}]
# )
#
# # CFR vs FFR, limited to CFR<1.5 (still no colour‑bar)
# make_smart_scatter(
#     data=df[(df['FFR'].notna()) & (df['CFR'].notna()) & (df['source'] == 'mine')],
#     x_col='FFR', y_col='CFR',
#     x_label='FFR', y_label='CFR',
#     add_threshold=[{'axis': 'y', 'value': 2.0},
#                    {'axis': 'x', 'value': 0.8}]
# )

# make_smart_scatter(
#     data=df[(df['FFR'].notna()) & (df['CFR'].notna()) & (df['source'] == 'mine')],
#     x_col='FFR', y_col='CFR', #color_col='BMR/HMR',
#     x_label='FFR', y_label='CFR',
#     # title='CFR vs FFR',
#     group=True,
#     connect_groups=True,
#     # color_label='BMR/HMR',
#     add_threshold=[{'axis': 'y', 'value': 2.0},
#                    {'axis': 'x', 'value': 0.8}],
#     show_singletons=False
# )
#
# make_smart_scatter(
#     data=df[(df['CFR'].notna()) & (df['HMR'].notna()) & (df['source'] == 'mine')],
#     x_col='HMR', y_col='CFR', #color_col='BMR/HMR',
#     x_label='HMR [mmHg/cm/s]', y_label='CFR',
#     # title='CFR vs FFR',
#     group=True,
#     connect_groups=True,
#     # color_label='BMR/HMR',
#     add_threshold=[{'axis': 'y', 'value': 2.0}],
#     show_singletons=False
# )
#
# make_smart_scatter(
#     data=df[(df['FFR'].notna()) & (df['HMR'].notna()) & (df['source'] == 'mine')],
#     x_col='HMR', y_col='FFR', #color_col='BMR/HMR',
#     x_label='HMR [mmHg/cm/s]', y_label='FFR',
#     # title='CFR vs FFR',
#     group=True,
#     connect_groups=True,
#     # color_label='BMR/HMR',
#     add_threshold=[{'axis': 'y', 'value': 0.8}],
#     show_singletons=False
# )

make_smart_scatter(
    data=df[(df['FFR'].notna()) & (df['HMR'].notna()) & (df['R_micro'] == 0)],
    x_col='HMR', y_col='FFR', color_col='P_Loss_Coeff',
    x_label='HMR', y_label='FFR',
    # title='CFR vs FFR',
    group=True,
    connect_groups=True,
    color_label='$ζ_{L}$',
    add_threshold=[{'axis': 'y', 'value': 0.8}],
    show_singletons=False
)