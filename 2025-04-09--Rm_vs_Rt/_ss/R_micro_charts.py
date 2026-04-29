#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: your_name
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# -----------------------------
# Global matplotlib settings
# -----------------------------
plt.rcParams.update({
    'font.size': 12,  # Base font size
    'axes.labelsize': 14,  # Axis label font size
    'axes.titlesize': 14,  # Title font size
    'xtick.labelsize': 12,  # X-tick label font size
    'ytick.labelsize': 12,  # Y-tick label font size
    'legend.fontsize': 8,  # Legend font size
    'figure.dpi': 600  # Higher DPI for crisp text
})

# -----------------------------
# 1) READ & PREPARE DATA
# -----------------------------
summary_file = './summary2.csv'
df = pd.read_csv(summary_file)

# Convert columns to numeric as needed
for col in ['R_micro', 'R_total', 'WSS_TE_min',
            'Stenosis Percentage', 'Length']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['Stenosis Percentage'] = df['Stenosis Percentage'].round(2)



# 1a) Identify (Stenosis, Length) combos where R_micro != 0
df_nonzero = df[(df['R_micro'].notna())
                & (df['R_micro'] != 0)
                & (df['Stenosis Percentage'] > 0.01)
                # & (df['Stenosis Percentage'] < 0.50)
]
valid_geoms = set(zip(df_nonzero['Stenosis Percentage'], df_nonzero['Length']))

# 1b) Keep *only* those combos (so we can plot old vs. new together)
# df_sub = df[df.apply(
#     lambda row: (row['Stenosis Percentage'], row['Length']) in valid_geoms,
#     axis=1
# )].copy()

df_sub = df[df.apply(
    lambda row: (row['Stenosis Percentage'], row['Length']) in valid_geoms and row['Condition'] == 'Hyperemic',
    axis=1
)].copy()


# Drop rows missing R_total or WSS_TE_min
df_sub = df_sub[df_sub['R_total'].notna() & df_sub['WSS_TE_min'].notna() & df_sub['R_micro'].notna()]

# -----------------------------
# 2) BUILD DISCRETE COLORMAP
#    (here: color by R_micro)
# -----------------------------
rmin = df_sub['R_micro'].min()
rmax = df_sub['R_micro'].max()
# Create 5 intervals + 1 boundary = 6
boundaries = np.linspace(rmin, rmax, 6)
cmap = plt.get_cmap('RdYlGn')
norm = colors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

# -----------------------------
# 3) PLOT: WSS_TE_min vs R_total
# -----------------------------
plt.figure(figsize=(6, 4))

# Group by geometry and connect points
df_sub['R_micro_nonzero'] = df_sub['R_micro'] != 0

# Locate the special point (R_micro = 0, R_total ≈ 0.24)
special_point = df_sub[(df_sub['R_micro'] == 0) & (np.isclose(df_sub['R_total'], 0.24))]

if not special_point.empty:
    # Duplicate it and fake R_micro_nonzero = True
    duplicate = special_point.copy()
    duplicate['R_micro_nonzero'] = True  # This forces it into the "nonzero" group

    # Append the duplicate to df_sub
    df_sub = pd.concat([df_sub, duplicate], ignore_index=True)

groups = df_sub.groupby(['Stenosis Percentage', 'Length', 'R_micro_nonzero'])

for (sten_val, len_val, is_nonzero), gdf in groups:
    gdf_sorted = gdf.sort_values(by='R_total')

    plt.scatter(
        gdf_sorted['R_total'],
        gdf_sorted['WSS_TE_min'],
        c=gdf_sorted['R_micro'],
        cmap=cmap,
        norm=norm,
        edgecolor='k',
        alpha=0.8,
        s=60
    )

    if len(gdf_sorted) > 1:
        plt.plot(
            gdf_sorted['R_total'],
            gdf_sorted['WSS_TE_min'],
            color='gray' if is_nonzero else 'lightgray',
            linestyle='-' if is_nonzero else '--',
            alpha=0.5
        )

    if is_nonzero:  # Only annotate the nonzero ones (optional)
        mid_x = gdf_sorted['R_total'].mean()
        mid_y = gdf_sorted['WSS_TE_min'].mean()
        plt.text(
            mid_x+0.025, mid_y,
            f"S:{sten_val:.2f}%, L={len_val}",
            color='gray', fontsize=9, ha='left'
        )


# 3a) Create an invisible reference scatter for colorbar
scatter_for_cbar = plt.scatter([], [], c=[], cmap=cmap, norm=norm)
cbar = plt.colorbar(scatter_for_cbar, ticks=boundaries)
cbar.set_label('R_micro', fontsize=12)

# 3b) Axis labels, title, etc.
plt.xlabel('R_total')
plt.ylabel('WSS_TE_min')
plt.title('WSS_TE_min vs R_total (colored by R_micro)')
plt.grid(False)
plt.tight_layout()
plt.show()
