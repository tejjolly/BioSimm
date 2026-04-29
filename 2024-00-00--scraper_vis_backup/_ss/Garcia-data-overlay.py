#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: ...
"""

import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt


# 1) LOAD DATA
data_file = "./summary2.csv"
df = pd.read_csv(data_file)

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
df['iFR'] = np.where(df['Condition'] == 'Non-hyperemic',
                          df['P_d/P_a'], np.nan)
df['FFR'] = np.where(df['Condition'] == 'Hyperemic',
                          df['P_d/P_a'], np.nan)

# OPTIONAL: define BMR if missing
if 'BMR' not in df.columns:
    df['BMR'] = df['BMR/HMR'] * df['HMR']

# 2) HELPER: separate out "mine" vs "garcia"
df_mine = df[df['source'] == 'mine']
df_garcia = df[df['source'] == 'garcia']

g_alpha = 0.35
g_size = 45
g_label = 'External Data'

############################################################################
# COMBINED PLOT FUNCTION (2x2 SUBPLOT)
############################################################################
def plot_four_subplots():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{data_file}", fontsize=16, fontweight='bold', y=.98)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # =========================================================================
    # PLOT 1: (Same as before)
    # HMR vs log10(P coeff), color by FFR
    # =========================================================================
    df_mine_p1 = df_mine[
        df_mine['P_Loss_Coeff'].notna() &
        df_mine['HMR'].notna() &
        df_mine['FFR'].notna()
    ]
    df_garcia_p1 = df_garcia[
        df_garcia['HMR'].notna() &
        df_garcia['P_Loss_Coeff'].notna() &
        df_garcia['FFR'].notna()
    ]

    # Define boundaries for FFR color scale
    ffr_min = min(df_mine_p1['FFR'].min(), df_garcia_p1['FFR'].min()) if not df_garcia_p1.empty else df_mine_p1['FFR'].min()
    ffr_max = max(df_mine_p1['FFR'].max(), df_garcia_p1['FFR'].max()) if not df_garcia_p1.empty else df_mine_p1['FFR'].max()
    boundaries_1 = np.linspace(ffr_min, ffr_max, 6)

    cmap1 = plt.get_cmap('RdYlGn')
    norm1 = colors.BoundaryNorm(boundaries_1, ncolors=256, clip=True)

    sc1 = ax1.scatter(
        df_mine_p1['HMR'], df_mine_p1['P_Loss_Coeff'],
        c=df_mine_p1['FFR'],
        cmap=cmap1, norm=norm1,
        edgecolor='k', alpha=1.0, s=60
    )
    cbar1 = fig.colorbar(sc1, ax=ax1, ticks=boundaries_1)
    cbar1.set_label('FFR', fontsize=12)
    ax1.set_xlabel('HMR [mmHg-s/cm]')
    ax1.set_ylabel('log₁₀(Pressure Loss Coeff.)')
    ax1.set_title('log₁₀(P coeff) vs HMR, Colored by FFR')
    ax1.set_xlim(0.25, 4.5)

    # if not df_garcia_p1.empty:
    #     ax1.scatter(
    #         df_garcia_p1['HMR'], df_garcia_p1['P_Loss_Coeff'],
    #         alpha=0.5, c=df_garcia_p1['FFR'],
    #         cmap=cmap1, norm=norm1,
    #         edgecolors='none', s=g_size, label=g_label
    #     )
    #     ax1.legend(loc='best')

    # =========================================================================
    # PLOT 2: FFR vs log10(P coeff), colored by HMR (including Garcia)
    # =========================================================================
    # We'll reuse the same subset from "Plot #1," but reorder axes:
    df_p2_mine = df_mine_p1.copy()
    df_p2_garc = df_garcia_p1.copy()

    # Boundaries for HMR color scale
    hmr_min = min(df_p2_mine['HMR'].min(), df_p2_garc['HMR'].min()) if not df_p2_garc.empty else df_p2_mine['HMR'].min()
    hmr_max = max(df_p2_mine['HMR'].max(), df_p2_garc['HMR'].max()) if not df_p2_garc.empty else df_p2_mine['HMR'].max()
    boundaries_2 = np.linspace(hmr_min, hmr_max, 6)

    cmap2 = plt.get_cmap('RdYlGn_r')
    norm2 = colors.BoundaryNorm(boundaries_2, ncolors=256, clip=True)

    sc2 = ax2.scatter(
        df_p2_mine['P_Loss_Coeff'], df_p2_mine['FFR'],
        c=df_p2_mine['HMR'],
        cmap=cmap2, norm=norm2,
        edgecolor='k', alpha=0.8, s=60
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, ticks=boundaries_2)
    cbar2.set_label('HMR [mmHg-s/cm]', fontsize=12)
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('log₁₀(Pressure Loss Coeff.)')
    ax2.set_ylabel('FFR')
    ax2.set_title('FFR vs log₁₀(P coeff), Colored by HMR')

    # if not df_p2_garc.empty:
    #     ax2.scatter(
    #         df_p2_garc['P_Loss_Coeff'], df_p2_garc['FFR'],
    #         alpha=g_alpha,
    #         c=df_p2_garc['HMR'],  # color Garcia by HMR
    #         cmap=cmap2, norm=norm2,
    #         edgecolors='none', s=g_size, label=g_label
    #     )
    #     ax2.legend(loc='best')

    # =========================================================================
    # PLOT 3: HMR vs FFR, colored by log10(P_Loss_Coeff)  (replacing HSR usage)
    # =========================================================================
    # We'll gather data that has HMR, FFR, and P_Loss_Coeff
    df_mine_p3 = df_mine[
        df_mine['HMR'].notna() &
        df_mine['FFR'].notna() &
        df_mine['P_Loss_Coeff'].notna()
    ]
    df_garcia_p3 = df_garcia[
        df_garcia['HMR'].notna() &
        df_garcia['FFR'].notna() &
        df_garcia['P_Loss_Coeff'].notna()
    ]

    # Determine color boundaries from combined set
    all_3 = pd.concat([df_mine_p3, df_garcia_p3], ignore_index=True)
    if not all_3.empty:
        pl_min = all_3['P_Loss_Coeff'].min()
        pl_max = all_3['P_Loss_Coeff'].max()
        boundaries_3 = np.linspace(pl_min, pl_max, 6)
    else:
        boundaries_3 = [0, 1, 2, 3, 4, 5]

    cmap3 = plt.get_cmap('RdYlGn_r')
    norm3 = colors.BoundaryNorm(boundaries_3, ncolors=256, clip=True)

    sc3 = ax3.scatter(
        df_mine_p3['HMR'], df_mine_p3['FFR'],
        c=df_mine_p3['P_Loss_Coeff'],
        cmap=cmap3, norm=norm3,
        edgecolor='k', alpha=0.7, s=60
    )
    cbar3 = fig.colorbar(sc3, ax=ax3, ticks=boundaries_3)
    cbar3.set_label('log₁₀(P_Loss_Coeff)', fontsize=12)
    ax3.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('HMR [mmHg/cm/s]')
    ax3.set_ylabel('FFR')
    ax3.set_title('FFR vs HMR, Colored by log₁₀(P coeff)')

    # if not df_garcia_p3.empty:
    #     ax3.scatter(
    #         df_garcia_p3['HMR'], df_garcia_p3['FFR'],
    #         alpha=g_alpha,
    #         c=df_garcia_p3['P_Loss_Coeff'],
    #         cmap=cmap3, norm=norm3,
    #         edgecolors='none', s=g_size, label=g_label
    #     )
    #     ax3.legend(loc='best')

    # =========================================================================
    # PLOT 4: CFR vs FFR, Colored by BMR/HMR (unchanged)
    # =========================================================================
    x_var = 'FFR'
    y_var = 'CFR'
    color_var = 'P_Loss_Coeff'
    df_p4_mine = df_mine[
        df_mine[x_var].notna() &
        df_mine[y_var].notna() &
        df_mine[color_var].notna()
    ]
    df_p4_garc = df_garcia[
        df_garcia[x_var].notna() &
        df_garcia[y_var].notna()
    ]
    # Boundaries for color_var from combined
    both_4 = pd.concat([df_p4_mine, df_p4_garc], ignore_index=True)
    if not both_4.empty:
        bmr_min = both_4[color_var].min()
        bmr_max = both_4[color_var].max()
        boundaries_4 = np.linspace(bmr_min, bmr_max, 6)
    else:
        boundaries_4 = [1, 3.5, 6, 9, 12, 15]

    cmap4 = plt.get_cmap('RdYlGn')
    norm4 = colors.BoundaryNorm(boundaries_4, ncolors=256, clip=True)

    sc4 = ax4.scatter(
        df_p4_mine[x_var], df_p4_mine[y_var],
        c=df_p4_mine[color_var],
        cmap=cmap4, norm=norm4,
        edgecolor='k', alpha=1.0, s=60
    )
    cbar4 = fig.colorbar(sc4, ax=ax4, ticks=boundaries_4)
    cbar4.set_label(color_var, fontsize=12)
    ax4.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.8)
    ax4.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('FFR')
    ax4.set_ylabel(y_var)
    ax4.set_title(f'{y_var} vs FFR, Colored by {color_var}')

    if not df_p4_garc.empty:
        ax4.scatter(
            df_p4_garc[x_var], df_p4_garc[y_var],
            alpha=0.5, c=df_p4_garc[color_var],
            cmap=cmap4, norm=norm4,
            edgecolors='none', s=g_size, label=g_label
        )
        ax4.legend(loc='best')

    plt.tight_layout()
    plt.show()

# Call the 4-plot function (only)
plot_four_subplots()
