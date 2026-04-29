#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: ...
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# 1) LOAD DATA
df = pd.read_csv("../data/data.csv")

# OPTIONAL: define BMR if missing
if 'BMR' not in df.columns:
    df['BMR'] = df['BMR/HMR'] * df['HMR']

# 2) HELPER: separate out "mine" vs "garcia"
df_mine = df[df['source'] == 'mine']
df_garcia = df[df['source'] == 'garcia']

g_alpha = 0.35
g_size = 45
g_label = 'External Data'

# 3) A "no-show" scatter so we can overlay points afterward
def make_smart_scatter_no_show(
    data, x_col, y_col, color_col,
    x_label, y_label, title,
    cmap_name='RdYlGn',
    custom_boundaries=None,
    color_label='',
    add_threshold=None,
    alpha_scatter=0.7,
    s_scatter=60
):
    """
    Identical to your original function, except it doesn't call plt.show().
    It still creates a new figure each time.
    """
    plt.figure(figsize=(6, 4))

    # Filter out rows missing x, y, or color
    df_plot = data.copy()
    df_plot = df_plot[
        df_plot[x_col].notna() &
        df_plot[y_col].notna() &
        df_plot[color_col].notna()
    ]

    if df_plot.empty:
        print(f"[make_smart_scatter_no_show] No data to plot with {x_col}, {y_col}, {color_col}")

    # If needed, define boundaries from data min/max
    if custom_boundaries is None and not df_plot.empty:
        cmin = df_plot[color_col].min()
        cmax = df_plot[color_col].max()
        custom_boundaries = np.linspace(cmin, cmax, 6)

    norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
    cmap = cm.get_cmap(cmap_name)

    sc = None
    if not df_plot.empty:
        sc = plt.scatter(
            df_plot[x_col],
            df_plot[y_col],
            c=df_plot[color_col],
            cmap=cmap,
            norm=norm,
            edgecolor='k',
            alpha=alpha_scatter,
            s=s_scatter,
        )

    # Add colorbar
    if sc is not None:
        cbar = plt.colorbar(sc, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label, fontsize=12)

    # Add threshold lines if desired
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get('axis', 'y')
            value = tdict.get('value', 0.0)
            style = tdict.get('style', '--')
            c = tdict.get('color', 'gray')
            w = tdict.get('width', 0.8)
            if axis_type == 'y':
                plt.axhline(y=value, color=c, linestyle=style, linewidth=w)
            else:
                plt.axvline(x=value, color=c, linestyle=style, linewidth=w)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    # No plt.show() here

def make_smart_scatter(
    data, x_col, y_col, color_col,
    x_label, y_label, title,
    cmap_name='RdYlGn',
    custom_boundaries=None,
    color_label='',
    add_threshold=None,
    alpha_scatter=0.7,
    s_scatter=60
):
    """This version calls show() at the end."""
    make_smart_scatter_no_show(
        data, x_col, y_col, color_col,
        x_label, y_label, title,
        cmap_name, custom_boundaries,
        color_label, add_threshold,
        alpha_scatter, s_scatter
    )
    plt.show()

# ---------------------------------------------------
# PLOT 1) CFR vs FFR, colored by BMR/HMR + Garcia overlay in triangles w/ color
# ---------------------------------------------------
# df_mine_plot1 = df_mine[
#     df_mine['P_d/P_a'].notna() &
#     df_mine['CFR'].notna() &
#     df_mine['BMR/HMR'].notna()
# ]
# df_garcia_plot1 = df_garcia[
#     df_garcia['P_d/P_a'].notna() &
#     df_garcia['CFR'].notna()
# ]
#
# boundaries_1 = np.linspace(1, 3.5, 6)
#
# make_smart_scatter_no_show(
#     data=df_mine_plot1,
#     x_col='P_d/P_a',
#     y_col='CFR',
#     color_col='BMR/HMR',
#     x_label='FFR',
#     y_label='CFR',
#     title='CFR vs FFR, Colored by BMR/HMR',
#     cmap_name='RdYlGn',
#     custom_boundaries=boundaries_1,
#     color_label='BMR/HMR',
#     alpha_scatter=1.0,
#     add_threshold=[
#         {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
#         {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
#     ]
# )
# if not df_garcia_plot1.empty:
#     # Overlaid in color, marker='^' for triangles
#     plt.scatter(
#         df_garcia_plot1['P_d/P_a'],
#         df_garcia_plot1['CFR'],
#         alpha=0.5,
#         c=df_garcia_plot1['BMR/HMR'],
#         cmap=cm.get_cmap('RdYlGn'),
#         norm=colors.BoundaryNorm(boundaries_1, ncolors=256, clip=True),
#         edgecolors='none',
#         s=g_size,
#         label=g_label,
#         # marker='^'
#     )
#     plt.legend(loc='best')
# plt.show()
df_mine_plot1 = df_mine[
    df_mine['P_Loss_Coeff'].notna() &
    df_mine['HMR'].notna() &
    df_mine['P_d/P_a'].notna()
]
df_garcia_plot1 = df_garcia[
    df_garcia['HMR'].notna() &
    df_garcia['P_Loss_Coeff'].notna()
]

boundaries_1 = np.linspace(.65, .9, 6)

make_smart_scatter_no_show(
    data=df_mine_plot1,
    x_col='HMR',
    y_col='P_Loss_Coeff',
    color_col='P_d/P_a',
    x_label='HMR [mmHg-s/cm]',
    y_label='log_10(Press. Coeff.)',
    title='log_10(Press. Coeff.) vs HMR, Colored by FFR',
    cmap_name='RdYlGn',
    custom_boundaries=boundaries_1,
    color_label='FFR',
    alpha_scatter=1.0,
    add_threshold=[
        {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
        {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
    ]
)
if not df_garcia_plot1.empty:
    # Overlaid in color, marker='^' for triangles
    plt.scatter(
        df_garcia_plot1['HMR'],
        df_garcia_plot1['P_Loss_Coeff'],
        alpha=0.5,
        c=df_garcia_plot1['P_d/P_a'],
        cmap=cm.get_cmap('RdYlGn'),
        norm=colors.BoundaryNorm(boundaries_1, ncolors=256, clip=True),
        edgecolors='none',
        s=g_size,
        label=g_label,
        # marker='^'
    )
    plt.legend(loc='best')
plt.show()
# ---------------------------------------------------
# PLOT 2) FFR vs HMR, colored by HSR, Garcia in gray circle
# ---------------------------------------------------
df_mine_plot2 = df_mine[
    df_mine['HMR'].notna() &
    df_mine['P_d/P_a'].notna() &
    df_mine['HSR'].notna()
]
df_garcia_plot2 = df_garcia[
    df_garcia['HMR'].notna() &
    df_garcia['P_d/P_a'].notna()
]

boundaries_2 = np.linspace(0.2, 1.2, 6)
make_smart_scatter_no_show(
    data=df_mine_plot2,
    x_col='HMR',
    y_col='P_d/P_a',
    color_col='HSR',
    x_label='HMR [mmHg/cm/s]',
    y_label='FFR',
    title='FFR vs HMR, Colored by HSR',
    cmap_name='RdYlGn_r',
    custom_boundaries=boundaries_2,
    color_label='HSR [mmHg/cm/s]',
    add_threshold=[{'axis': 'y', 'value': 0.8}]
)
if not df_garcia_plot2.empty:
    plt.scatter(
        df_garcia_plot2['HMR'],
        df_garcia_plot2['P_d/P_a'],
        alpha=g_alpha,
        color='gray',
        edgecolors='none',
        s=g_size,
        label=g_label
    )
    plt.legend(loc='best')
plt.show()

# ---------------------------------------------------
# PLOT 3) CFR vs HMR, colored by HSR, Garcia in gray circle
# ---------------------------------------------------
df_mine_plot3 = df_mine[
    df_mine['CFR'].notna() &
    df_mine['HMR'].notna() &
    df_mine['HSR'].notna()
]
df_garcia_plot3 = df_garcia[
    df_garcia['CFR'].notna() &
    df_garcia['HMR'].notna()
]

boundaries_3 = np.linspace(0.1, 1.1, 6)
make_smart_scatter_no_show(
    data=df_mine_plot3,
    x_col='HMR',
    y_col='CFR',
    color_col='HSR',
    x_label='HMR [mmHg/cm/s]',
    y_label='CFR',
    title='CFR vs HMR, Colored by HSR',
    cmap_name='RdYlGn_r',
    custom_boundaries=boundaries_3,
    color_label='HSR [mmHg/cm/s]',
    add_threshold=[{'axis': 'y', 'value': 2.0}]
)
if not df_garcia_plot3.empty:
    plt.scatter(
        df_garcia_plot3['HMR'],
        df_garcia_plot3['CFR'],
        alpha=g_alpha,
        color='gray',
        edgecolors='none',
        s=g_size,
        label=g_label
    )
    plt.legend(loc='best')
plt.show()

# ---------------------------------------------------
# PLOT 4) CFR/FFR vs HMR, colored by HSR, Garcia in gray circle
# ---------------------------------------------------
df_mine_plot4 = df_mine[
    df_mine['CFR/FFR'].notna() &
    df_mine['HMR'].notna() &
    df_mine['HSR'].notna()
]
df_garcia_plot4 = df_garcia[
    df_garcia['CFR/FFR'].notna() &
    df_garcia['HMR'].notna()
]

boundaries_4 = np.linspace(0.1, 1.1, 6)
make_smart_scatter_no_show(
    data=df_mine_plot4,
    x_col='HMR',
    y_col='CFR/FFR',
    color_col='HSR',
    x_label='HMR [mmHg/cm/s]',
    y_label='CFR/FFR',
    title='CFR/FFR vs HMR, Colored by HSR',
    cmap_name='RdYlGn_r',
    custom_boundaries=boundaries_4,
    color_label='HSR [mmHg/cm/s]',
    add_threshold=[{'axis': 'y', 'value': 2.0}]
)
if not df_garcia_plot4.empty:
    plt.scatter(
        df_garcia_plot4['HMR'],
        df_garcia_plot4['CFR/FFR'],
        alpha=g_alpha,
        color='gray',
        edgecolors='none',
        s=g_size,
        label=g_label
    )
    plt.legend(loc='best')
plt.show()

############################################################################
#  FUNCTION TO PUT THE FOUR PLOTS INTO ONE 2x2 SUBPLOT FIGURE
############################################################################
def plot_four_subplots():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ### --- PLOT 1: HMR vs log10(P coeff), color by FFR --- ###
    df_p1 = df_mine_plot1
    df_g = df_garcia_plot1
    boundaries_1 = np.linspace(0.65, 0.9, 6)
    cmap1 = cm.get_cmap('RdYlGn')
    norm1 = colors.BoundaryNorm(boundaries_1, ncolors=256, clip=True)

    sc1 = ax1.scatter(
        df_p1['HMR'], df_p1['P_Loss_Coeff'],
        c=df_p1['P_d/P_a'],
        cmap=cmap1, norm=norm1, edgecolor='k',
        alpha=1.0, s=60
    )
    cbar1 = fig.colorbar(sc1, ax=ax1, ticks=boundaries_1)
    cbar1.set_label('FFR', fontsize=12)
    ax1.set_xlabel('HMR [mmHg-s/cm]')
    ax1.set_ylabel('log₁₀(Pressure Loss Coeff.)')
    ax1.set_title('log₁₀(P coeff) vs HMR, Colored by FFR')
    # ax1.set_xlim(0.25, 4.5)

    if not df_g.empty:
        ax1.scatter(
            df_g['HMR'], df_g['P_Loss_Coeff'],
            alpha=0.5, c=df_g['P_d/P_a'],
            cmap=cmap1, norm=norm1,
            edgecolors='none', s=g_size, label=g_label
        )
        ax1.legend(loc='best')

    ### --- PLOT 2: log10(P coeff) vs FFR, color by HMR --- ###
    df_p2 = df_mine_plot1[df_mine_plot1['P_d/P_a'].notna()]
    df_g2 = df_garcia_plot1[df_garcia_plot1['P_d/P_a'].notna()]
    boundaries_2 = np.linspace(0.25, 4.5, 6)
    cmap2 = cm.get_cmap('RdYlGn_r')  # <--- switched to RdYlGn
    norm2 = colors.BoundaryNorm(boundaries_2, ncolors=256, clip=True)

    sc2 = ax2.scatter(
        df_p2['P_Loss_Coeff'], df_p2['P_d/P_a'],
        c=df_p2['HMR'],
        cmap=cmap2, norm=norm2,
        edgecolor='k', alpha=0.8, s=60
    )
    cbar2 = fig.colorbar(sc2, ax=ax2, ticks=boundaries_2)
    cbar2.set_label('HMR [mmHg-s/cm]', fontsize=12)
    ax2.set_xlabel('log₁₀(Pressure Loss Coeff.)')
    ax2.set_ylabel('FFR')
    ax2.set_title('FFR vs log₁₀(P coeff), Colored by HMR')
    ax2.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)

    if not df_g2.empty:
        ax2.scatter(
            df_g2['P_Loss_Coeff'], df_g2['P_d/P_a'],
            # alpha=g_alpha, color='gray', edgecolors='none', s=g_size, label=g_label

            alpha = 0.5, c = df_g2['HMR'],
            cmap = cmap2, norm = norm2,
            edgecolors = 'none', s = g_size, label = g_label
        )
        ax2.legend(loc='best')

    ### --- PLOT 3: HMR vs FFR, Colored by HSR --- ###
    df_p3 = df_mine_plot2
    df_g3 = df_garcia_plot2
    boundaries_3 = np.linspace(0.2, 1.2, 6)
    cmap3 = cm.get_cmap('RdYlGn_r')
    norm3 = colors.BoundaryNorm(boundaries_3, ncolors=256, clip=True)

    sc3 = ax3.scatter(
        df_p3['HMR'], df_p3['P_d/P_a'],
        c=df_p3['HSR'], cmap=cmap3, norm=norm3,
        edgecolor='k', alpha=0.7, s=60
    )
    cbar3 = fig.colorbar(sc3, ax=ax3, ticks=boundaries_3)
    cbar3.set_label('HSR [mmHg/cm/s]', fontsize=12)
    ax3.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_xlabel('HMR [mmHg/cm/s]')
    ax3.set_ylabel('FFR')
    ax3.set_title('FFR vs HMR, Colored by HSR')

    if not df_g3.empty:
        ax3.scatter(
            df_g3['HMR'], df_g3['P_d/P_a'],
            alpha=g_alpha, color='gray', edgecolors='none', s=g_size, label=g_label
        )
        ax3.legend(loc='best')

    ### --- PLOT 4: CFR vs FFR, Colored by BMR/HMR --- ###
    df_p4 = df_mine[
        df_mine['P_d/P_a'].notna() &
        df_mine['CFR'].notna() &
        df_mine['BMR/HMR'].notna()
    ]
    df_g4 = df_garcia[
        df_garcia['P_d/P_a'].notna() &
        df_garcia['CFR'].notna()
    ]
    boundaries_4 = np.linspace(1, 3.5, 6)
    cmap4 = cm.get_cmap('RdYlGn')
    norm4 = colors.BoundaryNorm(boundaries_4, ncolors=256, clip=True)

    sc4 = ax4.scatter(
        df_p4['P_d/P_a'], df_p4['CFR'],
        c=df_p4['BMR/HMR'],
        cmap=cmap4, norm=norm4,
        edgecolor='k', alpha=1.0, s=60
    )
    cbar4 = fig.colorbar(sc4, ax=ax4, ticks=boundaries_4)
    cbar4.set_label('BMR/HMR', fontsize=12)
    ax4.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.8)
    ax4.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.8)
    ax4.set_xlabel('FFR')
    ax4.set_ylabel('CFR')
    ax4.set_title('CFR vs FFR, Colored by BMR/HMR')

    if not df_g4.empty:
        ax4.scatter(
            df_g4['P_d/P_a'], df_g4['CFR'],
            alpha=0.5, c=df_g4['BMR/HMR'],
            cmap=cmap4, norm=norm4,
            edgecolors='none', s=g_size, label=g_label
        )
        ax4.legend(loc='best')

    plt.tight_layout()
    plt.show()

############################################################################
#  If you want a single 2x2 figure with these four plots, just call:
plot_four_subplots()
############################################################################

# If you do NOT call the function, you'll still get your 4 separate plots
# from the code above.
