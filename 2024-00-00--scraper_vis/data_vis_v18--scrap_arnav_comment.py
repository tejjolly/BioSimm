#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:57:13 2025

@author: tejjolly
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.patheffects as pe

manuscript_data = True
log_switch = False
exclude_no_stenosis = False

sensitivies_analysis = False

plot1   = False # CFR vs FFR, col BMR/HMR
plot1_5 = True # CFR vs FFR, col HMR
plot2   = False # HMR vs HSR, col FFR
plot25  = False # HSR vs HMR, col Average Flow
plot3   = False # CFR vs FFR, col HMR
plot35  = False # FFR vs BMR/HMR, col HSR
plot36  = True  # FFR vs HMR, col HSR
plot37  = False # FFR vs Length, col HSR
plot4   = False # FFR vs HSR, col HMR
plot45  = False # FFR vs HSR, col BMR/HMR
plot5   = True  # CFR vs HMR, col HSR
plot55  = False # CFR vs BMR/HMR, col HSR
plot56  = False # CFR vs HSR, col HMR
plot57  = False # CFR vs HSR, col BMR/HMR
plot6   = False # CFR/FFR vs. BMR/HMR, col HSR
plot7   = False # 7-10 are 2-5 w/ P_Loss_Coeff instead of HSR
plot8   = False
plot9   = False
plot10  = False
plot11  = False # Area Low LE WSS x-x-x-x
plot12  = False # Area Low TE WSS x-x-x-x-x
plot13  = False # Area Low Bif WSS -x-x-x-x
plot14  = True  # Area High LE WSS x-x-x-x
plot15  = True  # Area High TE WSS x-x-x-x
plot16  = False # Area High Bif WSS x-x-x-x-x
plot17  = False # FFR vs. v_distal, colored by HSR
plot18  = False # Q_distal vs. HMR, colored by HSR
plot19  = True  # Q_distal vs. HSR, colored by HMR
plot20  = False # 2D response surface: FFR across (HMR, ζ)
plot21  = False # FFR vs HMR/(HMR+HSR)
plot22  = False # P_Loss_Coeff vs HMR, col FFR
plot23  = False # FFR vs P_Loss_Coeff, col HMR
plot24  = False # FFR vs HMR, col P_Loss_Coeff
plot26  = False # FFR vs Q_distal, col HSR

all_flag = False
LAD_flag = False
LCX_flag = True

HMR_boundaries = np.linspace(1, 7, 7)
HSR_boundaries = np.linspace(0.1, 1.1, 6)
Q_boundaries   = np.linspace(1, 7, 6)
v_boundaries   = np.linspace(10, 50, 5)

location_filter = (['LAD', 'LCX'] if all_flag or (LAD_flag and LCX_flag)
                   else 'LAD' if LAD_flag
                   else 'LCX' if LCX_flag
                   else None)

FIG4SIZE = (6.5,4)
FIG5SIZE = (7,5)
FIG6SIZE = (8,5)
FIG7SIZE = (8.5, 5)
FIG8SIZE = (7.5, 5)

FIG7CMAPFLOOR = 0.15
FIG7CMAPCEIL = 0.8
LAB_BLUE_CMAP = colors.LinearSegmentedColormap.from_list(
    'lab_blue', ['#E8EBF4', '#556092', '#171B2B']
)
LAB_TEAL_CMAP = colors.LinearSegmentedColormap.from_list(
    'lab_teal', ['#EAF5F5', '#7BB3B4', '#1D3234']
)
FIG7_PLOT23_CMAP = 'BuPu'      # HMR rule
FIG7_PLOT24_CMAP = 'Oranges'
FIG7_PLOT26_CMAP = 'Blues'     # reuse to keep 4 unique colormaps across 5 plots
FIG7ALPHA=1.0

EXTERNAL_STYLE = 'ext_style_1'  # ext_style_1 (old): filled markers + black outlines; ext_style_2: current hollow-overlay style

# -----------------------------
# Global matplotlib settings
# -----------------------------
original_settings = False
if original_settings:
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 8,
        'figure.dpi': 600
    })
else:
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.dpi': 600
    })

# -----------------------------
# 1) READ DATA
# -----------------------------
if manuscript_data:
    data_file = '../data/data_manuscript.csv'
else:
    data_file = '../data/data.csv'

df = pd.read_csv(data_file)

cols_to_num = [
    'CFR', 'P_d/P_a', 'BMR/HMR', 'R_total',
    'Stenosis Percentage', 'Length', 'HMR', 'HSR', 'P_Loss_Coeff'
]
for col in cols_to_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# OPTIONAL: define BMR = (BMR/HMR) * HMR
if 'BMR' not in df.columns and ('BMR/HMR' in df.columns) and ('HMR' in df.columns):
    df['BMR'] = df['BMR/HMR'] * df['HMR']

# Create new columns: iFR (non-hyperemic) and FFR (hyperemic)
if 'Condition' in df.columns and 'P_d/P_a' in df.columns:
    df['iFR'] = np.where(df['Condition'] == 'Non-hyperemic', df['P_d/P_a'], np.nan)
    df['FFR'] = np.where(df['Condition'] == 'Hyperemic', df['P_d/P_a'], np.nan)

# -----------------------------
# 2) HELPERS
# -----------------------------
def stenosis_group(val, decimal_places=2, tolerance=0.02):
    """
    Round `val` to `decimal_places` and group it if within ±tolerance of that rounded value.
    Returns NaN if out of range, so it won't be grouped.
    """
    rounded_val = round(val, decimal_places)
    if abs(val - rounded_val) <= tolerance:
        return rounded_val
    else:
        return np.nan

def snap_length_style(val, tol=0.15):
    """Return linestyle for a length value:
       1.2 cm → ':' , 2.5 cm → '--'."""
    if pd.isna(val):
        return 'solid'
    if np.isclose(val, 1.2, atol=tol):
        return ':'   # short dash
    if np.isclose(val, 2.5, atol=tol):
        return '--'  # long dash

def vessel_line_color(loc):
    """LAD→grey, LCX→grey, else gray."""
    if loc == 'LAD': return 'grey'
    if loc == 'LCX': return 'grey'
    return 'gray'

def stenosis_marker(sten_val):
    """0%→circle; ~45%→square; ~60%→triangle; else 'x'."""
    if pd.isna(sten_val): return 'x'
    if sten_val < 0.10: return 'o'
    if 0.40 <= sten_val <= 0.50: return 's'
    if 0.55 <= sten_val <= 0.65: return '^'
    return 'x'

def sanitize_filename(text):
    """Make a safe filename segment for savefig output."""
    return str(text).replace('/', '_').replace('\\', '_')

# Stenosis Group column (used for internal grouping/connecting)
if 'Stenosis Percentage' in df.columns:
    df['Stenosis Group'] = df['Stenosis Percentage'].apply(stenosis_group)
else:
    df['Stenosis Group'] = np.nan

# -----------------------------
# 3) MAIN PLOTTER
# -----------------------------
def make_smart_scatter(
        data, x_col, y_col,
        x_label, y_label, title,
        color_col=None,
        cmap_name='BuPu',
        cmap_floor=0.0,
        cmap_ceil=1.0,
        custom_boundaries=None,
        continuous_colorbar=False,
        color_label='',
        add_threshold=None,
        alpha_scatter=0.8,
        s_scatter=60,
        connect_stenosis_groups=False,
        show_singletons=True,
        savefig=False,
        dpi=600,
        dir='images',
        labels=False,
        location_col='Location',
        location_filter=None,
        figsize=(8.5, 5),
        y_lim=None,
        external_data_source=None,      # str or list/tuple/set of str; values in df[source_col]
        source_col='source',
        internal_source_name='mine',    # your default internal label
        external_alpha=None,
        external_style=EXTERNAL_STYLE,
        use_colormap=True,
        no_color_marker_color='black',
        show_monochrome_hmr_colorbar=False,
):
    plt.figure(figsize=figsize)

    df_plot = data.copy()

    # Optional filter by location
    if location_filter is not None and location_col in df_plot.columns:
        if isinstance(location_filter, (list, tuple, set)):
            df_plot = df_plot[df_plot[location_col].isin(location_filter)]
        else:
            df_plot = df_plot[df_plot[location_col] == location_filter]

    # Always require x/y
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna()]

    # Color behavior:
    # - use_colormap=True  => requires color_col and draws colorbar
    # - use_colormap=False => ignores color_col and uses one marker color for all points
    # - show_monochrome_hmr_colorbar=True (with use_colormap=False) => draws HMR-valued colorbar in one color
    has_color_dimension = use_colormap
    if has_color_dimension:
        if color_col is None:
            raise ValueError("use_colormap=True requires color_col to be provided.")
        if color_col not in df_plot.columns:
            raise ValueError(f"'{color_col}' is not a column in the data.")

    # We'll construct subsets first so we don't accidentally drop external points lacking stenosis.
    external_mode = external_data_source is not None
    if external_style not in {'ext_style_1', 'ext_style_2'}:
        raise ValueError("external_style must be one of: 'ext_style_1', 'ext_style_2'")
    if external_alpha is None:
        external_alpha = 0.25 if external_style == 'ext_style_1' else 1.0

    if external_mode:
        if source_col not in df_plot.columns:
            raise ValueError(
                f"external_data_source was provided, but '{source_col}' is not a column in the data."
            )

        if isinstance(external_data_source, str):
            ext_sources = [external_data_source]
        else:
            ext_sources = list(external_data_source)

        # Keep only internal + specified external
        df_plot = df_plot[df_plot[source_col].isin([internal_source_name] + ext_sources)]

        df_internal = df_plot[df_plot[source_col] == internal_source_name].copy()
        df_external = df_plot[df_plot[source_col].isin(ext_sources)].copy()

        if has_color_dimension:
            # Internal must have color for colormapped scatter
            df_internal = df_internal[df_internal[color_col].notna()]

            # External: allow missing color (plot as light gray)
            df_external_col   = df_external[df_external[color_col].notna()].copy()
            df_external_nocol = df_external[df_external[color_col].isna()].copy()

            # For norms/boundaries, prefer internal colored data; if none, fall back to external colored
            df_for_color = df_internal if len(df_internal) > 0 else df_external_col

    else:
        if has_color_dimension:
            # Original behavior: require color for everything
            df_plot = df_plot[df_plot[color_col].notna()]
            df_for_color = df_plot

    # Abort if nothing remains after required filtering
    if len(df_plot) == 0:
        print(f"{title or f'{y_col} vs {x_col}'}: 0 points")
        print("Nothing to plot – aborting.")
        plt.close()
        return
    if has_color_dimension and len(df_for_color) == 0:
        print(f"{title or f'{y_col} vs {x_col}'}: 0 points")
        print("Nothing to plot – aborting.")
        plt.close()
        return

    # Singleton filtering (apply only to the grouped/internal series)
    if not show_singletons:
        if external_mode:
            # Only internal gets grouped/connected; don't drop external for lacking stenosis/length
            if ('Stenosis Group' in df_internal.columns) and ('Length' in df_internal.columns):
                counts = df_internal.groupby(['Stenosis Group', 'Length']).size()
                valid_groups = counts[counts > 1].index
                df_internal = df_internal.set_index(['Stenosis Group', 'Length']).loc[valid_groups].reset_index()
        else:
            counts = df_plot.groupby(['Stenosis Group', 'Length']).size()
            valid_groups = counts[counts > 1].index
            df_plot = df_plot.set_index(['Stenosis Group', 'Length']).loc[valid_groups].reset_index()

    # Color scaling
    if has_color_dimension:
        if custom_boundaries is None:
            cmin = df_for_color[color_col].min()
            cmax = df_for_color[color_col].max()
            custom_boundaries = np.linspace(cmin, cmax, 6)

        if continuous_colorbar:
            norm = colors.Normalize(
                vmin=np.min(custom_boundaries),
                vmax=np.max(custom_boundaries),
                clip=True
            )
        else:
            norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
        use_cmap_floor = cmap_floor
        use_cmap_ceil = cmap_ceil
        if (
            isinstance(cmap_name, str)
            and cmap_name.lower() == 'bupu'
            # and cmap_floor == 0.0
            # and cmap_ceil == 1.0
        ):
            # Match the BuPu range used in distribution--stack_v4--new_colors--codex.py.
            use_cmap_floor = 0.25
            use_cmap_ceil = 1.0

        if not (0.0 <= use_cmap_floor < use_cmap_ceil <= 1.0):
            raise ValueError("cmap_floor and cmap_ceil must satisfy 0 <= floor < ceil <= 1.")

        base_cmap = plt.get_cmap(cmap_name)
        if (use_cmap_floor > 0.0) or (use_cmap_ceil < 1.0):
            cmap = colors.LinearSegmentedColormap.from_list(
                f"{getattr(base_cmap, 'name', 'cmap')}_{use_cmap_floor:.2f}_{use_cmap_ceil:.2f}",
                base_cmap(np.linspace(use_cmap_floor, use_cmap_ceil, 256))
            )
        else:
            cmap = base_cmap
    else:
        norm = None
        cmap = None

    # Debug: series listing (internal grouping only)
    if external_mode:
        print("\nEXTERNAL MODE ON")
        print(f"Internal points (source='{internal_source_name}'): {len(df_internal)}")
        if has_color_dimension:
            print(f"External points (sources={ext_sources}): {len(df_external)} "
                  f"[colored={len(df_external_col)}, no-color={len(df_external_nocol)}]")
        else:
            print(f"External points (sources={ext_sources}): {len(df_external)}")

        series_to_plot = (
            df_internal[['Stenosis Group', 'Length']]
            .dropna()
            .drop_duplicates()
            .sort_values(['Stenosis Group', 'Length'])
        )
        print("\n UNIQUE INTERNAL SERIES THAT WILL PLOT  "
              f"(total = {len(series_to_plot)})")
        print(series_to_plot.to_string(index=False))
    else:
        series_to_plot = (
            df_plot[['Stenosis Group', 'Length']]
            .dropna()
            .drop_duplicates()
            .sort_values(['Stenosis Group', 'Length'])
        )
        print("\n UNIQUE SERIES THAT WILL PLOT  "
              f"(total = {len(series_to_plot)})")
        print(series_to_plot.to_string(index=False))

    first_scatter = None

    # --- External overlay (no stenosis/length required) ---
    if external_mode:
        if has_color_dimension:
            if len(df_external_col) > 0:
                if external_style == 'ext_style_1':
                    plt.scatter(
                        df_external_col[x_col], df_external_col[y_col],
                        c=df_external_col[color_col], cmap=cmap, norm=norm,
                        edgecolors='black', linewidths=1.0,
                        alpha=external_alpha, s=s_scatter,
                        marker='o', zorder=1
                    )
                else:
                    ext_vals = df_external_col[color_col].to_numpy(float)
                    ext_edgecols = cmap(norm(ext_vals))
                    plt.scatter(
                        df_external_col[x_col], df_external_col[y_col],
                        facecolors='none',
                        edgecolors=ext_edgecols,
                        linewidths=1.3,
                        alpha=external_alpha, s=s_scatter,
                        marker='o', zorder=1
                    )
                # OPTIONAL: try dashed/dotted outlines
                # sc_ext.set_linestyle((0, (1, 1)))  # dotted-ish
                # sc_ext.set_linestyle((0, (4, 2)))  # dashed

            if len(df_external_nocol) > 0:
                if external_style == 'ext_style_1':
                    plt.scatter(
                        df_external_nocol[x_col], df_external_nocol[y_col],
                        color='lightgray',
                        edgecolors='black', linewidths=1.0,
                        alpha=external_alpha, s=s_scatter,
                        marker='o', zorder=1
                    )
                else:
                    plt.scatter(
                        df_external_nocol[x_col], df_external_nocol[y_col],
                        facecolors='none',
                        edgecolors='lightgray',
                        linewidths=1.3,
                        alpha=external_alpha, s=s_scatter,
                        marker='o', zorder=1
                    )
                # OPTIONAL: try dashed/dotted outlines
                # sc_ext_nocol.set_linestyle((0, (1, 1)))
                # sc_ext_nocol.set_linestyle((0, (4, 2)))
        else:
            if len(df_external) > 0:
                if external_style == 'ext_style_1':
                    plt.scatter(
                        df_external[x_col], df_external[y_col],
                        color=no_color_marker_color,
                        edgecolors='black', linewidths=1.0,
                        alpha=external_alpha, s=s_scatter,
                        marker='o', zorder=1
                    )
                else:
                    plt.scatter(
                        df_external[x_col], df_external[y_col],
                        facecolors='none',
                        edgecolors=no_color_marker_color,
                        linewidths=1.3,
                        alpha=external_alpha, s=s_scatter,
                        marker='o', zorder=1
                    )

        # Internal series grouping/connecting
        if (('Stenosis Group' not in df_internal.columns) or
            ('Length' not in df_internal.columns) or
            (location_col not in df_internal.columns)):
            groups = []
        else:
            groups = df_internal.groupby(['Stenosis Group', 'Length', location_col])

    else:
        groups = df_plot.groupby(['Stenosis Group', 'Length', location_col]) if location_col in df_plot.columns else []

    # --- Plot grouped/internal points (original styling for non-external; circles in external mode) ---
    for (sten_val, length_val, loc), gdf in groups:
        if pd.isna(sten_val) or pd.isna(length_val):
            continue

        marker_style = 'o' if external_mode else stenosis_marker(sten_val)
        linestyle = snap_length_style(length_val, tol=0.15)
        line_color = vessel_line_color(loc)

        scatter_kwargs = dict(
            alpha=alpha_scatter,
            s=s_scatter,
            marker=marker_style,
            zorder=3,
        )
        if has_color_dimension:
            scatter_kwargs.update(dict(c=gdf[color_col], cmap=cmap, norm=norm))
            if external_mode and external_style == 'ext_style_2':
                scatter_kwargs.update(dict(edgecolors='face', linewidths=1.0))
            else:
                scatter_kwargs.update(dict(edgecolors='black', linewidths=1.0))
        else:
            scatter_kwargs.update(dict(color=no_color_marker_color))
            if external_mode and external_style == 'ext_style_2':
                scatter_kwargs.update(dict(edgecolors=no_color_marker_color, linewidths=1.0))
            else:
                scatter_kwargs.update(dict(edgecolors='black', linewidths=1.0))

        sc = plt.scatter(gdf[x_col], gdf[y_col], **scatter_kwargs)

        if labels and ('Geometry Number' in gdf.columns):
            for xi, yi, lab in zip(gdf[x_col], gdf[y_col], gdf['Geometry Number']):
                plt.text(
                    xi, yi, str(lab), fontsize=8, ha='right', va='top',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
                )

        if first_scatter is None:
            first_scatter = sc

        if connect_stenosis_groups and (len(gdf) > 1):
            gdf_sorted = gdf.sort_values(by=x_col)
            plt.plot(
                gdf_sorted[x_col], gdf_sorted[y_col],
                linestyle=linestyle, color=line_color,
                alpha=0.8, linewidth=2.0, zorder=2
            )

    # Colorbar
    if has_color_dimension and first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label)
    elif (not has_color_dimension) and show_monochrome_hmr_colorbar:
        hmr_ticks = np.asarray(HMR_boundaries, dtype=float)
        mono_norm = colors.Normalize(vmin=np.min(hmr_ticks), vmax=np.max(hmr_ticks), clip=True)
        mono_cmap = colors.ListedColormap([no_color_marker_color])
        mono_sm = plt.cm.ScalarMappable(norm=mono_norm, cmap=mono_cmap)
        mono_sm.set_array([])
        cbar = plt.colorbar(mono_sm, ax=plt.gca(), ticks=hmr_ticks)
        cbar.set_label(color_label if color_label else 'HMR [mmHg/cm/s]')

    # Threshold lines
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get('axis', 'y')
            value = tdict.get('value', 0.0)
            style = tdict.get('style', '--')
            c = tdict.get('color', 'gray')
            w = tdict.get('width', 2.5)
            if axis_type == 'y':
                plt.axhline(y=value, color=c, linestyle=style, linewidth=w)
            else:
                plt.axvline(x=value, color=c, linestyle=style, linewidth=w)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    # plt.ylim([-.01,.30])
    # plt.yticks([1, 2, 3])
    # plt.xticks([.6, .8, 1])
    plt.grid(False)
    # plt.title(title)
    plt.tight_layout()

    if savefig:
        if savefig is True:
            color_tag = color_col if has_color_dimension else f"no_color_{no_color_marker_color}"
            if x_col == 'P_d/P_a':
                fname = f"{y_col}_vs_FFR_col_{color_tag}"
            elif y_col == 'P_d/P_a':
                fname = f"FFR_vs_{x_col}_col_{color_tag}"
            else:
                fname = f"{y_col}_vs_{x_col}_col_{color_tag}"
        else:
            fname = str(savefig)
        fname = sanitize_filename(fname)
        plt.savefig(f'{dir}/{fname}.png', dpi=dpi, transparent=True, bbox_inches='tight')
        plt.savefig(f'{dir}/{fname}.svg', transparent=True, bbox_inches='tight')
        print(f"saved → {savefig}")

    # plt.show()
    plt.close()

    # Report counts
    if external_mode:
        n_total = len(df_plot)
        n_internal = len(df_internal)
        n_external = len(df_external)
        print(f"{title or f'{y_col} vs {x_col}'}: {n_total} points "
              f"(internal={n_internal}, external={n_external})")
    else:
        n_samples = len(df_plot)
        print(f"{title or f'{y_col} vs {x_col}'}: {n_samples} points")

# -----------------------------
# 4) PLOTS
# -----------------------------
if plot1:
    # PLOT 1) CFR vs FFR (P_d/P_a), colored by BMR/HMR
    # df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
    df_filtered_cfr = df[
        (df['CFR'].notna()) &
        (df['P_d/P_a'].notna())
        # (df['source'] == 'mine')
    ]
    boundaries_cfr = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='P_d/P_a', y_col='CFR', color_col='BMR/HMR',
        x_label='FFR', y_label='CFR',
        title='CFR vs FFR, Colored by BMR/HMR',
        cmap_name=LAB_BLUE_CMAP,
        cmap_floor=FIG7CMAPFLOOR-0.1,
        cmap_ceil=FIG7CMAPCEIL-0.1,
        alpha_scatter=FIG7ALPHA,
        custom_boundaries=boundaries_cfr,
        color_label='BMR/HMR',
        add_threshold=[
            {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
            {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=False,
        show_singletons=True,
        external_data_source='garcia',
        external_style=EXTERNAL_STYLE,
        savefig=True,
        figsize=FIG7SIZE
    )

# PLOT 1.5.....) CFR vs FFR (P_d/P_a), colored by HMR
if plot1_5:
    df_filtered_cfr = df[(df['CFR'].notna()) &
                         (df['P_d/P_a'].notna()) &
                         (df['source'] == 'mine')
                         # & (df['Location'] == 'LAD')
                         # & (df['R_micro'] == 0)
                         ]
    boundaries_cfr = np.linspace(1, 7, 7)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='P_d/P_a', y_col='CFR', color_col='HMR',
        x_label='FFR', y_label='CFR',
        title='',
        cmap_name='BuPu',
        custom_boundaries=HMR_boundaries,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[
            {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray'},
            {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray'}
        ],
        connect_stenosis_groups=True,
        show_singletons=True,
        savefig=True,
        labels=False,
        alpha_scatter=0.8,
        location_filter=location_filter,
        figsize=FIG8SIZE
    )

if plot2:
    # PLOT 2) HSR vs HMR, colored by FFR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna()
        & (df['Location'] == 'LAD')
        ]

    boundaries = np.linspace(0.5, 1.0, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col='HSR', color_col='P_d/P_a',
        x_label='HMR [mmHg/cm/s]', y_label='HSR [mmHg/cm/s]',
        title='HSR vs. HMR, Colored by FFR',
        cmap_name='cividis',
        custom_boundaries=boundaries,
        color_label='FFR',
        # add_threshold=[{'axis': 'y', 'value': 2.0}],
        show_singletons=False,
        connect_stenosis_groups=True
    )

if plot25:
    # PLOT 2) HSR vs HMR, no color map
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['Q_distal'].notna()
        # & (df['Location'] == 'LAD')
        ]
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='HSR [mmHg/cm/s]',
        title='HSR vs. HMR',
        alpha_scatter=0.8,
        use_colormap=False,
        no_color_marker_color='black',
        show_monochrome_hmr_colorbar=True,
        color_label='HMR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 2.0}],
        show_singletons=True,
        connect_stenosis_groups=True,
        savefig=True,
        location_filter=location_filter,
        figsize = (7.5,5)
    )

if plot3:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0) &
        ~(df['Stenosis Group'].round(2) == 0.48) &
        (df['Location'] == 'LCX')
        ]
    boundaries_third = np.linspace(1, 7, 5)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='P_d/P_a', y_col='CFR', color_col='HMR',
        x_label='FFR', y_label='CFR',
        title='FFR vs HMR, Colored by HSR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0},
                       {'axis': 'x', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        savefig=True
    )

if plot35:
    # PLOT 3.5) FFR (P_d/P_a) vs BMR/HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['HSR'].notna() &
        ~(df['Stenosis Group'].round(2) == 0.48) &
        df['P_d/P_a'].notna() &
        (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(0.2, 1.2, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='BMR/HMR', y_col='P_d/P_a', color_col='HSR',
        x_label='BMR/HMR', y_label='FFR',
        title='FFR vs BMR/HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        show_singletons=False,
        connect_stenosis_groups=True
    )

if plot36:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['v_distal'].notna() &
        df['P_d/P_a'].notna()
        # & (df['R_micro'] == 0)
        # & (df['Stenosis_Percentage'] == 0)
        # & (df['Location'] == 'LAD')
        ]
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HMR', y_col='P_d/P_a', color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by Distal Velocity',
        cmap_name='Reds',
        cmap_floor=0.25,
        cmap_ceil=0.9,
        custom_boundaries= HSR_boundaries,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        alpha_scatter=0.8,
        location_filter=location_filter,
        figsize=FIG4SIZE,
        savefig=True,
    )

if plot37:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['v_distal'].notna() &
        df['P_d/P_a'].notna()
        # & (df['R_micro'] == 0)
        # & (df['Stenosis_Percentage'] == 0)
        # & (df['Location'] == 'LAD')
        ]
    boundaries_third = HSR_boundaries
    make_smart_scatter(
        data=df_filtered_third,
        x_col='Length', y_col='P_d/P_a', color_col='HSR',
        x_label='Length [cm]', y_label='FFR',
        title='FFR vs HMR, Colored by Distal Velocity',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        alpha_scatter=0.8,
        location_filter=location_filter,
        savefig=True,
    )


if plot4:
    # PLOT 4) FFR (P_d/P_a) vs HSR, colored by HMR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HSR'].notna() &
        df['v_distal'].notna() &
        df['P_d/P_a'].notna()
        # & (df['R_micro'] == 0)
        # & (df['Stenosis_Percentage'] == 0)
        # & (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(1, 7, 7)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HSR', y_col='P_d/P_a', color_col='HMR',
        x_label='HSR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by Distal Velocity',
        cmap_name='BuPu',
        custom_boundaries=boundaries_third,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        alpha_scatter=0.8,
        location_filter=location_filter,
        savefig=True
    )

if plot45:
    # PLOT 4.5) FFR (P_d/P_a) vs HSR, colored by BMR/HMR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna()
        ]
    boundaries_fourth = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HSR', y_col='P_d/P_a', color_col='BMR/HMR',
        x_label='HSR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HSR, Colored by BMR/HMR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries_fourth,
        color_label='BMR/HMR',
        add_threshold=[{'axis': 'y', 'value': 0.8}]
    )

if plot5:
    # PLOT 5) CFR vs HMR, colored by HSR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['HMR'].notna() &
        df['HSR'].notna()
        # & (df['R_micro'] == 0)
        # & (df['Location'] == 'LCX')
        ]
    boundaries_cfr_hmr = np.linspace(0, 1.6, 5)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HMR', y_col='CFR', color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.25,
        cmap_ceil=0.9,
        custom_boundaries=HSR_boundaries,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
        connect_stenosis_groups=True,
        show_singletons=False,
        savefig=True,
        alpha_scatter=0.8,
        location_filter=location_filter,
        figsize = FIG4SIZE
    )

if plot55:
    # PLOT 3.5) FFR (P_d/P_a) vs BMR/HMR, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['HSR'].notna() &
        df['CFR'].notna() &
        (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(0.2, 1.2, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='BMR/HMR', y_col='CFR', color_col='HSR',
        x_label='BMR/HMR', y_label='CFR',
        title='CFR vs BMR/HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
        show_singletons=False,
        connect_stenosis_groups=True
    )

if plot56:
    # PLOT 3) FFR (P_d/P_a) vs HSR, colored by HMR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HSR'].notna() &
        df['v_distal'].notna() &
        df['P_d/P_a'].notna()
        # & (df['R_micro'] == 0)
        # & (df['Stenosis_Percentage'] == 0)
        # & (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(1, 7, 7)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HSR', y_col='CFR', color_col='HMR',
        x_label='HSR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HMR, Colored by Distal Velocity',
        cmap_name='BuPu',
        custom_boundaries=boundaries_third,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
        connect_stenosis_groups=True,
        show_singletons=False,
        alpha_scatter=0.8,
        location_filter=location_filter,
        savefig=True
    )

if plot57:
    # PLOT 5.6) CFR vs HSR, colored by BMR/HMR
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['BMR/HMR'].notna() &
        df['HSR'].notna()
        ]
    boundaries_cfr_hmr = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HSR', y_col='CFR', color_col='BMR/HMR',
        x_label='HSR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HSR, Colored by BMR/HMR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='BMR/HMR',
        add_threshold=[{'axis': 'y', 'value': 2.0}]
    )

if plot6:
    # PLOT 3.5) FFR (P_d/P_a) vs BMR/HMR, colored by HSR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['BMR/HMR'].notna() &
        df['HSR'].notna() &
        ~(df['Stenosis Group'].round(2) == 0.48) &
        df['P_d/P_a'].notna() &
        (df['Location'] == 'LAD')
        ]
    if log_switch:
        df_filtered.loc[:, 'CFR/FFR'] = np.log10(df_filtered['CFR/FFR'])
    boundaries_third = np.linspace(0.2, 1.2, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='BMR/HMR', y_col='CFR/FFR', color_col='HSR',
        x_label='BMR/HMR', y_label='CFR/FFR',
        title='CFR/FFR vs BMR/HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
        show_singletons=False,
        connect_stenosis_groups=True
    )

if plot7:
    # PLOT 2) HMR vs P_Loss_Coeff, colored by FFR
    df_filtered_hmr = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna() &
        df['P_d/P_a'].notna()
        ]
    boundaries = np.linspace(0.5, 1, 5)
    make_smart_scatter(
        data=df_filtered_hmr,
        x_col='P_Loss_Coeff', y_col='HMR', color_col='P_d/P_a',
        x_label='$log_{10}(ζ_{L})$', y_label='HMR [mmHg/cm/s]',
        title='$log_{10}(ζ_{L})$ vs HMR, Colored by FFR',
        cmap_name='RdYlGn',
        custom_boundaries=boundaries,
        color_label='FFR'
    )

if plot8:
    # PLOT 3) FFR (P_d/P_a) vs HMR, colored by P_Loss_Coeff
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna() &
        df['P_d/P_a'].notna()
        ]
    boundaries_third = np.linspace(0.5, 2.5, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='HMR', y_col='P_d/P_a', color_col='P_Loss_Coeff',
        x_label='HMR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by $log_{10}(ζ_{L})$',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_third,
        color_label='$log_{10}(ζ_{L})$',
        add_threshold=[{'axis': 'y', 'value': 0.8}]
    )

if plot9:
    # PLOT 4) FFR (P_d/P_a) vs P_Loss_Coeff, colored by HMR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0)
        ]
    boundaries_fourth = np.linspace(1, 3.5, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='P_Loss_Coeff', y_col='P_d/P_a', color_col='HMR',
        x_label='$log_{10}(ζ_{L})$', y_label='FFR',
        title='FFR vs P_Loss_Coeff, Colored by HMR',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_fourth,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False
    )

if plot10:
    # PLOT 5) CFR vs HMR, colored by P_Loss_Coeff
    df_filtered_cfr_hmr = df[
        df['CFR'].notna() &
        df['HMR'].notna() &
        df['P_Loss_Coeff'].notna()
        ]
    boundaries_cfr_hmr = np.linspace(0.5, 2.5, 6)
    make_smart_scatter(
        data=df_filtered_cfr_hmr,
        x_col='HMR', y_col='CFR', color_col='P_Loss_Coeff',
        x_label='HMR [mmHg/cm/s]', y_label='CFR',
        title='CFR vs HMR, Colored by $log_{10}(ζ_{L})$',
        cmap_name='RdYlGn_r',
        custom_boundaries=boundaries_cfr_hmr,
        color_label='$log_{10}(ζ_{L})$',
        add_threshold=[{'axis': 'y', 'value': 2.0}],
    )

if plot11:
    wss_var = 'WSS_LE_Area_min'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna()
        # (df[wss_var] != 0) &
        # (df['R_micro'] == 0) &
        # (df['Location'] == 'LAD')]
        ]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area < 0.5 Pa [cm2]',
        title='Area LE WSS < 0.5 Pa vs HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        location_filter=location_filter
    )

if plot12:
    wss_var = 'WSS_TE_Area_min'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna()
        # (df[wss_var] != 0) &
        # (df['R_micro'] == 0) &
        # (df['Location'] == 'LAD')
    ]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Area [cm2]',
        title='Area TE WSS < 0.5 Pa vs HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        location_filter=location_filter
    )
boundaries_thirteen_thru_sixteen = v_boundaries
color_col_thirteen_thru_sixteen = 'v_distal'

if plot13:
    wss_var = 'WSS_Area_Bifur_min'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df[wss_var].notna()
        # # (df[wss_var] != 0) &
        # (df['R_micro'] == 0) &
        # (df['Location'] == 'LAD')
    ]
    df_filtered.loc[:, wss_var] = 100* df_filtered[wss_var]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    # boundaries = HSR_boundaries
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col=color_col_thirteen_thru_sixteen,
        x_label='HMR [mmHg/cm/s]', y_label='Normalized ALWSS [%]',
        title='LAD-LCx Bifurcation',
        cmap_name='viridis',
        custom_boundaries=boundaries_thirteen_thru_sixteen,
        color_label='distal velocity [cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        location_filter=location_filter,
        savefig=True,
        figsize=FIG6SIZE,
        y_lim=[-.01, .30]
    )

if plot14:
    wss_var = 'WSS_LE_Area'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna()
        # df[wss_var].notna()
        # # (df[wss_var] != 0) &
        # (df['R_micro'] == 0) &
        # (df['Location'] == 'LAD')
    ]
    df_filtered.loc[:, wss_var] = 100* df_filtered[wss_var]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = HSR_boundaries
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col=color_col_thirteen_thru_sixteen,
        x_label='HMR [mmHg/cm/s]', y_label='Normalized AHWSS [%]',
        title='Plaque leading edge',
        cmap_name='viridis',
        custom_boundaries=boundaries_thirteen_thru_sixteen,
        color_label='distal velocity [cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        location_filter=location_filter,
        savefig=True,
        figsize=FIG6SIZE,
        y_lim=[0, 30]
    )

if plot15:
    wss_var = 'WSS_TE_Area'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna()
        # df[wss_var].notna()
        # (df[wss_var] != 0) &
        # (df['R_micro'] == 0) &
        # (df['Location'] == 'LAD')
    ]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = HSR_boundaries
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col=color_col_thirteen_thru_sixteen,
        x_label='HMR [mmHg/cm/s]', y_label='Normalized AHWSS [%]',
        title='Plaque trailing edge',
        cmap_name='viridis',
        custom_boundaries=boundaries_thirteen_thru_sixteen,
        color_label='distal velocity [cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        location_filter=location_filter,
        savefig=True,
        figsize=FIG6SIZE,
        y_lim=[-.01, .30]
    )

if plot16:
    wss_var = 'WSS_Area_Bifur'
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna()
        # df[wss_var].notna()
        # (df[wss_var] != 0) &
        # (df['R_micro'] == 0) &
        # (df['Location'] == 'LAD')
    ]
    df_filtered.loc[:, wss_var] = 100* df_filtered[wss_var]
    if log_switch:
        df_filtered.loc[:, wss_var] = np.log10(df_filtered[wss_var])
    boundaries_third = HSR_boundaries
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col=wss_var, color_col=color_col_thirteen_thru_sixteen,
        x_label='HMR [mmHg/cm/s]', y_label='Normalized AHWSS [%]',
        title='LAD-LCx Bifurcation',
        cmap_name='viridis',
        custom_boundaries=boundaries_thirteen_thru_sixteen,
        color_label='distal velocity [cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=True,
        show_singletons=False,
        location_filter=location_filter,
        savefig=True,
        figsize=FIG6SIZE,
        y_lim=[0, 30]
    )

if plot17:
    # PLOT 17) FFR (P_d/P_a) vs v_distal, colored by HSR
    df_filtered_third = df[
        (df['Condition'] == 'Hyperemic') &
        df['v_distal'].notna() &
        df['HSR'].notna() &
        df['P_d/P_a'].notna() &
        (df['R_micro'] == 0) &
        (df['Location'] == 'LAD')
        ]
    boundaries_third = np.linspace(0.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered_third,
        x_col='v_distal', y_col='P_d/P_a', color_col='HSR',
        x_label='distal velocity [cm/s]', y_label='FFR',
        title='FFR vs HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries_third,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[{'axis': 'y', 'value': 0.8}],
        connect_stenosis_groups=False,
        show_singletons=True
    )

if plot18:
    # PLOT 18) Q_dsital vs HMR, colored by HSR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['Q_distal'].notna()
        # & (df['Location']=='LAD')
        ]

    boundaries = np.linspace(.1, 1.1, 6)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR', y_col='v_distal', color_col='HSR',
        x_label='HMR [mmHg/cm/s]', y_label='Distal flow rate [cm3/s]',
        title='Q_distal vs. HMR, Colored by HSR',
        cmap_name='Reds',
        cmap_floor=0.33,
        cmap_ceil=0.9,
        custom_boundaries=boundaries,
        color_label='HSR [mmHg/cm/s]',
        # add_threshold=[{'axis': 'y', 'value': 2.0}],
        show_singletons=True,
        connect_stenosis_groups=True
    )

if plot19:
    # PLOT 19) v_dsital vs HSR, colored by HMR
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['v_distal'].notna()
        # & (df['Location']=='LCX')
        ]
    make_smart_scatter(
        data=df_filtered,
        x_col='HSR', y_col='Q_distal', color_col='HMR',
        x_label='HSR [mmHg/cm/s]', y_label='Distal flow rate [cm$^3$/s]',
        title='Q_distal vs. HSR, Colored by HMR',
        cmap_name='BuPu',
        custom_boundaries=HMR_boundaries,
        color_label='HMR [mmHg/cm/s]',
        location_filter = location_filter,
        show_singletons=True,
        connect_stenosis_groups=True,
        savefig=True,
        figsize=FIG5SIZE
    )

if plot21:
    # PLOT 2) HSR vs HMR, colored by flow rate
    df_filtered = df[
        (df['Condition'] == 'Hyperemic') &
        df['HMR'].notna() &
        df['HSR'].notna() &
        df['Q_distal'].notna()
        & (df['Location'] == 'LAD')
        ]
    df_filtered['HMR/(HMR+HSR)'] = df['HMR'] / (df['HMR'] + df['HSR'])

    boundaries = np.linspace(1, 1.1, 2)
    make_smart_scatter(
        data=df_filtered,
        x_col='HMR/(HMR+HSR)', y_col='FFR', color_col='Q_distal',
        x_label='HMR/(HMR+HSR)', y_label='FFR',
        title='HSR vs. HMR, Colored by Q_distal',
        cmap_name='binary',
        custom_boundaries=boundaries,
        color_label='flow rate [cm3/s]',
        # add_threshold=[{'axis': 'y', 'value': 2.0}],
        show_singletons=True,
        connect_stenosis_groups=True
    )

import matplotlib.tri as mtri
from matplotlib.lines import Line2D, lineStyles

# -----------------------------
# NEW FLAG
# -----------------------------

def make_ffr_response_surface(
    data,
    x_col,
    y_col,
    z_col,
    iso_levels,
    iso_target,
    show_points=True,
    dpi=200,
    savefig=False,
    labels=False
):
    """
    Draw a 2D contour of FFR over (HMR, ζ). Overlays iso-FFR=0.80.
    Works on scattered data via triangular contours (no SciPy required).
    """
    dfp = data.copy()
    dfp = dfp[dfp[x_col].notna() & dfp[y_col].notna() & dfp[z_col].notna()]
    # Apply stenosis filter if requested
    if exclude_no_stenosis and 'Stenosis Percentage' in dfp.columns:
        dfp = dfp[dfp['Stenosis Percentage'] >= 0.05]

    if len(dfp) < 6:
        print(f'Not enough points to triangulate ({len(dfp)}). Aborting.')
        return

    x = dfp[x_col].to_numpy(float)
    y = dfp[y_col].to_numpy(float)
    z = dfp[z_col].to_numpy(float)

    # Triangulate scattered (x,y)
    tri = mtri.Triangulation(x, y)

    plt.figure(figsize=(7.5, 5))

    # Filled contours of FFR
    cf = plt.tricontourf(tri, z, levels=iso_levels, cmap='RdYlGn')  # green=better FFR
    # Contour lines for readability
    cl = plt.tricontour(tri, z, levels=iso_levels, colors='k', linewidths=0.5, alpha=0.4)

    # Emphasize iso-FFR = 0.80 (drawn in black, thicker)
    plt.tricontour(tri, z, levels=[iso_target], colors='black', linewidths=2.2, linestyles=['--'])

    # # Optional polish
    # plt.scatter(x, y, c='k', s=18, alpha=0.5,
    #             edgecolors='white', linewidths=0.4, zorder=3)

    # If you want to visualize the triangulation mesh itself:
    plt.triplot(tri, color='white', alpha=0.5, linewidth=0.4)

    # Legend proxy for the iso-FFR line
    if z_col == 'P_d/P_a':
        legend_handles = [Line2D([0], [0], color='black', lw=2.2, label=f'FFR = {iso_target:.2f}', linestyle='--')]
    else:
        legend_handles = [Line2D([0], [0], color='black', lw=2.2, label=f'{z_col} = {iso_target:.2f}',linestyle='--')]

    plt.legend(handles=legend_handles, loc='upper right', frameon=False)

    # Optionally show samples
    if show_points:
        plt.scatter(x, y, c='k', s=18, alpha=0.5, edgecolors='white', linewidths=0.4, zorder=3)
        if labels and ('Geometry Number' in dfp.columns):
            import matplotlib.patheffects as pe
            for xi, yi, lab in zip(x, y, dfp['Geometry Number']):
                plt.text(xi, yi, str(lab), fontsize=8, ha='right', va='bottom',
                         path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    cbar = plt.colorbar(cf)
    cbar.set_label(z_col)

    plt.xlabel('HMR [mmHg/cm/s]')
    plt.ylabel(f'{y_col} [mmHg/cm/s]')
    plt.title(f'{z_col} across ({x_col}, {y_col})')
    plt.grid(False)
    plt.legend(loc='best', frameon=False)
    # plt.tight_layout()

    if savefig:
        out_png = f'images/surface_{z_col}_wrt_{x_col}_and_{y_col}.png'
        out_svg = f'images/surface_{z_col}_wrt_{x_col}_and_{y_col}.svg'
        plt.savefig(out_png, dpi=dpi, transparent=True, bbox_inches='tight')
        # plt.savefig(out_svg, transparent=True, bbox_inches='tight')
        print(f'saved: {out_png}, {out_svg}')
    plt.show()
    plt.close()

    # # Quick coverage printout for manuscript sanity-checks
    # print(f"{f'{z_col} across ({x_col}, {y_col})'}: n={len(dfp)} "
    #       f"| {x_col} range [{x.min():.2f}, {x.max():.2f}] "
    #       f"| {y_col} range [{y.min():.2f}, {y.max():.2f}] "
    #       f"| {z_col} range [{z.min():.2f}, {z.max():.2f}]")

x_var = 'HMR'
y_var = 'HSR'
z_var = 'FFR'

if plot20:
    # Use hyperemic rows (FFR), drop very sparse/odd groups if you want
    dff = df[
        (df['Condition'] == 'Hyperemic') &
        df[x_var].notna() &
        df[y_var].notna() &
        df[z_var].notna()
        # &(df['Location'] == 'LAD')
    ]

    # # Optional: gently clip extremes to stabilize contours (prevents skinny triangles)
    # dff['HMR'] = dff['HMR'].clip(lower=dff['HMR'].quantile(0.02), upper=dff['HMR'].quantile(0.98))
    # dff['P_Loss_Coeff'] = dff['P_Loss_Coeff'].clip(lower=dff['P_Loss_Coeff'].quantile(0.02),
    #                                                 upper=dff['P_Loss_Coeff'].quantile(0.98))
    make_ffr_response_surface(
        data=dff,
        x_col=x_var,
        y_col=y_var,
        z_col=z_var,
        iso_levels=(np.linspace(1, 3.5, 6) if z_var=='CFR' else np.linspace(0.5, 1, 6)),
        iso_target=(2 if z_var=='CFR' else 0.8),
        show_points=True,
        labels=False,
        savefig=False
    )

if plot22:
    # PLOT 22) P_Loss_Coeff vs HMR, colored by FFR
    # df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
    df_filtered_cfr = df[
        (df['P_Loss_Coeff'].notna()) &
        (df['P_d/P_a'].notna())
        & (df['HMR'].notna())
        # (df['source'] == 'mine')
    ]
    boundaries_cfr = np.linspace(0.5, 1, 6)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='HMR', y_col='P_Loss_Coeff', color_col='FFR',
        x_label='HMR [mmHg/cm/s]', y_label='$log_{10}(ζ_{L})$',
        title='P_loss_coeff vs HMR, Colored by BMR/HMR',
        cmap_name=LAB_TEAL_CMAP,
        cmap_floor=FIG7CMAPFLOOR,
        cmap_ceil=FIG7CMAPCEIL,
        alpha_scatter=FIG7ALPHA,
        custom_boundaries=boundaries_cfr,
        continuous_colorbar=False,
        color_label='FFR',
        add_threshold=[
            # {'axis': 'y', 'value': 2.0, 'style': '--', 'color': 'gray', 'width': 0.8},
            # {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=False,
        show_singletons=True,
        external_data_source='garcia',
        external_style=EXTERNAL_STYLE,
        savefig=True,
        figsize=FIG7SIZE

    )

if plot23:
    # PLOT 23) P_Loss_Coeff vs HMR, colored by FFR
    # df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
    df_filtered_cfr = df[
        (df['P_Loss_Coeff'].notna()) &
        (df['P_d/P_a'].notna())
        & (df['HMR'].notna())
        # (df['source'] == 'mine')
    ]
    boundaries_cfr = np.linspace(1, 7, 7)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='P_Loss_Coeff', y_col='FFR', color_col='HMR',
        x_label='$log_{10}(ζ_{L})$', y_label='FFR',
        title='FFR vs. P_loss_coeff, Colored by BMR/HMR',
        cmap_name=FIG7_PLOT23_CMAP,
        cmap_floor=FIG7CMAPFLOOR,
        cmap_ceil=FIG7CMAPCEIL,
        alpha_scatter=FIG7ALPHA,
        custom_boundaries=boundaries_cfr,
        continuous_colorbar=False,
        color_label='HMR [mmHg/cm/s]',
        add_threshold=[
            {'axis': 'y', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8},
            # {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=False,
        show_singletons=True,
        external_data_source='garcia',
        external_style=EXTERNAL_STYLE,
        savefig=True,
        figsize=FIG7SIZE
    )

if plot24:
    # PLOT 23) P_Loss_Coeff vs HMR, colored by FFR
    # df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
    df_filtered_cfr = df[
        (df['P_Loss_Coeff'].notna()) &
        (df['P_d/P_a'].notna())
        & (df['HMR'].notna())
        # (df['source'] == 'mine')
    ]
    boundaries_cfr = np.linspace(0, 4, 5)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='HMR', y_col='FFR', color_col='P_Loss_Coeff',
        x_label='HMR [mmHg/cm/s]', y_label='FFR',
        title='FFR vs. HMR, Colored by P_Loss_Coeff',
        cmap_name=FIG7_PLOT24_CMAP,
        cmap_floor=FIG7CMAPFLOOR,
        cmap_ceil=FIG7CMAPCEIL,
        alpha_scatter=FIG7ALPHA,
        custom_boundaries=boundaries_cfr,
        continuous_colorbar=False,
        color_label='$log_{10}(ζ_{L})$',
        add_threshold=[
            {'axis': 'y', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8},
            # {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=False,
        show_singletons=True,
        external_data_source='garcia',
        external_style=EXTERNAL_STYLE,
        savefig=True,
        figsize=FIG7SIZE
    )

if plot26:
    # PLOT 26) FFR vs Q_distal, colored by HSR
    # df['ash'] = df['HMR']/(df['HMR']+df['HSR'])
    df_filtered_cfr = df[
        (df['P_d/P_a'].notna())
        & (df['HMR'].notna())
        ]
    boundaries_cfr = np.linspace(0, 4, 5)
    make_smart_scatter(
        data=df_filtered_cfr,
        x_col='Q_distal', y_col='FFR', color_col='HSR',
        x_label='Q_distal [cm3/s]', y_label='FFR',
        title='FFR vs. Q_distal, Colored by HSR',
        cmap_name=FIG7_PLOT26_CMAP,
        cmap_floor=FIG7CMAPFLOOR,
        cmap_ceil=FIG7CMAPCEIL,
        alpha_scatter=0.8,
        custom_boundaries=boundaries_cfr,
        continuous_colorbar=True,
        color_label='HSR [mmHg/cm/s]',
        add_threshold=[
            {'axis': 'y', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8},
            # {'axis': 'x', 'value': 0.8, 'style': '--', 'color': 'gray', 'width': 0.8}
        ],
        connect_stenosis_groups=True,
        show_singletons=False,
        savefig=True
    )



# OPTIONAL: quantify local sensitivities
if sensitivies_analysis:
    dff = df[
        (df['Condition'] == 'Hyperemic') &
        df[x_var].notna() &
        df[y_var].notna() &
        df[z_var].notna()
        # &(df['Location'] == 'LAD')      # optionally restrict
        # & ~(df['Stenosis Group'].round(2) == 0.48)  # optionally exclude a stray bin
        # & ~(df['R_total'] == 0.81)
        ]
    if z_var == 'CFR':
        thresh = 2
        tol = 0.5
        near = dff[np.abs(dff[z_var] - thresh) <= tol].copy()

    else: # if y_var = FFR
        thresh = 0.8
        tol = 0.1
        near = dff[np.abs(dff[z_var] - thresh) <= tol].copy()

    if exclude_no_stenosis and 'Stenosis Percentage' in near.columns:
        near = near[near['Stenosis Percentage'] >= 0.05]

    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm

    # ... build `near` exactly as you already do ...

    if len(near) >= 8:
        # --- standardize predictors in-band (same as your code) ---
        X1 = (near[x_var] - near[x_var].mean()) / near[x_var].std(ddof=1)
        X2 = (near[y_var] - near[y_var].mean()) / near[y_var].std(ddof=1)
        Y  = near[z_var].to_numpy(float)

        X_sm = sm.add_constant(np.c_[X1.to_numpy(float), X2.to_numpy(float)])
        m_ols = sm.OLS(Y, X_sm).fit()

        # Residual normality (OLS-only assumption check for inference)
        W_resid, p_resid = stats.shapiro(m_ols.resid) if len(m_ols.resid) <= 5000 else (np.nan, np.nan)

        # Choose covariance: plain vs HC3 if non-normal
        use_hc3 = (not np.isnan(p_resid)) and (p_resid < 0.05)
        m = sm.OLS(Y, X_sm).fit(cov_type='HC3') if use_hc3 else m_ols
        cov_label = "HC3 robust SEs" if use_hc3 else "conventional SEs"

        # Coefs (standardized slopes are the coefficients on X1, X2)
        b0, b1_std, b2_std = m.params
        ci = m.conf_int(alpha=0.05)  # rows align with params: const, X1, X2
        ci_b1_std = ci[1]
        ci_b2_std = ci[2]

        # Fit stats
        resid = Y - m_ols.fittedvalues  # R^2, RMSE from the plain OLS fit
        n = len(Y); p = X_sm.shape[1]
        rss = float(np.sum(resid**2))
        sst = float(np.sum((Y - Y.mean())**2))
        r2 = 1 - rss / sst if sst > 0 else np.nan
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
        rmse = float(np.sqrt(rss / n))

        print(
            f"Local standardized sensitivities near {z_var} = {thresh:0.2f} ± {tol} "
            f"(n={n}; inference via {cov_label}; Shapiro-resid p={p_resid:.3g}):\n"
            f"  {x_var}: {b1_std:+.3f}  [ {ci_b1_std[0]:+.3f}, {ci_b1_std[1]:+.3f} ]\n"
            f"  {y_var}: {b2_std:+.3f}  [ {ci_b2_std[0]:+.3f}, {ci_b2_std[1]:+.3f} ]\n"
            f"  R^2={r2:.3f}, Adj R^2={adj_r2:.3f}, RMSE={rmse:.4f}\n"
        )

        # --- (optional) convert standardized slopes back to raw units, using same CIs ---
        mu1, s1 = near[x_var].mean(), near[x_var].std(ddof=1)
        mu2, s2 = near[y_var].mean(), near[y_var].std(ddof=1)
        muy      = near[z_var].mean()

        b1_per_unit = b1_std / s1
        b2_per_unit = b2_std / s2
        ci_b1_per_unit = ci_b1_std / s1
        ci_b2_per_unit = ci_b2_std / s2
        b0_raw = muy - b1_per_unit*mu1 - b2_per_unit*mu2

        print(f"HMR slope per 1 unit ({x_var}): {b1_per_unit:+.4f} "
              f"[{ci_b1_per_unit[0]:+.4f}, {ci_b1_per_unit[1]:+.4f}]")
        print(f"HSR slope per 1 unit ({y_var}): {b2_per_unit:+.4f} "
              f"[{ci_b2_per_unit[0]:+.4f}, {ci_b2_per_unit[1]:+.4f}]")
        print(f"Intercept in raw units (at means): {b0_raw:+.4f}")
    else:
        print("Not enough near-boundary points for local sensitivity estimate.")

    # X1, X2 are your standardized predictors: (x - mean)/std(ddof=1)
    # If you didn't save raw means/stds yet, recompute them from the "near" slice:
    x1_raw = near[x_var].to_numpy(float)
    x2_raw = near[y_var].to_numpy(float)
    y_raw  = near[z_var].to_numpy(float)

    mu1, s1 = x1_raw.mean(), x1_raw.std(ddof=1)
    mu2, s2 = x2_raw.mean(), x2_raw.std(ddof=1)
    muy      = y_raw.mean()

    # beta from your OLS on [1, X1_std, X2_std]
    # ci95 is the 3x2 array you printed (rows: intercept, HMR, HSR)
    # b0, b1_std, b2_std = beta
    # ci_b1_std = ci95[1]   # [low, high] for standardized b1
    # ci_b2_std = ci95[2]

    # Convert to raw-unit slopes (Y change per 1 unit of X)
    b1_per_unit = b1_std / s1
    b2_per_unit = b2_std / s2
    ci_b1_per_unit = ci_b1_std / s1
    ci_b2_per_unit = ci_b2_std / s2

    # Raw-unit intercept
    b0_raw = muy - b1_per_unit*mu1 - b2_per_unit*mu2

    print(f"HMR slope (per 1 unit of {x_var}): {b1_per_unit:+.4f} "
          f"[{ci_b1_per_unit[0]:+.4f}, {ci_b1_per_unit[1]:+.4f}]")
    print(f"HSR slope (per 1 unit of {y_var}): {b2_per_unit:+.4f} "
          f"[{ci_b2_per_unit[0]:+.4f}, {ci_b2_per_unit[1]:+.4f}]")
    print(f"Intercept in raw units (at X means): {b0_raw:+.4f}")

    E1 = b1_per_unit * (mu1 / muy)
    E2 = b2_per_unit * (mu2 / muy)
    print(f"Elasticity at mean: HMR {E1:+.3f}, HSR {E2:+.3f}  (Δ%Y per 1% ΔX)")


    check_points_near_boundary = False
    if check_points_near_boundary:
        ###------------------------------------------------------------------------####
        d0 = df.copy()
        print("TOTAL rows:", len(d0))

        d1 = d0[d0['Condition'] == 'Hyperemic']
        print("Hyperemic rows:", len(d1))

        d2 = d1[d1['HMR'].notna() & d1['P_Loss_Coeff'].notna() & d1['P_d/P_a'].notna()]
        print("…with HMR, ζ, FFR present:", len(d2))

        # how many near the decision boundary?
        band = 0.08  # try 0.03–0.05
        near = d2[np.abs(d2['P_d/P_a'] - 0.80) <= band]
        print(f"Near FFR=0.80 (±{band}):", len(near))
        print(near[['Geometry Number','HMR','P_Loss_Coeff','P_d/P_a']].to_string(index=False))

    linearity_check = False
    if linearity_check:
        import numpy as np
        def _build_X(df, x1, x2, kind="additive"):
            H = df[x1].to_numpy(float)
            Z = df[x2].to_numpy(float)
            if kind == "additive":
                X = np.c_[np.ones(len(df)), H, Z]
            elif kind == "interaction":
                X = np.c_[np.ones(len(df)), H, Z, H * Z]
            elif kind == "quadratic":
                X = np.c_[np.ones(len(df)), H, Z, H * Z, H ** 2, Z ** 2]
            else:
                raise ValueError("unknown kind")
            return X
        def _ols_metrics(X, y):
            # OLS via least squares
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            resid = y - yhat
            n, p = X.shape
            sse = np.sum(resid ** 2)
            sst = np.sum((y - y.mean()) ** 2)
            r2 = 1 - sse / sst if sst > 0 else np.nan
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
            sigma2 = sse / n
            # AIC for linear-Gaussian: n*log(2πσ²) + n + 2p  (constant shifts don’t affect deltas)
            aic = n * np.log(sigma2 + 1e-12) + 2 * p

            # LOO via PRESS (hat-diagonal without materializing full H)
            XtX_inv = np.linalg.inv(X.T @ X)
            # hat diagonal h_i = x_i^T (XtX)^{-1} x_i
            h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
            press = np.sum((resid / (1 - h + 1e-12)) ** 2)
            loo_rmse = np.sqrt(press / n)
            return {
                "beta": beta, "yhat": yhat, "resid": resid, "n": n, "p": p,
                "R2": r2, "AdjR2": adj_r2, "AIC": aic, "LOO_RMSE": loo_rmse
            }
        def assess_local_linearity_and_interaction(
                df,
                y_col="P_d/P_a",  # "FFR" (hyperemic) or "CFR"
                x1="HMR",
                x2="P_Loss_Coeff",  # ζ
                window_center=0.80,  # 0.80 for FFR, 2.0 for CFR
                window_halfwidth=0.05,  # ± band
                require_hyperemic=True,  # for FFR analyses
                exclude_no_stenosis=False,
                stenosis_col="Stenosis Percentage",
                min_n=8,
                label="FFR near 0.80"
        ):
            df_loc = df.copy()

            # pick rows for the outcome
            if y_col == "P_d/P_a" and require_hyperemic and "Condition" in df_loc.columns:
                df_loc = df_loc[df_loc["Condition"] == "Hyperemic"]

            # needed columns present and non-NaN
            cols = [y_col, x1, x2]
            df_loc = df_loc.dropna(subset=cols)

            # optional stenosis filter
            if exclude_no_stenosis and (stenosis_col in df_loc.columns):
                df_loc = df_loc[df_loc[stenosis_col] >= 0.05]

            # local band
            df_loc = df_loc[np.abs(df_loc[y_col] - window_center) <= window_halfwidth]

            n = len(df_loc)
            if n < min_n:
                print(f"[{label}] Not enough points in band (n={n} < {min_n}).")
                return

            y = df_loc[y_col].to_numpy(float)
            # correlations to check near-independence
            r_x1x2 = np.corrcoef(df_loc[x1], df_loc[x2])[0, 1]
            print(f"[{label}] n={n} | corr({x1},{x2}) = {r_x1x2:+.3f}")

            # Fit models
            X1 = _build_X(df_loc, x1, x2, "additive")
            X2 = _build_X(df_loc, x1, x2, "interaction")
            X3 = _build_X(df_loc, x1, x2, "quadratic")

            M1 = _ols_metrics(X1, y)
            M2 = _ols_metrics(X2, y)
            M3 = _ols_metrics(X3, y)

            # Report metrics
            def _brief(name, M):
                return (f"{name}: AdjR2={M['AdjR2']:.3f}, AIC={M['AIC']:.1f}, "
                        f"LOO-RMSE={M['LOO_RMSE']:.4f}")

            print(_brief("M1 additive", M1))
            print(_brief("M2 +interaction", M2))
            print(_brief("M3 +quadratic", M3))

            # Standardized coefficients from M1 (local SRCs)
            # z-score predictors to compare magnitude
            H = df_loc[x1].to_numpy(float)
            Z = df_loc[x2].to_numpy(float)
            H_z = (H - H.mean()) / (H.std(ddof=0) + 1e-12)
            Z_z = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-12)
            X_std = np.c_[np.ones(n), H_z, Z_z]
            beta_std, _, _, _ = np.linalg.lstsq(X_std, y, rcond=None)
            print(f"Local standardized slopes (M1): {x1}: {beta_std[1]:+.3f}, {x2}: {beta_std[2]:+.3f}, "
                  f"(intercept {beta_std[0]:+.3f})")

            # Simple decision rule for "close enough linear/additive"
            imp_12 = (M1["LOO_RMSE"] - M2["LOO_RMSE"]) / (M1["LOO_RMSE"] + 1e-12)
            imp_23 = (M2["LOO_RMSE"] - M3["LOO_RMSE"]) / (M2["LOO_RMSE"] + 1e-12)
            small_interaction_gain = (imp_12 <= 0.03)  # <~3% gain adding interaction
            small_curve_gain = (imp_23 <= 0.02)  # <~2% gain adding curvature
            low_collinearity = (abs(r_x1x2) <= 0.3)

            verdict = []
            if small_interaction_gain and small_curve_gain:
                verdict.append("no material benefit from interaction/curvature in-band")
            else:
                if not small_interaction_gain:
                    verdict.append("interaction improves fit meaningfully")
                if not small_curve_gain:
                    verdict.append("curvature improves fit meaningfully")

            if low_collinearity:
                verdict.append("predictors are weakly correlated (near-independent)")
            else:
                verdict.append("predictors moderately/highly correlated")

            print(f"Verdict: {', '.join(verdict)}")


        assess_local_linearity_and_interaction(
            df,
            y_col=z_var,  # FFR
            x1=x_var,
            x2=y_var,  # ← use HSR
            window_center=0.80 if z_var =='P_d/P_a' else 2.0,
            window_halfwidth=0.1 if z_var =='P_d/P_a' else 0.5,
            require_hyperemic=True,
            exclude_no_stenosis=True,
            label=f"FFR≈0.80 ({x_var}, {y_var})" if z_var =='P_d/P_a' else f"{z_var}≈2 ({x_var}, {y_var})"
        )
