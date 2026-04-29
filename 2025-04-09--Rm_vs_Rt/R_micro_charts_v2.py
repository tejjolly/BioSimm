import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# -----------------------------
# Global matplotlib settings
# -----------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 600
})

def plot_metric(df, y_var):
    # Drop rows with missing x/y values
    df_plot = df[df['R_total'].notna() & df[y_var].notna() & df['R_micro'].notna()].copy()
    df_plot['R_micro_nonzero'] = df_plot['R_micro'] != 0
    print("➡️ In df_plot:")
    print(df_plot[(df_plot['Stenosis Percentage'] == 0.0) & (np.isclose(df_plot['R_total'], 0.24))][
              ['R_total', 'R_micro', 'R_micro_nonzero', y_var]])

    # Add the special shared point (R_total ≈ 0.24, R_micro = 0) to both groups
    special_point = df_plot[(df_plot['R_micro'] == 0) & (np.isclose(df_plot['R_total'], 0.24))]
    if not special_point.empty:
        duplicate = special_point.copy()
        duplicate['R_micro_nonzero'] = True
        df_plot = pd.concat([df_plot, duplicate], ignore_index=True)

    # Begin plotting
    plt.figure(figsize=(6, 4))

    # unique_stenoses = sorted(df_plot['Stenosis Percentage'].unique())
    # marker_map = {
    #     unique_stenoses[0]: 'o',
    #     unique_stenoses[1]: '^'
    # }

    unique_stenoses = reversed(sorted(df_plot['Stenosis Percentage'].unique()))
    marker_styles = ['o', '^', 's']

    # Build marker map dynamically
    marker_map = {
        sten: marker_styles[i] if i < len(marker_styles) else 'o'
        for i, sten in enumerate(unique_stenoses)
    }

    groups = df_plot.groupby(['Stenosis Percentage', 'Length', 'R_micro_nonzero'])
    legend_elements = {}

    for (sten_val, len_val, is_nonzero), gdf in groups:
        gdf_sorted = gdf.sort_values(by='R_total')

        linestyle = '--' if is_nonzero else '-'
        marker = marker_map.get(sten_val, 'o')
        label = f"Sten.:{round(sten_val*100,-1):.0f}%, {'New (R_m)' if is_nonzero else 'Old'}"
        label_key = (sten_val, is_nonzero)

        if label_key not in legend_elements:
            legend_elements[label_key] = Line2D(
                [0], [0],
                marker=marker,
                color='black',
                linestyle=linestyle,
                markersize=7,
                label=label,
                markerfacecolor='none' if marker == 'o' else 'black',
                markeredgecolor='black'
            )

        plt.plot(
            gdf_sorted['R_total'],
            gdf_sorted[y_var],
            linestyle=linestyle,
            color='black',
            alpha=0.8
        )

        # plt.scatter(
        #     gdf_sorted['R_total'],
        #     gdf_sorted[y_var],
        #     marker=marker,
        #     color='black',
        #     edgecolor='k',
        #     s=60
        # )

        # Use hollow circle if marker is 'o'
        if marker == 'o':
            plt.scatter(
                gdf_sorted['R_total'],
                gdf_sorted[y_var],
                marker=marker,
                facecolors='none',
                edgecolors='black',
                s=60
            )
        else:
            plt.scatter(
                gdf_sorted['R_total'],
                gdf_sorted[y_var],
                marker=marker,
                color='black',
                edgecolor='k',
                s=60
            )

    plt.xlabel('R_total')
    plt.ylabel(f'{y_var}')
    # plt.title(f'{y_var} vs R_total')
    plt.xticks([0.24, 0.43, 0.62, 0.81])
    # plt.legend(handles=list(legend_elements.values()), frameon=False, loc='best')
    # Sort legend by stenosis %, then R_micro_nonzero (False before True)
    sorted_legend_keys = sorted(
        legend_elements.keys(),
        key=lambda k: (k[0], k[1])  # k[0]: stenosis %, k[1]: is_nonzero
    )
    sorted_legend_handles = [legend_elements[k] for k in sorted_legend_keys]

    plt.legend(handles=sorted_legend_handles, frameon=False, loc='best')

    plt.grid(False)
    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN EXECUTION
# -----------------------------
summary_file = './summary2.csv'
df = pd.read_csv(summary_file)

# Convert columns to numeric
for col in ['R_micro', 'R_total', 'Stenosis Percentage', 'Length',
            'WSS_TE_min', 'WSS_LE_min', 'WSS_LE_Area', 'WSS_TE_Area']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['Stenosis Percentage'] = df['Stenosis Percentage'].round(2)
df['Length'] = df['Length'].round(2)  # Add this line

# Normalize WSS_TE_min and WSS_LE_min by WSS_LMB
to_normalize = ["WSS_TE_min", "WSS_LE_min"]

# Ensure WSS_LMB is numeric
if 'WSS_LMB' in df.columns:
    df['WSS_LMB'] = pd.to_numeric(df['WSS_LMB'], errors='coerce')

    for col in to_normalize:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df.apply(
                lambda row: row[col] / row['WSS_LMB']
                if pd.notna(row[col]) and pd.notna(row['WSS_LMB']) and row['WSS_LMB'] != 0 else np.nan,
                axis=1
            )


# Filter to valid (Stenosis, Length) combos where R_micro != 0
df_nonzero = df[(df['R_micro'].notna()) & (df['R_micro'] != 0) & (df['Stenosis Percentage'] > -0.1)]
valid_geoms = set(zip(df_nonzero['Stenosis Percentage'], df_nonzero['Length']))

# Keep only those geometries and 'Hyperemic' condition
df_filtered = df[df.apply(
    lambda row: (row['Stenosis Percentage'], row['Length']) in valid_geoms and row['Condition'] == 'Hyperemic',
    axis=1
)].copy()

# -----------------------------
# LOOP OVER METRICS
# -----------------------------
# metrics_to_plot = ['WSS_TE_min', 'WSS_LE_min', 'WSS_LE_Area', 'WSS_TE_Area']
metrics_to_plot = ['CFR']

for metric in metrics_to_plot:
    plot_metric(df_filtered, metric)

print(df[(df['Stenosis Percentage'] == 0.0) & (df['R_micro'] != 0)][['Stenosis Percentage', 'Length', 'R_total']])
print(valid_geoms)
df_check = df[
    (df['Stenosis Percentage'] == 0.0) &
    (df['Length'] == 0.1) &
    (np.isclose(df['R_total'], 0.24))
]

print(df_check[['Stenosis Percentage', 'Length', 'R_micro', 'R_total', 'Condition']])
print(df[(df['Stenosis Percentage'] == 0.0) & (np.isclose(df['R_total'], 0.24))][['WSS_LMB', 'WSS_TE_min']])

