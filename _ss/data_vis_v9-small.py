import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# -----------------------------
# Global matplotlib settings (fonts, ticks, etc.) for smaller figures
# -----------------------------
plt.rcParams.update({
    'font.size': 12,            # Increase base font size
    'axes.labelsize': 14,       # Axis label font size
    'axes.titlesize': 14,       # Title font size
    'xtick.labelsize': 12,      # X-tick label font size
    'ytick.labelsize': 12,      # Y-tick label font size
    'legend.fontsize': 12,      # Legend font size
    'figure.dpi': 300           # Higher DPI for clearer text in smaller figure
})

# Path to the summary.csv file
summary_file = '/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary2.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(summary_file)

# Convert columns to numeric, coercing errors to NaN
df['CFR'] = pd.to_numeric(df['CFR'], errors='coerce')
df['P_d/P_a'] = pd.to_numeric(df['P_d/P_a'], errors='coerce')
df['BMR/HMR'] = pd.to_numeric(df['BMR/HMR'], errors='coerce')
df['R_total'] = pd.to_numeric(df['R_total'], errors='coerce')
df['Stenosis Percentage'] = pd.to_numeric(df['Stenosis Percentage'], errors='coerce')
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
df['HMR'] = pd.to_numeric(df['HMR'], errors='coerce')
df['HSR'] = pd.to_numeric(df['HSR'], errors='coerce')

# =========================================================
# 1) FIRST PLOT: CFR vs FFR (P_d/P_a), colored by BMR/HMR
# =========================================================

# Filter out rows where 'CFR' and 'P_d/P_a' are NaN
df_filtered_cfr = df[df['CFR'].notna() & df['P_d/P_a'].notna()]

# Create a mask for data points where 'BMR/HMR' exists
mask_cfr = df_filtered_cfr['BMR/HMR'].notna()

# Define the colormap (from red to green)
cmap_cfr = cm.get_cmap('RdYlGn')
boundaries_cfr = np.linspace(1, 3.5, 6)  # 5 intervals between 1 and 3.5
norm_cfr = colors.BoundaryNorm(boundaries_cfr, ncolors=cmap_cfr.N, clip=True)

plt.figure(figsize=(6, 4))  # Reduced figure size

# Scatter where BMR/HMR is present
scatter_cfr = plt.scatter(
    df_filtered_cfr['P_d/P_a'][mask_cfr],
    df_filtered_cfr['CFR'][mask_cfr],
    c=df_filtered_cfr['BMR/HMR'][mask_cfr],
    cmap=cmap_cfr,
    norm=norm_cfr,
    edgecolor='k',
    alpha=0.7,
    s=60
)

# Scatter where BMR/HMR not present
plt.scatter(
    df_filtered_cfr['P_d/P_a'][~mask_cfr],
    df_filtered_cfr['CFR'][~mask_cfr],
    color='black',
    edgecolor='k',
    alpha=0.7,
    s=60
)

# Group by 'Length' and 'Stenosis Percentage' — lines connecting points
grouped_cfr = df_filtered_cfr.groupby(['Length', 'Stenosis Percentage'])
for name, group in grouped_cfr:
    if len(group) > 1:
        group = group.sort_values(by='P_d/P_a')
        plt.plot(group['P_d/P_a'], group['CFR'], linestyle='-', color='gray', alpha=0.5)
        
        # -- Commented out S and R annotations --
        # mean_ffr = group['P_d/P_a'].mean()
        # mean_cfr = group['CFR'].mean()
        # plt.text(mean_ffr, mean_cfr, f"S:{stenosis_val:.2f}%", ...)
        # for idx, row in group.iterrows():
        #     plt.text(row['P_d/P_a'], row['CFR'], f"R:{row['R_total']}", ...)

# Add colorbar
cbar_cfr = plt.colorbar(scatter_cfr, ticks=boundaries_cfr)
cbar_cfr.set_label('BMR/HMR', fontsize=12)
cbar_cfr.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# Axis labels and title
plt.xlabel('FFR')
plt.ylabel('CFR')
plt.title('CFR vs FFR, Colored by BMR/HMR')

# Threshold lines (CFR=2, FFR=0.8)
plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.8)

plt.grid(False)
plt.tight_layout()
plt.show()


# =========================================================
# 2) SECOND PLOT: HSR vs HMR, colored by FFR (P_d/P_a)
# =========================================================
# NOTE: This plot does NOT have FFR or CFR on an axis.
#       Thus, no threshold lines are added here.

df_filtered_hmr = df[(df['Condition'] == 'Hyperemic') & 
                     df['HMR'].notna() & 
                     df['HSR'].notna() & 
                     df['P_d/P_a'].notna()]

plt.figure(figsize=(6, 4))

cmap_hmr = cm.get_cmap('RdYlGn')
boundaries_hmr = np.linspace(0.65, 0.9, 6)  # 5 intervals between 0.65 & 0.9
norm_hmr = colors.BoundaryNorm(boundaries_hmr, ncolors=cmap_hmr.N, clip=True)

scatter_hmr = plt.scatter(
    df_filtered_hmr['HMR'],
    df_filtered_hmr['HSR'],
    c=df_filtered_hmr['P_d/P_a'],
    cmap=cmap_hmr,
    norm=norm_hmr,
    edgecolor='k',
    alpha=0.7,
    s=60
)

grouped_hmr = df_filtered_hmr.groupby(['Length', 'Stenosis Percentage'])
for name, group in grouped_hmr:
    if len(group['R_total'].unique()) > 1:
        group = group.sort_values(by='R_total')
        plt.plot(group['HMR'], group['HSR'], linestyle='-', color='gray', alpha=0.5)

        # -- Commented out S and R annotations --
        # mean_hmr = group['HMR'].mean()
        # mean_hsr = group['HSR'].mean()
        # plt.text(mean_hmr, mean_hsr, f"S:{stenosis_val:.2f}%", ...)
        # for idx, row in group.iterrows():
        #     plt.text(row['HMR'], row['HSR'], f"R:{row['R_total']}", ...)

cbar_hmr = plt.colorbar(scatter_hmr, ticks=boundaries_hmr)
cbar_hmr.set_label('FFR')
cbar_hmr.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

plt.xlabel('HMR [mmHg/cm/s]')
plt.ylabel('HSR [mmHg/cm/s]')
plt.title('HSR vs HMR, Colored by FFR')
plt.grid(False)
plt.tight_layout()
plt.show()


# =========================================================
# 3) THIRD PLOT: FFR vs HMR, colored by HSR
# =========================================================
# y-axis is FFR => add a horizontal line at FFR = 0.8

df_filtered_third = df[(df['Condition'] == 'Hyperemic') & 
                       df['HMR'].notna() & 
                       df['HSR'].notna() & 
                       df['P_d/P_a'].notna()]

plt.figure(figsize=(6, 4))

cmap_third = cm.get_cmap('RdYlGn_r')
hsr_min = df_filtered_third['HSR'].min()
hsr_max = df_filtered_third['HSR'].max()
boundaries_third = np.linspace(hsr_min, hsr_max, 6)  # 5 intervals
norm_third = colors.BoundaryNorm(boundaries_third, ncolors=cmap_third.N, clip=True)

scatter_third = plt.scatter(
    df_filtered_third['HMR'],
    df_filtered_third['P_d/P_a'],
    c=df_filtered_third['HSR'],
    cmap=cmap_third,
    norm=norm_third,
    edgecolor='k',
    alpha=0.7,
    s=60
)

grouped_third = df_filtered_third.groupby(['Length', 'Stenosis Percentage'])
for name, group in grouped_third:
    if len(group['R_total'].unique()) > 1:
        group = group.sort_values(by='R_total')
        plt.plot(group['HMR'], group['P_d/P_a'], linestyle='-', color='gray', alpha=0.5)

        # -- Commented out S and R annotations --
        # mean_hmr = group['HMR'].mean()
        # mean_ffr = group['P_d/P_a'].mean()
        # plt.text(mean_hmr, mean_ffr, f"S:{stenosis_val:.2f}%", ...)
        # for idx, row in group.iterrows():
        #     plt.text(row['HMR'], row['P_d/P_a'], f"R:{row['R_total']}", ...)

# Threshold line for FFR=0.8
plt.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)

cbar_third = plt.colorbar(scatter_third, ticks=boundaries_third)
cbar_third.set_label('HSR [mmHg/cm/s]')
cbar_third.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

plt.xlabel('HMR [mmHg/cm/s]')
plt.ylabel('FFR')
plt.title('FFR vs HMR, Colored by HSR')
plt.grid(False)
plt.tight_layout()
plt.show()


# =========================================================
# 4) FOURTH PLOT: FFR vs HSR, colored by HMR
# =========================================================
# y-axis is FFR => add a horizontal line at FFR = 0.8

df_filtered_fourth = df_filtered_third  # same DataFrame filter

plt.figure(figsize=(6, 4))

cmap_fourth = cm.get_cmap('RdYlGn_r')
hmr_min = df_filtered_fourth['HMR'].min()
hmr_max = df_filtered_fourth['HMR'].max()
boundaries_fourth = np.linspace(hmr_min, hmr_max, 6)
norm_fourth = colors.BoundaryNorm(boundaries_fourth, ncolors=cmap_fourth.N, clip=True)

scatter_fourth = plt.scatter(
    df_filtered_fourth['HSR'],
    df_filtered_fourth['P_d/P_a'],
    c=df_filtered_fourth['HMR'],
    cmap=cmap_fourth,
    norm=norm_fourth,
    edgecolor='k',
    alpha=0.7,
    s=60
)

grouped_fourth = df_filtered_fourth.groupby(['Length', 'Stenosis Percentage'])
for name, group in grouped_fourth:
    if len(group['R_total'].unique()) > 1:
        group = group.sort_values(by='R_total')
        plt.plot(group['HSR'], group['P_d/P_a'], linestyle='-', color='gray', alpha=0.5)

        # -- Commented out S and R annotations --
        # mean_hsr = group['HSR'].mean()
        # mean_ffr = group['P_d/P_a'].mean()
        # plt.text(mean_hsr, mean_ffr, f"S:{stenosis_val:.2f}%", ...)
        # for idx, row in group.iterrows():
        #     plt.text(row['HSR'], row['P_d/P_a'], f"R:{row['R_total']}", ...)

# Threshold line for FFR=0.8
plt.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.8)

cbar_fourth = plt.colorbar(scatter_fourth, ticks=boundaries_fourth)
cbar_fourth.set_label('HMR [mmHg/cm/s]')
cbar_fourth.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

plt.xlabel('HSR [mmHg/cm/s]')
plt.ylabel('FFR')
plt.title('FFR vs HSR, Colored by HMR')
plt.grid(False)
plt.tight_layout()
plt.show()


# =========================================================
# 5) FIFTH PLOT: CFR vs HMR, colored by HSR
# =========================================================
# y-axis is CFR => add a horizontal line at CFR = 2.0

df_filtered_cfr_hmr = df[df['CFR'].notna() & df['HMR'].notna() & df['HSR'].notna()]

plt.figure(figsize=(6, 4))

cmap_cfr_hmr = cm.get_cmap('RdYlGn_r')
hsr_min_cfr_hmr = df_filtered_cfr_hmr['HSR'].min()
hsr_max_cfr_hmr = df_filtered_cfr_hmr['HSR'].max()
boundaries_cfr_hmr = np.linspace(hsr_min_cfr_hmr, hsr_max_cfr_hmr, 6)
norm_cfr_hmr = colors.BoundaryNorm(boundaries_cfr_hmr, ncolors=cmap_cfr_hmr.N, clip=True)

scatter_cfr_hmr = plt.scatter(
    df_filtered_cfr_hmr['HMR'],
    df_filtered_cfr_hmr['CFR'],
    c=df_filtered_cfr_hmr['HSR'],
    cmap=cmap_cfr_hmr,
    norm=norm_cfr_hmr,
    edgecolor='k',
    alpha=0.7,
    s=60
)

grouped_cfr_hmr = df_filtered_cfr_hmr.groupby(['Length', 'Stenosis Percentage'])
for name, group in grouped_cfr_hmr:
    if len(group['R_total'].unique()) > 1:
        group = group.sort_values(by='R_total')
        plt.plot(group['HMR'], group['CFR'], linestyle='-', color='gray', alpha=0.5)

        # -- Commented out S and R annotations --
        # first_point = group.iloc[0]
        # plt.text(first_point['HMR'], first_point['CFR'], f"S:{stenosis_val:.2f}%", ...)
        # for idx, row in group.iterrows():
        #     plt.text(row['HMR'], row['CFR'], f"R:{row['R_total']}", ...)

# Threshold line for CFR=2.0
plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.8)

cbar_cfr_hmr = plt.colorbar(scatter_cfr_hmr, ticks=boundaries_cfr_hmr)
cbar_cfr_hmr.set_label('HSR [mmHg/cm/s]')
cbar_cfr_hmr.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

plt.xlabel('HMR [mmHg/cm/s]')
plt.ylabel('CFR')
plt.title('CFR vs HMR, Colored by HSR')
plt.grid(False)
plt.tight_layout()
plt.show()
##################################################################################
df['BMR'] = df['BMR/HMR'] * df['HMR']
# 1) If you do NOT already have a 'BMR' column, create it from BMR/HMR * HMR
df['BMR'] = df['BMR/HMR'] * df['HMR']  # Skip if BMR already exists

# 2) Filter out invalid rows
df_hmr_bmr = df[df['BMR'].notna() & df['HMR'].notna() & df['HSR'].notna()]

# 3) Define figure & colormap
plt.figure(figsize=(6, 4))
cmap_hmr_bmr = cm.get_cmap('RdYlGn_r')

# Setup boundaries for HSR
hsr_min = df_hmr_bmr['HSR'].min()
hsr_max = df_hmr_bmr['HSR'].max()
boundaries_hmr_bmr = np.linspace(hsr_min, hsr_max, 6)  # 5 intervals
norm_hmr_bmr = colors.BoundaryNorm(boundaries_hmr_bmr, ncolors=cmap_hmr_bmr.N, clip=True)

# 4) Scatter plot
scatter_hmr_bmr = plt.scatter(
    df_hmr_bmr['HMR'],
    df_hmr_bmr['BMR'],
    c=df_hmr_bmr['HSR'],
    cmap=cmap_hmr_bmr,
    norm=norm_hmr_bmr,
    edgecolor='k',
    alpha=0.7,
    s=60
)

# 5) Colorbar
cbar_hmr_bmr = plt.colorbar(scatter_hmr_bmr, ticks=boundaries_hmr_bmr)
cbar_hmr_bmr.set_label('HSR [mmHg/cm/s]')
cbar_hmr_bmr.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# 6) Labels, title, etc.
plt.xlabel('HMR [mmHg/cm/s]')
plt.ylabel('BMR [mmHg/cm/s]')  # Adjust units/label as needed
plt.title('HMR vs. BMR, Colored by HSR')
plt.grid(False)
plt.tight_layout()
plt.show()

# 1) Assuming you've already created 'BMR' (or have it as a column)
df_hsr_bmr = df[df['BMR'].notna() & df['HSR'].notna() & df['HMR'].notna()]

# 2) Define figure & colormap
plt.figure(figsize=(6, 4))
cmap_hsr_bmr = cm.get_cmap('RdYlGn')

# Setup boundaries for HMR
hmr_min = df_hsr_bmr['HMR'].min()
hmr_max = df_hsr_bmr['HMR'].max()
boundaries_hsr_bmr = np.linspace(hmr_min, hmr_max, 6)  # 5 intervals
norm_hsr_bmr = colors.BoundaryNorm(boundaries_hsr_bmr, ncolors=cmap_hsr_bmr.N, clip=True)

# 3) Scatter plot
scatter_hsr_bmr = plt.scatter(
    df_hsr_bmr['HSR'],
    df_hsr_bmr['BMR'],
    c=df_hsr_bmr['HMR'],
    cmap=cmap_hsr_bmr,
    norm=norm_hsr_bmr,
    edgecolor='k',
    alpha=0.7,
    s=60
)

# 4) Colorbar
cbar_hsr_bmr = plt.colorbar(scatter_hsr_bmr, ticks=boundaries_hsr_bmr)
cbar_hsr_bmr.set_label('HMR [mmHg/cm/s]')
cbar_hsr_bmr.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# 5) Labels, title, etc.
plt.xlabel('HSR [mmHg/cm/s]')
plt.ylabel('BMR [mmHg/cm/s]')  # Adjust units/label as needed
plt.title('HSR vs. BMR, Colored by HMR')
plt.grid(False)
plt.tight_layout()
plt.show()



