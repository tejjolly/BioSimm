import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

# Path to the summary.csv file
summary_file = '/Users/tejjolly/Documents/BioSimm/Simulations/summary.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(summary_file)

# Convert columns to numeric, coercing errors to NaN
df['CFR'] = pd.to_numeric(df['CFR'], errors='coerce')
df['P_d/P_a'] = pd.to_numeric(df['P_d/P_a'], errors='coerce')
df['BMR/HMR'] = pd.to_numeric(df['BMR/HMR'], errors='coerce')
df['Rtotal_cor Value'] = pd.to_numeric(df['Rtotal_cor Value'], errors='coerce')
df['Stenosis Percentage'] = pd.to_numeric(df['Stenosis Percentage'], errors='coerce')
df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
df['HMR'] = pd.to_numeric(df['HMR'], errors='coerce')
df['HSR'] = pd.to_numeric(df['HSR'], errors='coerce')

# -----------------------------
# First Plot: CFR vs FFR (P_d/P_a)
# -----------------------------

# Filter out rows where 'CFR' and 'P_d/P_a' are NaN
# Since 'CFR' is calculated only for FFR runs, iFR runs are not included
df_filtered_cfr = df[df['CFR'].notna() & df['P_d/P_a'].notna()]

# Create a mask for data points where 'BMR/HMR' exists
mask_cfr = df_filtered_cfr['BMR/HMR'].notna()

# Define the colormap (from red to green)
cmap_cfr = cm.get_cmap('RdYlGn')

# Define boundaries for five levels between 1 and 3.5
boundaries_cfr = np.linspace(1, 3.5, 6)  # 6 boundaries create 5 intervals
norm_cfr = colors.BoundaryNorm(boundaries_cfr, ncolors=cmap_cfr.N, clip=True)

# Create the scatter plot
plt.figure(figsize=(12, 8))

# Swap axes: x-axis is 'P_d/P_a' (FFR), y-axis is 'CFR'
# Plot data points where 'BMR/HMR' exists, colored by 'BMR/HMR' value
scatter_cfr = plt.scatter(df_filtered_cfr['P_d/P_a'][mask_cfr], df_filtered_cfr['CFR'][mask_cfr],
                          c=df_filtered_cfr['BMR/HMR'][mask_cfr], cmap=cmap_cfr, norm=norm_cfr, edgecolor='k', alpha=0.7)

# Plot data points where 'BMR/HMR' doesn't exist, in black
plt.scatter(df_filtered_cfr['P_d/P_a'][~mask_cfr], df_filtered_cfr['CFR'][~mask_cfr],
            color='black', edgecolor='k', alpha=0.7)

# Group the data by 'Length' and 'Stenosis Percentage'
grouped_cfr = df_filtered_cfr.groupby(['Length', 'Stenosis Percentage'])

i = 0
# For each group, plot lines connecting the points and add labels
for name, group in grouped_cfr:
    if len(group) > 1:
        i += 1
        print('i',i)
        # Unpack the group name
        length_val, stenosis_val = name
        # Format 'Stenosis Percentage' to two decimal places
        stenosis_formatted = f"{stenosis_val:.2f}"
        # Sort the group by 'P_d/P_a' or another variable
        group = group.sort_values(by='P_d/P_a')
        # Plot the line connecting the points
        plt.plot(group['P_d/P_a'], group['CFR'], linestyle='-', color='gray', alpha=0.5)
        # Get the mean position to place the line label
        mean_ffr = group['P_d/P_a'].mean()
        mean_cfr = group['CFR'].mean()
        # Add the line label
        if i == 3:
            print('entered')
            plt.text(mean_ffr-.03, mean_cfr, f'S:{stenosis_formatted}%',
                     fontsize=10, color='gray', ha='center', va='center')
        else:
            plt.text(mean_ffr-.02, mean_cfr, f'S:{stenosis_formatted}%',
                     fontsize=10, color='gray', ha='center', va='center')
        # Add data point labels for Rtotal_cor Value
        for idx, row in group.iterrows():
            # print('idx',{idx})
            # print(f'R:{row["Rtotal_cor Value"]}')
            # print(row['P_d/P_a'])

            if idx == 4:
                plt.text(row['P_d/P_a']+.01, row['CFR']-.1, f'R:{row["Rtotal_cor Value"]}',
                         fontsize=11, ha='right', va='bottom')
            else:
                plt.text(row['P_d/P_a']-.005, row['CFR']-.05, f'R:{row["Rtotal_cor Value"]}',
                         fontsize=11, ha='right', va='bottom')

# Add colorbar for the 'BMR/HMR' values
cbar_cfr = plt.colorbar(scatter_cfr, ticks=boundaries_cfr)
cbar_cfr.set_label('BMR/HMR')

# Adjust color bar ticks to display at the boundaries
cbar_cfr.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Ensure five ticks

# Add labels and title
plt.xlabel('FFR')
plt.ylabel('CFR')
plt.title('CFR vs FFR, Colored by BMR/HMR')

# plt.grid(True)
plt.grid(False)
plt.axhline(y=2.0, color='gray', linestyle='--', linewidth=0.5)  # Horizontal gridline at y=2.0
plt.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5)  # Vertical gridline at x=0.8


# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the first plot
plt.show()

# -----------------------------
# Second Plot: HSR vs HMR, colored by FFR (P_d/P_a)
# -----------------------------

# Filter out rows where 'HMR', 'HSR', or 'P_d/P_a' are NaN
# Include only FFR runs (Condition == 'Hyperemic')
df_filtered_hmr = df[(df['Condition'] == 'Hyperemic') & df['HMR'].notna() & df['HSR'].notna() & df['P_d/P_a'].notna()]

# Create the figure for the second plot
plt.figure(figsize=(12, 8))

# Define the colormap (from red to green)
cmap_hmr = cm.get_cmap('RdYlGn')

# Define boundaries for five levels between 0.65 and 0.9
boundaries_hmr = np.linspace(0.65, 0.9, 6)  # 6 boundaries create 5 intervals
norm_hmr = colors.BoundaryNorm(boundaries_hmr, ncolors=cmap_hmr.N, clip=True)

# Plot data points, colored by 'P_d/P_a' (FFR)
scatter_hmr = plt.scatter(df_filtered_hmr['HMR'], df_filtered_hmr['HSR'],
                          c=df_filtered_hmr['P_d/P_a'], cmap=cmap_hmr, norm=norm_hmr, edgecolor='k', alpha=0.7)

# Group the data by 'Length' and 'Stenosis Percentage'
grouped_hmr = df_filtered_hmr.groupby(['Length', 'Stenosis Percentage'])

# For each group, check if there are multiple Rtotal_cor Values
for name, group in grouped_hmr:
    unique_r_values = group['Rtotal_cor Value'].unique()
    if len(unique_r_values) > 1:
        # Group has multiple Rtotal_cor Values (varying R values)
        # Unpack the group name
        length_val, stenosis_val = name
        # Format 'Stenosis Percentage' to two decimal places
        stenosis_formatted = f"{stenosis_val:.2f}"
        # Sort the group by 'Rtotal_cor Value' or another variable
        group = group.sort_values(by='Rtotal_cor Value')
        # Plot the line connecting the points
        plt.plot(group['HMR'], group['HSR'], linestyle='-', color='gray', alpha=0.5)
        # Optionally, add line labels if desired
        # Get the mean position to place the line label
        mean_hmr = group['HMR'].mean()
        mean_hsr = group['HSR'].mean()
        plt.text(mean_hmr, mean_hsr, f'S:{stenosis_formatted}%',
                 fontsize=10, color='gray', ha='center', va='center')
        # Add data point labels for Rtotal_cor Value
        for idx, row in group.iterrows():
            plt.text(row['HMR'], row['HSR'], f'R:{row["Rtotal_cor Value"]}',
                     fontsize=11, ha='right', va='bottom')

# Add colorbar for the 'P_d/P_a' values
cbar_hmr = plt.colorbar(scatter_hmr, ticks=boundaries_hmr)
cbar_hmr.set_label('FFR')

# Add labels and title
plt.xlabel('HMR')
plt.ylabel('HSR')
plt.title('HSR vs HMR, Colored by FFR')

# Optional: Add gridlines
plt.grid(False)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the second plot
plt.show()

# -----------------------------
# Third Plot: FFR vs HMR, Colored by HSR
# -----------------------------

# Filter out rows where 'HMR', 'HSR', or 'P_d/P_a' are NaN
df_filtered_third = df[(df['Condition'] == 'Hyperemic') & df['HMR'].notna() & df['HSR'].notna() & df['P_d/P_a'].notna()]

# Define the colormap (from red to green)
cmap_third = cm.get_cmap('RdYlGn_r')

# Define boundaries for 'HSR' to create 5 bins
hsr_min = df_filtered_third['HSR'].min()
hsr_max = df_filtered_third['HSR'].max()
boundaries_third = np.linspace(hsr_min, hsr_max, 6)  # 6 boundaries create 5 intervals
norm_third = colors.BoundaryNorm(boundaries_third, ncolors=cmap_third.N, clip=True)

# Create the figure for the third plot
plt.figure(figsize=(12, 8))

# Plot data points, colored by 'HSR'
scatter_third = plt.scatter(df_filtered_third['HMR'], df_filtered_third['P_d/P_a'],
                            c=df_filtered_third['HSR'], cmap=cmap_third, norm=norm_third, edgecolor='k', alpha=0.7)

# Group the data by 'Length' and 'Stenosis Percentage'
grouped_third = df_filtered_third.groupby(['Length', 'Stenosis Percentage'])

# For each group, check if there are multiple Rtotal_cor Values
for name, group in grouped_third:
    unique_r_values = group['Rtotal_cor Value'].unique()
    if len(unique_r_values) > 1:
        length_val, stenosis_val = name
        stenosis_formatted = f"{stenosis_val:.2f}"
        group = group.sort_values(by='Rtotal_cor Value')
        plt.plot(group['HMR'], group['P_d/P_a'], linestyle='-', color='gray', alpha=0.5)
        mean_hmr = group['HMR'].mean()
        mean_ffr = group['P_d/P_a'].mean()
        plt.text(mean_hmr, mean_ffr, f'S:{stenosis_formatted}%',
                 fontsize=10, color='gray', ha='center', va='center')
        for idx, row in group.iterrows():
            plt.text(row['HMR'], row['P_d/P_a'], f'R:{row["Rtotal_cor Value"]}',
                     fontsize=11, ha='right', va='bottom')

# Add colorbar for the 'HSR' values
cbar_third = plt.colorbar(scatter_third, ticks=boundaries_third)
cbar_third.set_label('HSR')
cbar_third.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Ensure five ticks

# Add labels and title
plt.xlabel('HMR')
plt.ylabel('FFR')
plt.title('FFR vs HMR, Colored by HSR')

plt.grid(False)
plt.tight_layout()
plt.show()

# -----------------------------
# Fourth Plot: FFR vs HSR, Colored by HMR
# -----------------------------

df_filtered_fourth = df_filtered_third  # Reuse the same DataFrame

# Define the colormap (from red to green)
cmap_fourth = cm.get_cmap('RdYlGn_r')

# Define boundaries for 'HMR' to create 5 bins
hmr_min = df_filtered_fourth['HMR'].min()
hmr_max = df_filtered_fourth['HMR'].max()
boundaries_fourth = np.linspace(hmr_min, hmr_max, 6)
norm_fourth = colors.BoundaryNorm(boundaries_fourth, ncolors=cmap_fourth.N, clip=True)

# Create the figure for the fourth plot
plt.figure(figsize=(12, 8))

# Plot data points, colored by 'HMR'
scatter_fourth = plt.scatter(df_filtered_fourth['HSR'], df_filtered_fourth['P_d/P_a'],
                             c=df_filtered_fourth['HMR'], cmap=cmap_fourth, norm=norm_fourth, edgecolor='k', alpha=0.7)

# Group the data by 'Length' and 'Stenosis Percentage'
grouped_fourth = df_filtered_fourth.groupby(['Length', 'Stenosis Percentage'])

# For each group, check if there are multiple Rtotal_cor Values
for name, group in grouped_fourth:
    unique_r_values = group['Rtotal_cor Value'].unique()
    if len(unique_r_values) > 1:
        length_val, stenosis_val = name
        stenosis_formatted = f"{stenosis_val:.2f}"
        group = group.sort_values(by='Rtotal_cor Value')
        plt.plot(group['HSR'], group['P_d/P_a'], linestyle='-', color='gray', alpha=0.5)
        mean_hsr = group['HSR'].mean()
        mean_ffr = group['P_d/P_a'].mean()
        plt.text(mean_hsr, mean_ffr, f'S:{stenosis_formatted}%',
                 fontsize=10, color='gray', ha='center', va='center')
        for idx, row in group.iterrows():
            plt.text(row['HSR'], row['P_d/P_a'], f'R:{row["Rtotal_cor Value"]}',
                     fontsize=11, ha='right', va='bottom')

# Add colorbar for the 'HMR' values
cbar_fourth = plt.colorbar(scatter_fourth, ticks=boundaries_fourth)
cbar_fourth.set_label('HMR')
cbar_fourth.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Ensure five ticks

# Add labels and title
plt.xlabel('HSR')
plt.ylabel('FFR')
plt.title('FFR vs HSR, Colored by HMR')

plt.grid(False)
plt.tight_layout()
plt.show()
