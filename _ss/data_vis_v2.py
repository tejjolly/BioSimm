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

# Filter out rows where 'CFR' and 'P_d/P_a' are NaN
df_filtered = df[df['CFR'].notna() & df['P_d/P_a'].notna()]

# Create a mask for data points where 'BMR/HMR' exists
mask = df_filtered['BMR/HMR'].notna()

# Define the colormap (from red to green)
cmap = cm.get_cmap('RdYlGn')

# Define boundaries for five levels between 1 and 3.5
boundaries = np.linspace(1, 3.5, 6)  # 6 boundaries create 5 intervals
norm = colors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

# Create the scatter plot
plt.figure(figsize=(12, 8))

# Swap axes: x-axis is 'P_d/P_a' (FFR), y-axis is 'CFR'
# Plot data points where 'BMR/HMR' exists, colored by 'BMR/HMR' value
scatter = plt.scatter(df_filtered['P_d/P_a'][mask], df_filtered['CFR'][mask],
                      c=df_filtered['BMR/HMR'][mask], cmap=cmap, norm=norm, edgecolor='k', alpha=0.7)

# Plot data points where 'BMR/HMR' doesn't exist, in black
plt.scatter(df_filtered['P_d/P_a'][~mask], df_filtered['CFR'][~mask],
            color='black', edgecolor='k', alpha=0.7)

# Now, add lines connecting points that match in 'Length' and 'Stenosis Percentage'

# Group the data by 'Length' and 'Stenosis Percentage'
grouped = df_filtered.groupby(['Length', 'Stenosis Percentage'])

# For each group, plot lines connecting the points and add labels
for name, group in grouped:
    if len(group) > 1:
        # Unpack the group name
        length_val, stenosis_val = name
        # Format 'Stenosis Percentage' to two decimal places
        stenosis_formatted = f"{stenosis_val:.2f}"
        # Sort the group by 'P_d/P_a' or another variable
        group = group.sort_values(by='P_d/P_a')
        # Plot the line connecting the points
        plt.plot(group['P_d/P_a'], group['CFR'], linestyle='-', color='gray', alpha=0.9)
        # Get the mean position to place the line label
        mean_ffr = group['P_d/P_a'].mean()
        mean_cfr = group['CFR'].mean()
        # Add the line label
        plt.text(mean_ffr, mean_cfr, f'S:{stenosis_formatted}%, L:{length_val}',
                 fontsize=10, color='grey', ha='center', va='center')
        # Add data point labels for Rtotal_cor Value
        for idx, row in group.iterrows():
            plt.text(row['P_d/P_a'], row['CFR'], f'R:{row["Rtotal_cor Value"]}',
                     fontsize=11, ha='right', va='bottom')

# Add colorbar for the 'BMR/HMR' values
cbar = plt.colorbar(scatter, ticks=boundaries)
cbar.set_label('BMR/HMR')

# Adjust color bar ticks to display at the boundaries
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Ensure five ticks

# Add labels and title
plt.xlabel('FFR (P_d/P_a)')
plt.ylabel('CFR')
plt.title('Scatter Plot of CFR vs FFR Colored by BMR/HMR')

# Optional: Add gridlines
plt.grid(True)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Show the plot
plt.show()
