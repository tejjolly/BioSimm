import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# User-defined directories (iFR first, FFR second)
ifr_base_dir = "/Users/tejjolly/Documents/BioSimm/Simulations/master_ifr"
ffr_base_dir = "/Users/tejjolly/Documents/BioSimm/Simulations/master_ffr"

# Ensure the output directory exists
output_dir = "./tables"
os.makedirs(output_dir, exist_ok=True)

# Function to extract values from CSVs for a given base directory and source (iFR or FFR)
def extract_values(geometry_n, base_dir, source):
    base_path = os.path.join(base_dir, f"Geometry_{geometry_n}")
    extracted_data = {}

    # 1. Extract length from plaque_1_details.csv
    plaque_details_path = os.path.join(base_path, "measurements", "plaque_1_details.csv")
    if not os.path.exists(plaque_details_path):
        print(f"Geometry {geometry_n}: Missing plaque_1_details.csv in {plaque_details_path}")
        return None
    plaque_details_df = pd.read_csv(plaque_details_path)
    length = round(plaque_details_df.iloc[0, 0], 1)  # Round length to 1 decimal place

    # 2. Extract narrowing from stenosis_acc_1.csv
    stenosis_acc_path = os.path.join(base_path, "measurements", "stenosis_acc_1.csv")
    if not os.path.exists(stenosis_acc_path):
        print(f"Geometry {geometry_n}: Missing stenosis_acc_1.csv in {stenosis_acc_path}")
        return None
    with open(stenosis_acc_path, 'r') as file:
        narrowing = round(float(file.read().strip()), 2)  # Round narrowing to 2 decimal places

    # Storing extracted values
    extracted_data['length'] = length
    extracted_data['narrowing'] = narrowing
    extracted_data['geometry'] = geometry_n
    extracted_data['source'] = source

    # 3. Extract ffr or ifr from plaque_lad_ffr.csv
    plaque_lad_ffr_path = os.path.join(base_path, "results-processed-new", "plaque_lad_ffr.csv")
    if not os.path.exists(plaque_lad_ffr_path):
        print(f"Geometry {geometry_n}: Missing plaque_lad_ffr.csv in {plaque_lad_ffr_path}")
        return None
    plaque_lad_ffr_df = pd.read_csv(plaque_lad_ffr_path)
    value = float(plaque_lad_ffr_df.iloc[0, 0])  # Ensure value is a float
    if source == 'ifr':
        extracted_data['ifr'] = value
    elif source == 'ffr':
        extracted_data['ffr'] = value

    # 4. Extract HMR and HSR from plaque_lad_HMR.csv if it exists (only for FFR data)
    if source == 'ffr':
        plaque_lad_HMR_path = os.path.join(base_path, "results-processed-new", "plaque_lad_HMR.csv")
        if os.path.exists(plaque_lad_HMR_path):
            plaque_lad_HMR_df = pd.read_csv(plaque_lad_HMR_path)
            hmr = float(plaque_lad_HMR_df.iloc[0, 0])  # Ensure hmr is a float
            hsr = float(plaque_lad_HMR_df.iloc[1, 0])  # Ensure hsr is a float
            extracted_data['hmr'] = hmr
            extracted_data['hsr'] = hsr
        else:
            extracted_data['hmr'] = np.nan
            extracted_data['hsr'] = np.nan

    # 5. Extract average flow from all_results-flows.txt
    flow_path = os.path.join(base_path, "results-processed-new", "all_results-flows.txt")
    if not os.path.exists(flow_path):
        print(f"Geometry {geometry_n}: Missing all_results-flows.txt in {flow_path}")
        extracted_data['flow'] = np.nan
    else:
        try:
            with open(flow_path, 'r') as file:
                lines = file.readlines()
                # Remove empty lines and strip whitespace
                data_lines = [line.strip() for line in lines if line.strip()]
                # If first line is header, skip it
                if data_lines[0].startswith('inlet_'):
                    data_lines = data_lines[1:]
                values = []
                for line in data_lines:
                    # Split the line by hyphens
                    parts = line.split('-')
                    for part in parts:
                        # Attempt to convert each part to a float
                        try:
                            num = float(part)
                            values.append(num)
                        except ValueError:
                            # Ignore parts that cannot be converted to float
                            continue
                if values:
                    flow_avg = np.sum(values)
                    extracted_data['flow'] = flow_avg
                else:
                    print(f"Geometry {geometry_n}: No valid flow data found in {flow_path}")
                    extracted_data['flow'] = np.nan
        except Exception as e:
            print(f"Geometry {geometry_n}: Error reading flow data from {flow_path}: {e}")
            extracted_data['flow'] = np.nan

    return extracted_data

# Function to extract data for a list of geometries from a given base directory and source
def extract_dataset(geometry_numbers, base_dir, source):
    data_list = []
    for n in geometry_numbers:
        data = extract_values(n, base_dir, source)
        if data is not None:
            data_list.append(data)
    df = pd.DataFrame(data_list)
    return df

# Specify geometry numbers (iFR first, FFR second)
geometry_numbers_ifr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 25, 26]#, 27, 28, 29]
geometry_numbers_ffr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 25, 26]#, 27, 28, 29]
# geometry_numbers_ifr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33]
# geometry_numbers_ffr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33]
# geometry_numbers_ifr = [18, 31, 32, 33]
# geometry_numbers_ffr = [18, 31, 32, 33]

# Extract datasets (iFR first)
ifr_data = extract_dataset(geometry_numbers_ifr, ifr_base_dir, source='ifr')
ffr_data = extract_dataset(geometry_numbers_ffr, ffr_base_dir, source='ffr')

# Combine the datasets (iFR data first)
all_data = pd.concat([ifr_data, ffr_data], ignore_index=True)

# Define the function to create tables and plot them
def create_tables(all_data):
    # Define initial narrowing and length values
    initial_narrowing_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.62]
    length_values = [0.9, 1.2, 1.5, 1.7, 2.0, 2.5]

    # Extract narrowing values from data
    narrowing_values = list(set(all_data['narrowing'].tolist() + initial_narrowing_values))
    narrowing_values = sorted(narrowing_values)

    # Initialize tables
    ifr_table = pd.DataFrame(np.nan, index=narrowing_values, columns=length_values)
    ffr_table = pd.DataFrame(np.nan, index=narrowing_values, columns=length_values)
    cfr_table = pd.DataFrame(np.nan, index=narrowing_values, columns=length_values)
    hmr_table = pd.DataFrame(np.nan, index=narrowing_values, columns=length_values)
    hsr_table = pd.DataFrame(np.nan, index=narrowing_values, columns=length_values)

    # Define previous values to populate tables with the provided arrays
    ifr_values = [
        [0.99, 0.99, 0.98, 0.98],
        [0.99, 0.98, 0.98, 0.98],
        [0.99, 0.97, 0.98, 0.97],
        [0.98, 0.97, 0.96, 0.96],
        [0.95, 0.91, 0.90, 0.88],
        [0.88, 0.79, 0.77, 0.74]
    ]
    hsr_values = [
        0.09, 0.14, 0.14, 0.18,
        0.11, 0.17, 0.17, 0.21,
        0.14, 0.29, 0.24, 0.33,
        0.19, 0.38, 0.40, 0.45,
        0.59, 0.76, 0.80, 0.93,
        1.11, 1.38, 1.45, 1.56
    ]
    hmr_values = [
        2.24, 2.24, 2.17, 2.05,
        2.36, 2.30, 2.21, 2.08,
        2.35, 2.24, 2.22, 2.00,
        2.45, 2.17, 2.07, 1.99,
        1.96, 1.61, 1.55, 1.52,
        1.73, 1.30, 1.25, 1.18
    ]
    ffr_values = [
        0.96, 0.94, 0.94, 0.92,
        0.96, 0.93, 0.93, 0.91,
        0.94, 0.89, 0.90, 0.86,
        0.93, 0.85, 0.84, 0.82,
        0.77, 0.68, 0.66, 0.62,
        0.61, 0.49, 0.46, 0.43
    ]
    length = [0.9, 1.7, 2.0, 2.5] * 6
    stenosis = [0.10] * 4 + [0.20] * 4 + [0.30] * 4 + [0.40] * 4 + [0.50] * 4 + [0.62] * 4
    cfr_values = [
        3.379, 3.384, 3.373, 3.371, 3.379, 3.363, 3.352, 3.361,
        3.357, 3.300, 3.324, 3.275, 3.342, 3.247, 3.231, 3.194,
        3.099, 2.933, 2.855, 2.744, 2.802, 2.541, 2.490, 2.423
    ]

    # Create flag DataFrames to track extracted data
    is_extracted_flag = {
        'ifr': pd.DataFrame(False, index=narrowing_values, columns=length_values),
        'ffr': pd.DataFrame(False, index=narrowing_values, columns=length_values),
        'cfr': pd.DataFrame(False, index=narrowing_values, columns=length_values),
        'hmr': pd.DataFrame(False, index=narrowing_values, columns=length_values),
        'hsr': pd.DataFrame(False, index=narrowing_values, columns=length_values)
    }

    # Create DataFrames to track the Geometry numbers for each cell
    geometry_labels_ifr = pd.DataFrame('', index=narrowing_values, columns=length_values)
    geometry_labels_ffr = pd.DataFrame('', index=narrowing_values, columns=length_values)
    geometry_labels_hmr = pd.DataFrame('', index=narrowing_values, columns=length_values)
    geometry_labels_hsr = pd.DataFrame('', index=narrowing_values, columns=length_values)

    # Prepare dictionaries to store flow values per geometry
    flow_ffr = {}
    flow_ifr = {}

    # Prepare a list to collect data for scatter plots
    scatter_data = []

    # Initialize previous geometry counter
    prev_geometry_counter = 100

    # Populate previous values into the tables and collect scatter data
    for i, narrowing in enumerate(stenosis):
        len_value = length[i]
        if narrowing in narrowing_values and len_value in length_values:
            # Existing code to populate tables
            ifr_value = ifr_values[i // 4][i % 4]
            ffr_value = ffr_values[i]
            hmr_value = hmr_values[i]
            hsr_value = hsr_values[i]
            cfr_value = cfr_values[i]

            ifr_table.loc[narrowing, len_value] = ifr_value
            ffr_table.loc[narrowing, len_value] = ffr_value
            hmr_table.loc[narrowing, len_value] = hmr_value
            hsr_table.loc[narrowing, len_value] = hsr_value
            cfr_table.loc[narrowing, len_value] = cfr_value

            # Set flags
            is_extracted_flag['ifr'].loc[narrowing, len_value] = True
            is_extracted_flag['ffr'].loc[narrowing, len_value] = True
            is_extracted_flag['hmr'].loc[narrowing, len_value] = True
            is_extracted_flag['hsr'].loc[narrowing, len_value] = True
            is_extracted_flag['cfr'].loc[narrowing, len_value] = True

            # Create a geometry label
            # geometry_label = f'prev{prev_geometry_counter}'
            geometry_label = ''


            # Update geometry labels
            geometry_labels_ifr.loc[narrowing, len_value] += geometry_label
            geometry_labels_ffr.loc[narrowing, len_value] += geometry_label
            geometry_labels_hmr.loc[narrowing, len_value] += geometry_label
            geometry_labels_hsr.loc[narrowing, len_value] += geometry_label

            # Collect data into scatter_data
            if pd.notna(hmr_value) and pd.notna(hsr_value):
                scatter_data.append({
                    'geometry': geometry_label,
                    'hmr': hmr_value,
                    'ffr': ffr_value,
                    'ifr': ifr_value,
                    'cfr': cfr_value,
                    'hsr': hsr_value
                })

            # Increment the counter
            prev_geometry_counter +=1

    # Now, process the extracted data and collect scatter data as before
    for idx, row in all_data.iterrows():
        narrowing = row['narrowing']
        length = row['length']
        geometry = row['geometry']
        source = row['source']

        key = (narrowing, length)
        geometry_label = f'#{geometry}'

        if narrowing in narrowing_values and length in length_values:
            if source == 'ifr':
                ifr = row.get('ifr', np.nan)
                flow = row.get('flow', np.nan)
                if not pd.isna(ifr):
                    ifr_table.loc[narrowing, length] = ifr
                    is_extracted_flag['ifr'].loc[narrowing, length] = True
                    geometry_labels_ifr.loc[narrowing, length] += geometry_label
                if not pd.isna(flow):
                    flow_ifr[geometry] = flow  # Store flow per geometry
            elif source == 'ffr':
                ffr = row.get('ffr', np.nan)
                hmr = row.get('hmr', np.nan)
                hsr = row.get('hsr', np.nan)
                flow = row.get('flow', np.nan)
                if not pd.isna(ffr):
                    ffr_table.loc[narrowing, length] = ffr
                    is_extracted_flag['ffr'].loc[narrowing, length] = True
                    geometry_labels_ffr.loc[narrowing, length] += geometry_label
                if not pd.isna(hmr):
                    hmr_table.loc[narrowing, length] = hmr
                    is_extracted_flag['hmr'].loc[narrowing, length] = True
                    geometry_labels_hmr.loc[narrowing, length] += geometry_label
                if not pd.isna(hsr):
                    hsr_table.loc[narrowing, length] = hsr
                    is_extracted_flag['hsr'].loc[narrowing, length] = True
                    geometry_labels_hsr.loc[narrowing, length] += geometry_label
                if not pd.isna(flow):
                    flow_ffr[geometry] = flow  # Store flow per geometry

    # Now, compute CFR using flow values where both FFR and iFR flows are available for the same geometry
    for geometry in geometry_numbers_ifr:
        flow_ffr_val = flow_ffr.get(geometry, np.nan)
        flow_ifr_val = flow_ifr.get(geometry, np.nan)
        if pd.notna(flow_ffr_val) and pd.notna(flow_ifr_val) and flow_ifr_val != 0:
            cfr_val = flow_ffr_val / flow_ifr_val
            # Find the location in the table corresponding to this geometry
            mask = (all_data['geometry'] == geometry)
            narrowing = all_data.loc[mask, 'narrowing'].iloc[0]
            length = all_data.loc[mask, 'length'].iloc[0]
            cfr_table.loc[narrowing, length] = cfr_val
            is_extracted_flag['cfr'].loc[narrowing, length] = True
            ffr_val = ffr_table.loc[narrowing, length]
            ifr_val = ifr_table.loc[narrowing, length]
            hmr_val = hmr_table.loc[narrowing, length]
            hsr_val = hsr_table.loc[narrowing, length]
            geometry_label = geometry_labels_ffr.loc[narrowing, length]
            # Collect data for scatter plot if HMR and HSR are available
            if pd.notna(hmr_val) and pd.notna(hsr_val):
                scatter_data.append({
                    'geometry': str(geometry),
                    'hmr': hmr_val,
                    'ffr': ffr_val,
                    'ifr': ifr_val,
                    'cfr': cfr_val,
                    'hsr': hsr_val
                })

    # Ensure narrowing is in ascending order (lowest at bottom, largest at top)
    ifr_table = ifr_table.sort_index(ascending=False)
    ffr_table = ffr_table.sort_index(ascending=False)
    cfr_table = cfr_table.sort_index(ascending=False)
    hmr_table = hmr_table.sort_index(ascending=False)
    hsr_table = hsr_table.sort_index(ascending=False)
    geometry_labels_ifr = geometry_labels_ifr.sort_index(ascending=False)
    geometry_labels_ffr = geometry_labels_ffr.sort_index(ascending=False)
    geometry_labels_hmr = geometry_labels_hmr.sort_index(ascending=False)
    geometry_labels_hsr = geometry_labels_hsr.sort_index(ascending=False)
    for key in is_extracted_flag:
        is_extracted_flag[key] = is_extracted_flag[key].sort_index(ascending=False)

    # Define a function to plot and save a table with annotations and custom colormap
    def plot_table_with_bold(data_table, flag_table, geometry_labels, title, vmin=None, vmax=None):
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['red', 'yellow', 'green', 'blue'])
        # Convert data_table to numeric, coercing errors to NaN
        data_table = data_table.apply(pd.to_numeric, errors='coerce')
        sns.heatmap(data_table, ax=ax, cmap=custom_cmap, cbar=True, annot=False, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Lesion Length [cm]')
        ax.set_ylabel('Stenosis [%]')

        # Add annotations with bold font where data is extracted
        for (i, j), val in np.ndenumerate(data_table.values):
            narrowing = data_table.index[i]
            length = data_table.columns[j]
            label = geometry_labels.iloc[i, j]
            is_extracted = flag_table.iloc[i, j]
            # Safely format display_val
            if pd.isna(val):
                display_val = ''
            else:
                try:
                    display_val = f'{float(val):.2f}'
                except (ValueError, TypeError):
                    display_val = ''

            if label:
                text = f'{display_val}\n{label}'
            else:
                text = display_val

            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    fontweight='bold' if is_extracted else 'normal', fontsize=8, color='black')
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(output_path)
        print(f"Saved {title} figure at {output_path}")
        plt.show()

    # Plot iFR table (iFR first)
    plot_table_with_bold(ifr_table, is_extracted_flag['ifr'], geometry_labels_ifr, 'LAD Lesion iFR', vmin=0.7, vmax=1)
    # Plot FFR table
    plot_table_with_bold(ffr_table, is_extracted_flag['ffr'], geometry_labels_ffr, 'LAD Lesion FFR', vmin=0.5, vmax=1)
    # Plot CFR table
    plot_table_with_bold(cfr_table, is_extracted_flag['cfr'], geometry_labels_ffr, 'LAD Lesion CFR', vmin=0.5, vmax=3.5)
    # Plot HMR table
    if not hmr_table.isnull().all().all():
        plot_table_with_bold(hmr_table, is_extracted_flag['hmr'], geometry_labels_hmr, 'HMR Table')
    else:
        print("No HMR data available.")
    # Plot HSR table
    if not hsr_table.isnull().all().all():
        plot_table_with_bold(hsr_table, is_extracted_flag['hsr'], geometry_labels_hsr, 'HSR Table')
    else:
        print("No HSR data available.")

    # Convert scatter_data to DataFrame and return it
    scatter_df = pd.DataFrame(scatter_data)
    return scatter_df


# Create tables and get data for scatter plots
scatter_df = create_tables(all_data)

def create_scatter_plots(scatter_df):
    if scatter_df.empty:
        print("No valid data available for scatter plots.")
        return

    # Custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("RedWhiteGreen", ["#ff0000", "#ffffff", "#00ff00"])
    custom_cmap_reverse = LinearSegmentedColormap.from_list("GreenWhiteRed", ["#00ff00", "#ffffff", "#ff0000"])

    # Scatter plot for FFR vs HMR colored by HSR
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(scatter_df['hmr'], scatter_df['ffr'], c=scatter_df['hsr'], cmap=custom_cmap, s=100, edgecolor='k')
    plt.colorbar(scatter, label="HSR [mmHg/cm/s]")
    for i, row in scatter_df.iterrows():
        if row['geometry']:  # Only add label if geometry is not empty
            plt.annotate(
                f"#{row['geometry']}",
                xy=(row['hmr'], row['ffr']),
                xytext=(row['hmr'] - 0.08, row['ffr'] + 0.02),  # Adjust these values for better positioning
                arrowprops=dict(arrowstyle="-", color='gray'),
                fontsize=10,
                ha='right'
            )
    plt.xlabel("HMR [mmHg/cm/s]")
    plt.ylabel("FFR")
    plt.title("FFR vs HMR, colored by HSR")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "hmr_vs_ffr_scatter.png")
    plt.savefig(output_path)
    print(f"Saved FFR scatter plot at {output_path}")
    plt.show()

    # Scatter plot for CFR vs FFR colored by HMR
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(scatter_df['ffr'], scatter_df['cfr'], c=scatter_df['hmr'], cmap=custom_cmap_reverse, s=100, edgecolor='k')
    plt.colorbar(scatter, label="HMR [mmHg/cm/s]")
    for i, row in scatter_df.iterrows():
        if row['geometry']:
            plt.annotate(
                f"#{row['geometry']}",
                xy=(row['ffr'], row['cfr']),
                xytext=(row['ffr'] + 0.025, row['cfr'] - 0.06),  # Adjust these values for better positioning
                arrowprops=dict(arrowstyle="-", color='gray'),
                fontsize=10,
                ha='right'
            )
    plt.xlabel("FFR")
    plt.ylabel("CFR")
    plt.title("CFR vs FFR, colored by HMR")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "ffr_vs_cfr_scatter.png")
    plt.savefig(output_path)
    print(f"Saved CFR scatter plot at {output_path}")
    plt.show()

    # Scatter plot for flows colored by CFR
    ifr_flow_data = all_data[all_data['source'] == 'ifr'][['geometry', 'flow']].rename(columns={'flow': 'flow_ifr'})
    ffr_flow_data = all_data[all_data['source'] == 'ffr'][['geometry', 'flow']].rename(columns={'flow': 'flow_ffr'})

    ifr_flow_data['geometry'] = ifr_flow_data['geometry'].astype(str)
    ffr_flow_data['geometry'] = ffr_flow_data['geometry'].astype(str)
    scatter_df['geometry'] = scatter_df['geometry'].astype(str)

    flow_comparison = pd.merge(ifr_flow_data, ffr_flow_data, on='geometry', how='inner')
    cfr_data = scatter_df[['geometry', 'cfr']].drop_duplicates()
    flow_comparison = flow_comparison.merge(cfr_data, on='geometry', how='left')

    if flow_comparison.empty:
        print("No overlapping geometries between iFR and FFR cases for flow comparison.")
    else:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            flow_comparison['flow_ifr'],
            flow_comparison['flow_ffr'],
            c=flow_comparison['cfr'],
            cmap=custom_cmap,
            s=100,
            edgecolor='k'
        )
        plt.colorbar(scatter, label="CFR")
        for _, row in flow_comparison.iterrows():
            plt.annotate(
                f"#{row['geometry']}",
                xy=(row['flow_ifr'], row['flow_ffr']),
                xytext=(row['flow_ifr'] - 0.1, row['flow_ffr'] + 0.1),
                arrowprops=dict(arrowstyle="-", color='gray'),
                fontsize=10,
                ha='right'
            )
        plt.xlabel("iFR Flow")
        plt.ylabel("FFR Flow")
        plt.title("iFR Flow vs FFR Flow, colored by CFR")
        plt.tight_layout()
        output_path = os.path.join(output_dir, "ifr_vs_ffr_flow_scatter_cfr.png")
        plt.savefig(output_path)
        print(f"Saved iFR vs FFR flow scatter plot colored by CFR at {output_path}")
        plt.show()



# Create scatter plots
create_scatter_plots(scatter_df)

# Print flow values for iFR geometries
print("Flow values for iFR geometries:")
ifr_flow_data = all_data[all_data['source'] == 'ifr'][['geometry', 'flow']]
for _, row in ifr_flow_data.iterrows():
    print(f"Geometry {row['geometry']:.0f}: Flow = {row['flow']}")

# Print flow values for FFR geometries
print("\nFlow values for FFR geometries:")
ffr_flow_data = all_data[all_data['source'] == 'ffr'][['geometry', 'flow']]
for _, row in ffr_flow_data.iterrows():
    print(f"Geometry {row['geometry']:.0f}: Flow = {row['flow']}")
    
# Print HMR values for FFR geometries
print("\nHMR values for FFR geometries:")
hmr_data = all_data[all_data['source'] == 'ffr'][['geometry', 'hmr']]
for _, row in hmr_data.iterrows():
    print(f"Geometry {row['geometry']:.0f}: HMR = {row['hmr']}")

