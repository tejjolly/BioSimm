import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Ensure the directory exists
output_dir = "./tables"
os.makedirs(output_dir, exist_ok=True)

# Function to extract values from CSVs
def extract_values(geometry_n):
    base_path = f"./master_ffr/Geometry_{geometry_n}"  # Adjust this path if necessary
    extracted_data = {}

    # 1. Extract length from plaque_1_details.csv
    plaque_details_path = os.path.join(base_path, "measurements", "plaque_1_details.csv")
    plaque_details_df = pd.read_csv(plaque_details_path)
    length = plaque_details_df.iloc[0, 0]  # Extracting length from row index 0

    # 2. Extract narrowing from stenosis_acc_1.csv
    stenosis_acc_path = os.path.join(base_path, "measurements", "stenosis_acc_1.csv")
    with open(stenosis_acc_path, 'r') as file:
        narrowing = float(file.read().strip())  # Reading the single value and converting to float

    # Print original length and narrowing values
    print(f"Geometry {geometry_n}: Original Length = {length}, Original Narrowing = {narrowing}")

    # Storing extracted values
    extracted_data['length'] = length
    extracted_data['narrowing'] = narrowing

    # 5. Extract ffr from plaque_lad_ffr.csv
    plaque_lad_ffr_path = os.path.join(base_path, "results-processed-new", "plaque_lad_ffr.csv")
    plaque_lad_ffr_df = pd.read_csv(plaque_lad_ffr_path)
    ffr = plaque_lad_ffr_df.iloc[0, 0]# * 0.01  # Extracting ffr from row index 1

    # Storing extracted values
    extracted_data['ffr'] = ffr
    extracted_data['geometry'] = geometry_n
    
    return extracted_data

# Function to create tables of the extracted data and plot them
def create_tables(geometry_numbers):
    # Start with an empty list for narrowing values
    narrowing_values = []

    # # Define the initial target narrowing values
    # initial_narrowing_values = [0.40, 0.50, 0.60]
    length_values = [0.9, 1.2, 1.5, 1.7, 2.0, 2.5]

    # Extract values for each specified Geometry folder
    for n in geometry_numbers:
        data = extract_values(n)

        # Adjust length values to fit within the table's range
        length = min(max(round(data['length'], 1), min(length_values)), max(length_values))
        narrowing = round(data['narrowing'], 2)  # Use real narrowing value rounded to 2 decimal places

        # Add the real narrowing value to the narrowing_values list if not already present
        if narrowing not in narrowing_values:
            narrowing_values.append(narrowing)

    # Add the initial narrowing values to narrowing_values and sort the entire list
    narrowing_values = sorted(list(set(narrowing_values)))

    # Print the sorted narrowing values for verification
    print(f"Sorted Narrowing Values: {narrowing_values}")

    # Initialize tables for each value type with NaNs
    ffr_table = pd.DataFrame(np.nan, index=narrowing_values, columns=length_values)

    # Create flag DataFrames to track extracted data
    is_extracted_flag = {
        'ffr': pd.DataFrame(False, index=narrowing_values, columns=length_values)
    }

    # Create a DataFrame to track the Geometry numbers for each cell
    geometry_labels = pd.DataFrame('', index=narrowing_values, columns=length_values)

    # Update tables and flag with the extracted values for each specified Geometry folder
    for n in geometry_numbers:
        data = extract_values(n)
        length = min(max(round(data['length'], 1), min(length_values)), max(length_values))
        narrowing = round(data['narrowing'], 2)

        if narrowing in narrowing_values and length in length_values:
            ffr_table.loc[narrowing, length] = data['ffr']

            # Set the flag to True for extracted data
            is_extracted_flag['ffr'].loc[narrowing, length] = True

            # Add Geometry number label
            geometry_labels.loc[narrowing, length] = f'#{n}'

    # # Add data from the image to the tables
    # image_stenosis_values = [0.40, 0.50, 0.60]
    # image_lengths = [0.9, 1.7, 2.0, 2.5]
    # image_ffr_values = [
    #     [0.98, 0.97, 0.96, 0.96],
    #     [0.95, 0.91, 0.90, 0.88],
    #     [0.88, 0.79, 0.77, 0.74]
    # ]

    # for i, narrowing in enumerate(image_stenosis_values):
    #     for j, length in enumerate(image_lengths):
    #         if narrowing in narrowing_values and length in length_values:
    #             ffr_table.loc[narrowing, length] = image_ffr_values[i][j]

    # Plot the tables with a color scale
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Extracted Data Tables with Color Scale', fontsize=16)

    def plot_table_with_bold(data_table, flag_table, geometry_labels, ax, title):
        # Apply bold annotation only where the flag is True
        for (i, j), val in np.ndenumerate(data_table):
            label = geometry_labels.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f'{val:.2f}\n{label}' if not np.isnan(val) else label,
                    ha='center', va='center',
                    fontweight='bold' if flag_table.iloc[i, j] else 'normal', fontsize=8, color='white')
        
        # Define a custom colormap that goes from red to yellow to green to blue
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['red', 'yellow', 'green', 'blue'])
        
        sns.heatmap(data_table, ax=ax, cmap=custom_cmap, cbar=True, annot=False,
                    cbar_kws={'ticks': np.arange(0.7, 1.05, 0.05)}, vmin=0.7, vmax=1)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Lesion Length [cm]')
        ax.set_ylabel('Stenosis [%]')

    plot_table_with_bold(ffr_table, is_extracted_flag['ffr'], geometry_labels, axes[2, 0], 'FFR Table')

    # Hide the last subplot as we have 5 tables and 6 subplots
    fig.delaxes(axes[2, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Plot the FFR table separately
    plt.figure(figsize=(8, 6))
    plot_table_with_bold(ffr_table, is_extracted_flag['ffr'], geometry_labels, plt.gca(), 'LAD Lesion iFR')
    plt.tight_layout()
    
    # Save the figure as "ffr.png" in the "/tables" directory
    output_path = os.path.join(output_dir, "ffr.png")
    plt.savefig(output_path)
    print(f"Saved FFR table figure at {output_path}")
    
    plt.show()

geometry_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,22,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42]  # Replace this with geometry numbers
create_tables(geometry_numbers)
