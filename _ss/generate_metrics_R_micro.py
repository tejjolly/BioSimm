#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:34:00 2024

@author: tejjolly
"""

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
geometry_numbers_ifr = [18, 26, 31, 32, 33, 34, 35, 36]
geometry_numbers_ffr = [18, 26, 31, 32, 33, 34, 35, 36]

# Extract datasets (iFR first)
ifr_data = extract_dataset(geometry_numbers_ifr, ifr_base_dir, source='ifr')
ffr_data = extract_dataset(geometry_numbers_ffr, ffr_base_dir, source='ffr')

# Combine the datasets (iFR data first)
all_data = pd.concat([ifr_data, ffr_data], ignore_index=True)

# Define the function to create tables and plot them
def create_tables(all_data):
    # Prepare a list to collect data for scatter plots
    scatter_df_list = []

    # Now, process the extracted data and collect scatter data per geometry
    for geometry in geometry_numbers_ifr:
        # Get FFR data for the geometry
        ffr_row = all_data[(all_data['geometry'] == geometry) & (all_data['source'] == 'ffr')]
        # Get iFR data for the geometry
        ifr_row = all_data[(all_data['geometry'] == geometry) & (all_data['source'] == 'ifr')]

        if not ffr_row.empty and not ifr_row.empty:
            flow_ffr_val = ffr_row['flow'].iloc[0]
            flow_ifr_val = ifr_row['flow'].iloc[0]
            if flow_ifr_val != 0:
                cfr_val = flow_ffr_val / flow_ifr_val
                hmr_val = ffr_row['hmr'].iloc[0]
                hsr_val = ffr_row['hsr'].iloc[0]
                ffr_val = ffr_row['ffr'].iloc[0]
                ifr_val = ifr_row['ifr'].iloc[0]
                narrowing_val = ffr_row['narrowing'].iloc[0]  # Get narrowing value
                scatter_df_list.append({
                    'geometry': str(geometry),
                    'hmr': hmr_val,
                    'hsr': hsr_val,
                    'ffr': ffr_val,
                    'ifr': ifr_val,
                    'cfr': cfr_val,
                    'narrowing': narrowing_val  # Include narrowing in scatter_df
                })

    scatter_df = pd.DataFrame(scatter_df_list)
    return scatter_df

# Create tables and get data for scatter plots
scatter_df = create_tables(all_data)

# Define R_values (user-defined), must be same length as scatter_df
# You should replace the following list with your actual R values
R_values = [0.24, 0.24, 0.43, 0.62, 0.81, 0.43, 0.62, 0.81]  # Example values; ensure this list matches the length of scatter_df

def create_scatter_plots(scatter_df, R_values):
    if scatter_df.empty:
        print("No valid data available for scatter plots.")
        return

    # Ensure R_values length matches scatter_df
    if len(R_values) != len(scatter_df):
        print("Error: Length of R_values does not match number of data points.")
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
            label = f"S: {row['narrowing']:.2f}. R: {R_values[i]}"
            plt.annotate(
                label,
                xy=(row['hmr'], row['ffr']),
                xytext=(row['hmr'] + 0.55, row['ffr'] - 0.02),  # Adjust these values for better positioning
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
            label = f"S: {row['narrowing']:.2f}., R: {R_values[i]}"
            plt.annotate(
                label,
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
    cfr_data = scatter_df[['geometry', 'cfr', 'narrowing']].drop_duplicates()
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
        for i, row in flow_comparison.iterrows():
            label = f"S: {row['narrowing']:.2f}. R: {R_values[i]}"
            plt.annotate(
                label,
                xy=(row['flow_ifr'], row['flow_ffr']),
                xytext=(row['flow_ifr'] + 0.035, row['flow_ffr'] + 10),
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

# Create scatter plots with the provided R_values
create_scatter_plots(scatter_df, R_values)

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
