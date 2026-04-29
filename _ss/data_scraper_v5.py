#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:42:39 2024

@author: tejjolly
"""

import os
import csv
import re

def find_key_within_tolerance(lookup, target_key, tolerance=0.003):
    for key in lookup.keys():
        if abs(key[0] - target_key[0]) <= tolerance and abs(key[1] - target_key[1]) <= tolerance:
            return key
    return None

# Base directory
base_dir = '/Users/tejjolly/Documents/BioSimm/Simulations/'

# Output file path
output_file = os.path.join(base_dir, 'summary.csv')

# Initialize list to store data
data = []

# For each of 'master_ifr' and 'master_ffr'
for master_folder in ['master_ifr', 'master_ffr']:
    master_path = os.path.join(base_dir, master_folder)
    # Determine if it's hyperemic or non-hyperemic
    if master_folder == 'master_ifr':
        hyperemic_status = 'Non-hyperemic'
    elif master_folder == 'master_ffr':
        hyperemic_status = 'Hyperemic'
    else:
        hyperemic_status = 'Unknown'

    # Check if master_path exists
    if not os.path.exists(master_path):
        print(f"Master folder not found: {master_path}")
        continue

    # List all items in master_path
    for item in os.listdir(master_path):
        item_path = os.path.join(master_path, item)
        # Check if item is a directory and starts with 'Geometry_'
        if os.path.isdir(item_path) and item.startswith('Geometry_'):
            # Extract the Geometry number
            geometry_number = item.replace('Geometry_', '')
            # Paths to 'measurements' and 'results-processed-new' folders
            measurements_path = os.path.join(item_path, 'measurements')
            results_path = os.path.join(item_path, 'results-processed-new')

            # Initialize variables
            stenosis_percentage = ''
            length = ''
            width = ''
            average_flow = ''
            pd_pa = ''
            hmr = ''
            hsr = ''
            rtotal_cor_value = ''
            wss = ''

            # Set 'rtotal_cor_value' based on conditions
            if hyperemic_status == 'Non-hyperemic':
                # For all 'master_ifr' runs, set value to 1.0
                rtotal_cor_value = 0.24
            elif hyperemic_status == 'Hyperemic':
                # For 'master_ffr' runs
                # Default value if file does not exist
                rtotal_cor_value = 0.24
                # Check if 'WriteCoronaryLPN.py' exists in the 'Geometry_n' directory
                lpn_file_path = os.path.join(item_path, 'WriteCoronaryLPN.py')
                if os.path.exists(lpn_file_path):
                    with open(lpn_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            # Remove whitespace and check if line starts with 'Rtotal_cor ='
                            stripped_line = line.strip()
                            if stripped_line.startswith('Rtotal_cor ='):
                                # Use regex to extract the float value at the end
                                match = re.search(r'\*\s*([0-9.]+)', stripped_line)
                                if match:
                                    rtotal_cor_value = float(match.group(1))
                                else:
                                    print(f"Could not extract 'Rtotal_cor' value from line: {stripped_line}")
                                break  # Exit the loop after finding the line
                else:
                    # File does not exist, value remains as 0.24
                    pass

            # Read 'stenosis_acc_1.csv'
            stenosis_file = os.path.join(measurements_path, 'stenosis_acc_1.csv')
            if os.path.exists(stenosis_file):
                with open(stenosis_file, 'r') as f:
                    reader = csv.reader(f)
                    # Read the first non-empty line
                    for row in reader:
                        if row:  # Non-empty row
                            stenosis_percentage = row[0]
                            break
            else:
                print(f"File not found: {stenosis_file}")

            # Read 'plaque_1_details.csv'
            plaque_file = os.path.join(measurements_path, 'plaque_1_details.csv')
            if os.path.exists(plaque_file):
                with open(plaque_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        # Length is in the second row, first column
                        length = rows[1][0]
                        # Width is in the second row, second column
                        width = rows[1][1]
                    else:
                        print(f"Unexpected format in file: {plaque_file}")
            else:
                print(f"File not found: {plaque_file}")

            # Read 'all_results-flows.txt' and calculate average flow
            flows_file = os.path.join(results_path, 'all_results-flows.txt')
            if os.path.exists(flows_file):
                with open(flows_file, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    flows = []
                    header = next(reader, None)  # Skip header
                    for row in reader:
                        if row:
                            # Assuming the last column is at the end of the row
                            flow_value = row[-1]
                            try:
                                flows.append(-float(flow_value))
                            except ValueError:
                                print(f"Invalid flow value in file: {flows_file}")
                    if flows:
                        average_flow = sum(flows) / len(flows)
                    else:
                        print(f"No flow data found in file: {flows_file}")
            else:
                print(f"File not found: {flows_file}")

            # Read 'plaque_lad_ffr.csv' to get P_d/P_a
            ffr_file = os.path.join(results_path, 'plaque_lad_ffr.csv')
            if os.path.exists(ffr_file):
                with open(ffr_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        pd_pa = rows[1][0]
                    else:
                        print(f"Unexpected format in file: {ffr_file}")
            else:
                print(f"File not found: {ffr_file}")

            # Read 'plaque_lad_HMR.csv' to get HMR and HMS
            hmr_file = os.path.join(results_path, 'plaque_lad_HMR.csv')
            if os.path.exists(hmr_file):
                with open(hmr_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 3:
                        hmr = rows[1][0]
                        hsr = rows[2][0]
                    else:
                        print(f"Unexpected format in file: {hmr_file}")
            else:
                print(f"File not found: {hmr_file}")
                
            # Read 'plaque_lad_avg_wss.csv' for WSS (avg.)
            wss_file = os.path.join(results_path, 'plaque_lad_avg_wss.csv')
            if os.path.exists(wss_file):
                with open(wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2 and len(rows[1]) >= 4:
                        wss_avg = rows[1][3]  # Extract second row, fourth column
                    else:
                        print(f"Unexpected format in file: {wss_file}")
            else:
                print(f"File not found: {wss_file}")

            # Create a data dictionary for this run
            data_dict = {
                'Condition': hyperemic_status,
                'Geometry Number': geometry_number,
                'Stenosis Percentage': stenosis_percentage,
                'Length': length,
                'Width': width,
                'Average Flow': average_flow,
                'P_d/P_a': pd_pa,
                'HMR': hmr,
                'HSR': hsr,
                'WSS': wss_avg,
                'Rtotal_cor Value': rtotal_cor_value,
                # Initialize new metrics to empty strings
                'CFR': '',
                'BMR/HMR': ''
            }

            # Append the data dictionary to the data list
            data.append(data_dict)

# Now, calculate CFR and HMR/BMR
# First, build a lookup table for iFR runs keyed by (Stenosis Percentage, Length)
ifr_lookup = {}
for run in data:
    if run['Condition'] == 'Non-hyperemic':
        key = (round(float(run['Stenosis Percentage']), 3), round(float(run['Length']), 3))
        # Ensure that Average Flow and HMR are valid numbers
        try:
            avg_flow = float(run['Average Flow'])
            hmr_value = float(run['HMR']) if run['HMR'] != '' else None
        except ValueError:
            avg_flow = None
            hmr_value = None
        # Store the values in the lookup table
        ifr_lookup[key] = {
            'Average Flow': avg_flow,
            'HMR': hmr_value
        }

# Now, for each FFR run, find a matching iFR run and calculate CFR and HMR/BMR
for run in data:
    if run['Condition'] == 'Hyperemic':
        key = (round(float(run['Stenosis Percentage']), 3), round(float(run['Length']), 3))
        
        # Try exact match first
        matching_ifr = ifr_lookup.get(key)
        
        # If no exact match, try finding a key within tolerance
        if not matching_ifr:
            matching_key = find_key_within_tolerance(ifr_lookup, key, tolerance=0.003)
            if matching_key:
                matching_ifr = ifr_lookup[matching_key]
        
        if matching_ifr:
            # Calculate CFR
            try:
                ffr_avg_flow = float(run['Average Flow'])
                ifr_avg_flow = matching_ifr['Average Flow']
                if ffr_avg_flow is not None and ifr_avg_flow is not None and ifr_avg_flow != 0:
                    cfr = ffr_avg_flow / ifr_avg_flow
                    run['CFR'] = cfr
                else:
                    run['CFR'] = ''
            except ValueError:
                run['CFR'] = ''

            # Calculate HMR/BMR
            try:
                ffr_hmr = float(run['HMR'])
                ifr_hmr = matching_ifr['HMR']
                if ffr_hmr is not None and ifr_hmr is not None and ifr_hmr != 0:
                    bmr_hmr = ifr_hmr / ffr_hmr
                    run['BMR/HMR'] = bmr_hmr
                else:
                    run['BMR/HMR'] = ''
            except ValueError:
                run['BMR/HMR'] = ''
        else:
            # No matching iFR run found
            run['CFR'] = ''
            run['BMR/HMR'] = ''
    else:
        # For iFR runs, CFR and HMR/BMR remain empty
        pass

# Write the data to CSV file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    header = [
        'Condition',
        'Geometry Number',
        'Stenosis Percentage',
        'Length',
        'Width',
        'Average Flow',
        'P_d/P_a',
        'HMR',
        'HSR',
        'WSS',
        'Rtotal_cor Value',
        'CFR',
        'BMR/HMR'
    ]
    writer.writerow(header)
    # Write data rows
    for run in data:
        row = [
            run['Condition'],
            run['Geometry Number'],
            run['Stenosis Percentage'],
            run['Length'],
            run['Width'],
            run['Average Flow'],
            run['P_d/P_a'],
            run['HMR'],
            run['HSR'],
            run['WSS'],
            run['Rtotal_cor Value'],
            run['CFR'],
            run['BMR/HMR']
        ]
        writer.writerow(row)

print(f"Summary file written to {output_file}")
