#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:54:36 2025

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
output_file = os.path.join(base_dir, 'Post_Processing/summary.csv')

# Initialize list to store data
data = []

# Count simulations where location was assumed (i.e. neither ffr file was found)
assumption_made_count = 0

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
            location = ''

            # Set 'rtotal_cor_value' based on conditions
            if hyperemic_status == 'Non-hyperemic':
                # For 'master_ifr' runs, set value to 0.24
                rtotal_cor_value = 0.24
            elif hyperemic_status == 'Hyperemic':
                # For 'master_ffr' runs, default value if file does not exist
                rtotal_cor_value = 0.24
                # Check if 'WriteCoronaryLPN.py' exists in the Geometry folder
                lpn_file_path = os.path.join(item_path, 'WriteCoronaryLPN.py')
                if os.path.exists(lpn_file_path):
                    with open(lpn_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            stripped_line = line.strip()
                            if stripped_line.startswith('Rtotal_cor ='):
                                match = re.search(r'\*\s*([0-9.]+)', stripped_line)
                                if match:
                                    rtotal_cor_value = float(match.group(1))
                                else:
                                    print(f"Could not extract 'Rtotal_cor' value from line: {stripped_line}")
                                break  # Exit loop after finding the line
                # Else remains 0.24 if not found

            # --- Determine Simulation Location and set appropriate filenames ---
            plaque_lcx_ffr_file = os.path.join(results_path, 'plaque_lcx_ffr.csv')
            plaque_lad_ffr_file = os.path.join(results_path, 'plaque_lad_ffr.csv')
            if os.path.exists(plaque_lcx_ffr_file):
                location = "LCX"
                ffr_file = plaque_lcx_ffr_file
            elif os.path.exists(plaque_lad_ffr_file):
                location = "LAD"
                ffr_file = plaque_lad_ffr_file
            else:
                # If neither file exists, assume LAD and record assumption.
                location = "LAD"
                ffr_file = plaque_lad_ffr_file
                assumption_made_count += 1

            # Choose the appropriate file names based on location
            if location == "LCX":
                # In measurements folder the stenosis file changes to stenosis_acc_5.csv
                stenosis_file = os.path.join(measurements_path, 'stenosis_acc_5.csv')
                # For plaque details, use plaque_5_details.csv for LCX
                plaque_file = os.path.join(measurements_path, 'plaque_5_details.csv')
                # In results folder, update HMR and WSS file names accordingly
                hmr_file = os.path.join(results_path, 'plaque_lcx_HMR.csv')
                wss_file = os.path.join(results_path, 'plaque_lcx_avg_wss_max.csv')
            else:
                # LAD simulation filenames
                stenosis_file = os.path.join(measurements_path, 'stenosis_acc_1.csv')
                plaque_file = os.path.join(measurements_path, 'plaque_1_details.csv')
                hmr_file = os.path.join(results_path, 'plaque_lad_HMR.csv')
                wss_file = os.path.join(results_path, 'plaque_lad_avg_wss_max.csv')

            # --- Read files using the appropriate filenames ---

            # Read stenosis percentage
            if os.path.exists(stenosis_file):
                with open(stenosis_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # first non-empty row
                            stenosis_percentage = row[0]
                            break
            else:
                print(f"File not found: {stenosis_file}")

            # Read plaque details
            if os.path.exists(plaque_file):
                with open(plaque_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        length = rows[1][0]
                        width = rows[1][1]
                    else:
                        print(f"Unexpected format in file: {plaque_file}")
            else:
                print(f"File not found: {plaque_file}")

            # Read flows
            flows_file = os.path.join(results_path, 'all_results-flows.txt')
            if os.path.exists(flows_file):
                with open(flows_file, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    flows = []
                    header = next(reader, None)  # skip header
                    for row in reader:
                        if row:
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

            # Read the ffr file
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

            # Read HMR & HSR
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

            # Read WSS
            if os.path.exists(wss_file):
                with open(wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {wss_file}")
                    else:
                        print(f"Unexpected format in file: {wss_file}")
            else:
                print(f"File not found: {wss_file}")

            # Create a data dictionary including new column 'Location'
            # ### NEW/CHANGED: Add 'CFR/FFR': '' as a placeholder
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
                'WSS': wss,
                'Rtotal_cor Value': rtotal_cor_value,
                'Location': location,
                'CFR': '',
                'BMR/HMR': '',
                'CFR/FFR': ''  # ### NEW
            }

            data.append(data_dict)

# --- Calculate CFR and HMR/BMR ---
ifr_lookup = {}
for run in data:
    if run['Condition'] == 'Non-hyperemic':
        try:
            key = (round(float(run['Stenosis Percentage']), 3), round(float(run['Length']), 3))
        except ValueError:
            continue
        try:
            avg_flow = float(run['Average Flow'])
            hmr_value = float(run['HMR']) if run['HMR'] != '' else None
        except ValueError:
            avg_flow = None
            hmr_value = None
        ifr_lookup[key] = {
            'Average Flow': avg_flow,
            'HMR': hmr_value
        }

# For each FFR run, find a matching iFR run and calculate CFR and HMR/BMR
for run in data:
    if run['Condition'] == 'Hyperemic':
        try:
            key = (round(float(run['Stenosis Percentage']), 3), round(float(run['Length']), 3))
        except ValueError:
            continue

        matching_ifr = ifr_lookup.get(key)
        if not matching_ifr:
            matching_key = find_key_within_tolerance(ifr_lookup, key, tolerance=0.003)
            if matching_key:
                matching_ifr = ifr_lookup[matching_key]

        if matching_ifr:
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

            try:
                ffr_hmr = float(run['HMR'])
                ifr_hmr = matching_ifr['HMR']
                if ffr_hmr is not None and ifr_hmr is not None and ffr_hmr != 0:
                    bmr_hmr = ifr_hmr / ffr_hmr
                    run['BMR/HMR'] = bmr_hmr
                else:
                    run['BMR/HMR'] = ''
            except ValueError:
                run['BMR/HMR'] = ''
        else:
            run['CFR'] = ''
            run['BMR/HMR'] = ''

        # ### NEW: Compute CFR/FFR for hyperemic runs
        try:
            if run['CFR'] != '' and run['P_d/P_a'] != '':
                cfr_val = float(run['CFR'])
                ffr_val = float(run['P_d/P_a'])
                if ffr_val != 0:
                    run['CFR/FFR'] = cfr_val / ffr_val
                else:
                    run['CFR/FFR'] = ''
            else:
                run['CFR/FFR'] = ''
        except ValueError:
            run['CFR/FFR'] = ''
    else:
        # Non-hyperemic => keep CFR, BMR/HMR, CFR/FFR as ''
        run['CFR/FFR'] = ''

# --- Write the summary CSV file ---
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
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
        'Location',
        'CFR',
        'BMR/HMR',
        'CFR/FFR'  # ### NEW
    ]
    writer.writerow(header)
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
            run['Location'],
            run['CFR'],
            run['BMR/HMR'],
            run['CFR/FFR']  # ### NEW
        ]
        writer.writerow(row)

print(f"Summary file written to {output_file}")

if assumption_made_count > 0:
    print(f"Location assumption made for {assumption_made_count} simulation(s): "
          f"neither plaque_lad_ffr.csv nor plaque_lcx_ffr.csv was found. LAD assumed.")
