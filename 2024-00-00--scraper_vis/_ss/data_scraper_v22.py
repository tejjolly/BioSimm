#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 23:56:31 2025

@author: tejjolly
"""

import os
import csv
import re
import numpy as np


def find_key_within_tolerance(lookup, target_key, tolerance=0.003):
    for key in lookup.keys():
        if abs(key[0] - target_key[0]) <= tolerance and abs(key[1] - target_key[1]) <= tolerance:
            return key
    return None


# Base directory
base_dir = '/Users/tejjolly/Documents/BioSimm/Simulations/'

# Output file path
output_file = os.path.join(base_dir, 'Post_Processing/summary2.csv')

# The Garcia data file
garcia_file = os.path.join(base_dir, 'Post_Processing/garcia_data_v2.csv')
garcia = False

# Initialize list to store data
data = []

# Count simulations where location was assumed (i.e. neither ffr file was found)
assumption_made_count = 0

# ----------------------------------------------------------------------------
# 1) GATHER YOUR EXISTING DATA (with "source"="mine")
# ----------------------------------------------------------------------------
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
            max_flow = ''
            pd_pa = ''
            hmr = ''
            hsr = ''
            p_loss_coeff = ''
            rtotal_cor_value = ''
            location = ''

            # We still read in WSS-related values but do NOT store them if they
            # are just "WSS", "WSS_Avg_Area", or "WSS_Avg_Area_min".
            wss = ''
            wss_avg_area = ''
            wss_avg_area_min = ''

            # But we keep these if needed:
            wss_te = ''
            wss_le = ''
            wss_te_area = ''
            wss_le_area = ''
            wss_area_bifur = ''
            wss_bif = ''
            wss_lmb = ''

            # NEW VARIABLES for min values
            wss_min = ''  # We'll keep WSS_min if you still want it
            wss_te_min = ''
            wss_le_min = ''
            wss_te_area_min = ''
            wss_le_area_min = ''
            wss_area_bifur_min = ''

            # NEW variable for v_distal
            v_distal = ''  ### NEW ###

            # NEW variable for R_micro
            r_micro_value = 0  ### NEW ###

            # 1) Set 'rtotal_cor_value'
            #    (and check if we need to parse WriteCoronaryLPN.py in any case)
            lpn_file_path = os.path.join(item_path, 'WriteCoronaryLPN.py')
            if hyperemic_status == 'Non-hyperemic':
                rtotal_cor_value = 0.24
                # We still open the file to see if Rtotal_cor_new is there
                if os.path.exists(lpn_file_path):
                    with open(lpn_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            stripped_line = line.strip()

                            # If we see a normal "Rtotal_cor =" line
                            if stripped_line.startswith('Rtotal_cor ='):
                                match = re.search(r'\*\s*([0-9.]+)', stripped_line)
                                if match:
                                    rtotal_cor_value = float(match.group(1))
                                    # No break here if we want to continue scanning
                                    # for Rtotal_cor_new as well.
                                    # Or break if you prefer only the first match.

                            # NEW: Search for Rtotal_cor_new
                            if 'Rtotal_cor_new' in stripped_line:
                                # Example line:
                                # Rtotal_cor_new = pressure*pconv/Qtotal/coronary_aorta_flow_split * 0.43 # new
                                # We just parse out the trailing float after the '*'
                                float_match = re.search(r'\*+\s*([\d.]+)', stripped_line)
                                if float_match:
                                    rtotal_cor_new_val = float(float_match.group(1))
                                    r_micro_value = rtotal_cor_new_val - 0.24
            else:
                # hyperemic
                rtotal_cor_value = 0.24  # default if file not found
                if os.path.exists(lpn_file_path):
                    with open(lpn_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            stripped_line = line.strip()

                            # Normal "Rtotal_cor ="
                            if stripped_line.startswith('Rtotal_cor ='):
                                match = re.search(r'\*\s*([0-9.]+)', stripped_line)
                                if match:
                                    rtotal_cor_value = float(match.group(1))

                            # NEW: Search for Rtotal_cor_new
                            if 'Rtotal_cor_new' in stripped_line:
                                float_match = re.search(r'\*+\s*([\d.]+)', stripped_line)
                                if float_match:
                                    rtotal_cor_new_val = float(float_match.group(1))
                                    r_micro_value = rtotal_cor_new_val - 0.24

            # 2) Determine Simulation Location & set ffr_file
            plaque_lcx_ffr_file = os.path.join(results_path, 'plaque_lcx_ffr.csv')
            plaque_lad_ffr_file = os.path.join(results_path, 'plaque_lad_ffr.csv')
            if os.path.exists(plaque_lcx_ffr_file):
                location = "LCX"
                ffr_file = plaque_lcx_ffr_file
            elif os.path.exists(plaque_lad_ffr_file):
                location = "LAD"
                ffr_file = plaque_lad_ffr_file
            else:
                location = "LAD"
                ffr_file = plaque_lad_ffr_file
                assumption_made_count += 1

            # 3) Choose the appropriate file names based on location
            #    (including the new _v_distal file)
            if location == "LCX":
                # LCX filenames
                stenosis_file = os.path.join(measurements_path, 'stenosis_acc_5.csv')
                plaque_file = os.path.join(measurements_path, 'plaque_5_details.csv')
                hmr_file = os.path.join(results_path, 'plaque_lcx_HMR.csv')

                # Distal velocity file
                v_distal_file = os.path.join(results_path, 'plaque_lcx_v_distal.csv')  ### NEW ###

                # MAX WSS files
                wss_file = os.path.join(results_path, 'plaque_lcx_avg_wss_max.csv')  # Not stored in final
                te_wss_file = os.path.join(results_path, 'plaque_lcx_TE_wss_max.csv')
                le_wss_file = os.path.join(results_path, 'plaque_lcx_LE_wss_max.csv')
                high_te_area_file = os.path.join(results_path, 'plaque_high_lcx_TE_wss_area.csv')
                high_le_area_file = os.path.join(results_path, 'plaque_high_lcx_LE_wss_area.csv')
                mb_wss_file = os.path.join(results_path, 'high_mb_wss_area.csv')
                sb_wss_file = os.path.join(results_path, 'high_sb_wss_area.csv')
                avg_high_area_file = os.path.join(results_path, 'plaque_high_lcx_avg_wss_area.csv')  # Not stored

                # MIN WSS files
                wss_min_file = os.path.join(results_path, 'plaque_lcx_avg_wss_min.csv')
                te_wss_min_file = os.path.join(results_path, 'plaque_lcx_TE_wss_min.csv')
                le_wss_min_file = os.path.join(results_path, 'plaque_lcx_LE_wss_min.csv')
                te_wss_area_min_file = os.path.join(results_path, 'plaque_lcx_TE_wss_area.csv')
                le_wss_area_min_file = os.path.join(results_path, 'plaque_lcx_LE_wss_area.csv')
                mb_wss_area_min_file = os.path.join(results_path, 'mb_wss_area.csv')
                sb_wss_area_min_file = os.path.join(results_path, 'sb_wss_area.csv')
                avg_area_min_file = os.path.join(results_path, 'plaque_lcx_avg_wss_area.csv')  # Not stored

            else:
                # LAD filenames
                stenosis_file = os.path.join(measurements_path, 'stenosis_acc_1.csv')
                plaque_file = os.path.join(measurements_path, 'plaque_1_details.csv')
                hmr_file = os.path.join(results_path, 'plaque_lad_HMR.csv')

                # Distal velocity file
                v_distal_file = os.path.join(results_path, 'plaque_lad_v_distal.csv')  ### NEW ###

                # MAX WSS files
                wss_file = os.path.join(results_path, 'plaque_lad_avg_wss_max.csv')  # Not stored in final
                te_wss_file = os.path.join(results_path, 'plaque_lad_TE_wss_max.csv')
                le_wss_file = os.path.join(results_path, 'plaque_lad_LE_wss_max.csv')
                high_te_area_file = os.path.join(results_path, 'plaque_high_lad_TE_wss_area.csv')
                high_le_area_file = os.path.join(results_path, 'plaque_high_lad_LE_wss_area.csv')
                mb_wss_file = os.path.join(results_path, 'high_mb_wss_area.csv')
                sb_wss_file = os.path.join(results_path, 'high_sb_wss_area.csv')
                avg_high_area_file = os.path.join(results_path, 'plaque_high_lad_avg_wss_area.csv')  # Not stored

                # MIN WSS files
                wss_min_file = os.path.join(results_path, 'plaque_lad_avg_wss_min.csv')
                te_wss_min_file = os.path.join(results_path, 'plaque_lad_TE_wss_min.csv')
                le_wss_min_file = os.path.join(results_path, 'plaque_lad_LE_wss_min.csv')
                te_wss_area_min_file = os.path.join(results_path, 'plaque_lad_TE_wss_area.csv')
                le_wss_area_min_file = os.path.join(results_path, 'plaque_lad_LE_wss_area.csv')
                mb_wss_area_min_file = os.path.join(results_path, 'mb_wss_area.csv')
                sb_wss_area_min_file = os.path.join(results_path, 'sb_wss_area.csv')
                avg_area_min_file = os.path.join(results_path, 'plaque_lad_avg_wss_area.csv')  # Not stored

            # 4) Read your existing measurement/result files
            # a) Stenosis
            if os.path.exists(stenosis_file):
                with open(stenosis_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # first non-empty row
                            stenosis_percentage = row[0]
                            break
            else:
                print(f"File not found: {stenosis_file}")

            # b) Plaque details
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

            # c) Flows
            flows_file = os.path.join(results_path, 'all_results-flows.txt')
            flows = []
            if os.path.exists(flows_file):
                with open(flows_file, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None)  # skip header
                    for row in reader:
                        if row:
                            flow_value = row[-1]
                            try:
                                # Typically negative for outflow -> adjust sign as needed
                                flows.append(-float(flow_value))
                            except ValueError:
                                print(f"Invalid flow value in file: {flows_file}")
                if flows:
                    average_flow = sum(flows) / len(flows)
                    max_flow = max(flows)
                else:
                    print(f"No flow data found in file: {flows_file}")
            else:
                print(f"File not found: {flows_file}")

            # d) FFR or iFR
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

            # e) HMR & HSR
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

            # f) v_distal from plaque_*_v_distal.csv file  ### NEW ###
            if os.path.exists(v_distal_file):
                with open(v_distal_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:
                        # try row[1][2], else row[1][0]
                        possible_values = rows[1]
                        v_temp = ''
                        if len(possible_values) > 2 and possible_values[2].strip():
                            v_temp = possible_values[2].strip()
                        elif len(possible_values) > 0 and possible_values[0].strip():
                            v_temp = possible_values[0].strip()
                        v_distal = v_temp
                    else:
                        print(f"Unexpected format or empty v_distal file: {v_distal_file}")
            else:
                print(f"File not found: {v_distal_file}")

            # -----------------------------------------------------------------
            # WSS (MAX) files
            # -----------------------------------------------------------------
            if os.path.exists(te_wss_file):
                with open(te_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te = rows[1][2]

            if os.path.exists(le_wss_file):
                with open(le_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le = rows[1][2]

            if os.path.exists(high_te_area_file):
                with open(high_te_area_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te_area = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te_area = rows[1][2]

            if os.path.exists(high_le_area_file):
                with open(high_le_area_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le_area = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le_area = rows[1][2]

            # Bifur "high" areas (MAX area)
            mb_wss_file_exists = os.path.exists(mb_wss_file)
            sb_wss_file_exists = os.path.exists(sb_wss_file)
            wss_mb_area = ''
            wss_sb_area = ''
            if mb_wss_file_exists:
                with open(mb_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            try:
                                wss_mb_area = float(rows[1][5])
                            except ValueError:
                                wss_mb_area = ''
                        elif len(rows[1]) >= 3:
                            try:
                                wss_mb_area = float(rows[1][2])
                            except ValueError:
                                wss_mb_area = ''

            if sb_wss_file_exists:
                with open(sb_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            try:
                                wss_sb_area = float(rows[1][5])
                            except ValueError:
                                wss_sb_area = ''
                        elif len(rows[1]) >= 3:
                            try:
                                wss_sb_area = float(rows[1][2])
                            except ValueError:
                                wss_sb_area = ''

            wss_area_bifur = ''
            if wss_mb_area != '' and wss_sb_area != '':
                wss_area_bifur = wss_mb_area + wss_sb_area

            # Bifur max
            bif_mb_wss_file = os.path.join(results_path, 'bifurc_mb_wss_max.csv')
            bif_sb_wss_file = os.path.join(results_path, 'bifurc_sb_wss_max.csv')
            wss_mb_bif_value = ''
            wss_sb_bif_value = ''
            if os.path.exists(bif_mb_wss_file):
                with open(bif_mb_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            try:
                                wss_mb_bif_value = float(rows[1][5])
                            except ValueError:
                                wss_mb_bif_value = ''
                        elif len(rows[1]) >= 3:
                            try:
                                wss_mb_bif_value = float(rows[1][2])
                            except ValueError:
                                wss_mb_bif_value = ''

            if os.path.exists(bif_sb_wss_file):
                with open(bif_sb_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            try:
                                wss_sb_bif_value = float(rows[1][5])
                            except ValueError:
                                wss_sb_bif_value = ''
                        elif len(rows[1]) >= 3:
                            try:
                                wss_sb_bif_value = float(rows[1][2])
                            except ValueError:
                                wss_sb_bif_value = ''

            wss_bif = ''
            if wss_mb_bif_value != '' and wss_sb_bif_value != '':
                wss_bif_temp = wss_mb_bif_value + wss_sb_bif_value
                if wss_bif_temp == 0.0:
                    wss_bif = ''
                else:
                    wss_bif = wss_bif_temp

            # LMB WSS max
            lmb_wss_file = os.path.join(results_path, 'lmb_wss_max.csv')
            if os.path.exists(lmb_wss_file):
                with open(lmb_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_lmb = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_lmb = rows[1][2]

            # -----------------------------------------------------------------
            # READ MIN VALUES
            # -----------------------------------------------------------------
            if os.path.exists(wss_min_file):
                with open(wss_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_min = rows[1][2]

            if os.path.exists(te_wss_min_file):
                with open(te_wss_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te_min = rows[1][2]

            if os.path.exists(le_wss_min_file):
                with open(le_wss_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le_min = rows[1][2]

            if os.path.exists(te_wss_area_min_file):
                with open(te_wss_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te_area_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te_area_min = rows[1][2]

            if os.path.exists(le_wss_area_min_file):
                with open(le_wss_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le_area_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le_area_min = rows[1][2]

            mb_wss_area_min = ''
            sb_wss_area_min = ''
            if os.path.exists(mb_wss_area_min_file):
                with open(mb_wss_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            try:
                                mb_wss_area_min = float(rows[1][5])
                            except ValueError:
                                mb_wss_area_min = ''
                        elif len(rows[1]) >= 3:
                            try:
                                mb_wss_area_min = float(rows[1][2])
                            except ValueError:
                                mb_wss_area_min = ''

            if os.path.exists(sb_wss_area_min_file):
                with open(sb_wss_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            try:
                                sb_wss_area_min = float(rows[1][5])
                            except ValueError:
                                sb_wss_area_min = ''
                        elif len(rows[1]) >= 3:
                            try:
                                sb_wss_area_min = float(rows[1][2])
                            except ValueError:
                                sb_wss_area_min = ''

            wss_area_bifur_min = ''
            if mb_wss_area_min != '' and sb_wss_area_min != '':
                wss_area_bifur_min = mb_wss_area_min + sb_wss_area_min

            # 5) Create data dictionary
            data_dict = {
                'Condition': hyperemic_status,
                'Geometry Number': geometry_number,
                'Stenosis Percentage': stenosis_percentage,
                'Length': length,
                'Width': width,
                'Average Flow': average_flow,
                'Max Flow': max_flow,
                'P_d/P_a': pd_pa,
                'HMR': hmr,
                'HSR': hsr,
                'P_Loss_Coeff': p_loss_coeff,  # We'll fill in later if we can
                'WSS_TE': wss_te,
                'WSS_LE': wss_le,
                'WSS_TE_Area': wss_te_area,
                'WSS_LE_Area': wss_le_area,
                'Rtotal_cor Value': rtotal_cor_value,
                'Location': location,
                'CFR': '',
                'BMR/HMR': '',
                'CFR/FFR': '',
                'WSS_Area_Bifur': wss_area_bifur,
                'WSS_Bif': wss_bif,
                'WSS_LMB': wss_lmb,

                'WSS_min': wss_min,
                'WSS_TE_min': wss_te_min,
                'WSS_LE_min': wss_le_min,
                'WSS_TE_Area_min': wss_te_area_min,
                'WSS_LE_Area_min': wss_le_area_min,
                'WSS_Area_Bifur_min': wss_area_bifur_min,

                'v_distal': v_distal,
                'discord': '',
                'source': 'mine',

                # NEW: store R_micro
                'R_micro': r_micro_value
            }

            data.append(data_dict)

# ----------------------------------------------------------------------------
# 2) CALCULATE CFR, BMR/HMR FOR HYPEREMIC RUNS BY MATCHING NON-HYPEREMIC
# ----------------------------------------------------------------------------
ifr_lookup = {}
for run in data:
    if run['Condition'] == 'Non-hyperemic':
        try:
            key = (round(float(run['Stenosis Percentage']), 3),
                   round(float(run['Length']), 3))
        except ValueError:
            continue

        try:
            avg_flow = float(run['Average Flow']) if run['Average Flow'] != '' else None
            max_flow_val = float(run['Max Flow']) if run['Max Flow'] != '' else None
            hmr_value = float(run['HMR']) if run['HMR'] != '' else None
        except ValueError:
            avg_flow = None
            max_flow_val = None
            hmr_value = None

        ifr_lookup[key] = {
            'Average Flow': avg_flow,
            'Max Flow': max_flow_val,
            'HMR': hmr_value
        }

# For each FFR (Hyperemic) run, find a matching iFR (Non-hyperemic) run
for run in data:
    if run['Condition'] == 'Hyperemic' and run['source'] == 'mine':
        try:
            key = (round(float(run['Stenosis Percentage']), 3),
                   round(float(run['Length']), 3))
        except ValueError:
            continue

        matching_ifr = ifr_lookup.get(key)
        if not matching_ifr:
            matching_key = find_key_within_tolerance(ifr_lookup, key, tolerance=0.003)
            if matching_key:
                matching_ifr = ifr_lookup[matching_key]

        if matching_ifr:
            # a) Compute CFR = (hyperemic max_flow / baseline max_flow)
            try:
                ffr_max_flow = float(run['Max Flow']) if run['Max Flow'] != '' else None
                ifr_max_flow = matching_ifr['Max Flow']
                if (ffr_max_flow is not None) and (ifr_max_flow is not None) and (ifr_max_flow != 0):
                    cfr = ffr_max_flow / ifr_max_flow
                    run['CFR'] = cfr
                else:
                    run['CFR'] = ''
            except ValueError:
                run['CFR'] = ''

            # b) Compute BMR/HMR = (IFR run HMR) / (FFR run HMR)
            try:
                ffr_hmr = float(run['HMR'])
                ifr_hmr = matching_ifr['HMR']
                if (ffr_hmr is not None) and (ifr_hmr is not None) and (ffr_hmr != 0):
                    bmr_hmr = ifr_hmr / ffr_hmr
                    run['BMR/HMR'] = bmr_hmr
                else:
                    run['BMR/HMR'] = ''
            except ValueError:
                run['BMR/HMR'] = ''
        else:
            run['CFR'] = ''
            run['BMR/HMR'] = ''

        # c) Compute CFR/FFR
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
    elif run['Condition'] == 'Hyperemic' and run['source'] == 'garcia':
        pass
    else:
        run['CFR/FFR'] = ''

# ----------------------------------------------------------------------------
# 3) CREATE "discord" COLUMN
# ----------------------------------------------------------------------------
for run in data:
    run['discord'] = ''
    try:
        cfr_val = float(run['CFR'])
        ffr_val = float(run['P_d/P_a'])
    except (ValueError, TypeError):
        continue

    if cfr_val > 2 and ffr_val > 0.8:
        run['discord'] = 0
    elif cfr_val > 2 and ffr_val < 0.8:
        run['discord'] = 1
    elif cfr_val < 2 and ffr_val > 0.8:
        run['discord'] = 2
    elif cfr_val < 2 and ffr_val < 0.8:
        run['discord'] = 3
    else:
        pass  # exactly cfr=2 or ffr=0.8 => remains ''

# ----------------------------------------------------------------------------
# 4) APPEND DATA FROM garcia_data.csv
# ----------------------------------------------------------------------------
if garcia:
    print('!!!!!!!!!!!!!!!!ENTERED GARCIA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    largest_geom = 0
    for d in data:
        try:
            gn = int(d['Geometry Number'])
            if gn > largest_geom:
                largest_geom = gn
        except ValueError:
            pass

    next_geom_number = largest_geom + 1

    if os.path.exists(garcia_file):
        with open(garcia_file, 'r') as gf:
            reader = csv.DictReader(gf, delimiter=',')
            for row in reader:
                hmr_g = row.get('HMR', '')
                ffr_g = row.get('FFR', '')
                cfr_g = row.get('CFR', '')
                bmr_hmr_g = row.get('BMR/HMR', '')
                p_loss_coeff = row.get('HSR_Coeff', '')

                try:
                    cfr_ffr_g = float(cfr_g) / float(ffr_g)
                except (ValueError, ZeroDivisionError):
                    cfr_ffr_g = ''

                discord_value = ''
                try:
                    cfr_val = float(cfr_g)
                    ffr_val = float(ffr_g)
                    if cfr_val > 2 and ffr_val > 0.8:
                        discord_value = 0
                    elif cfr_val > 2 and ffr_val < 0.8:
                        discord_value = 1
                    elif cfr_val < 2 and ffr_val > 0.8:
                        discord_value = 2
                    elif cfr_val < 2 and ffr_val < 0.8:
                        discord_value = 3
                except ValueError:
                    pass

                garcia_dict = {
                    'Condition': 'Hyperemic',
                    'Geometry Number': str(next_geom_number),
                    'Stenosis Percentage': '',
                    'Length': '',
                    'Width': '',
                    'Average Flow': '',
                    'Max Flow': '',
                    'P_d/P_a': ffr_g,
                    'HMR': hmr_g,
                    'HSR': '',
                    'P_Loss_Coeff': p_loss_coeff,
                    'WSS_TE': '',
                    'WSS_LE': '',
                    'WSS_TE_Area': '',
                    'WSS_LE_Area': '',
                    'Rtotal_cor Value': '',
                    'Location': '',
                    'CFR': cfr_g,
                    'BMR/HMR': bmr_hmr_g,
                    'CFR/FFR': cfr_ffr_g,
                    'WSS_Area_Bifur': '',
                    'WSS_Bif': '',
                    'WSS_LMB': '',
                    'WSS_min': '',
                    'WSS_TE_min': '',
                    'WSS_LE_min': '',
                    'WSS_TE_Area_min': '',
                    'WSS_LE_Area_min': '',
                    'WSS_Area_Bifur_min': '',
                    'v_distal': '',
                    'discord': discord_value,
                    'source': 'garcia',
                    # For Garcia, R_micro is just 0
                    'R_micro': 0
                }

                data.append(garcia_dict)
                next_geom_number += 1
    else:
        print(f"Garcia data file not found: {garcia_file}")

# ----------------------------------------------------------------------------
# 5) CALCULATE THE P_Loss_Coeff IF POSSIBLE (ONLY FOR 'mine')
# ----------------------------------------------------------------------------
rho = 1.06  # g/ml

for run in data:
    if run['source'] != 'mine':
        continue

    try:
        pd_pa_val = float(run['P_d/P_a'])  # e.g. FFR or iFR
        v_d = float(run['v_distal'])
        hmr_val = float(run['HMR'])

        if pd_pa_val == 0 or v_d == 0:
            continue

        P_d = hmr_val * v_d
        P_d *= 1333  # mmHg -> dyn/cm^2

        P_a = P_d / pd_pa_val
        delta_p = P_a - P_d
        denom = 0.5 * rho * (v_d ** 2)

        p_loss_coeff_calc = (delta_p) / denom
        p_loss_coeff_calc = np.log10(p_loss_coeff_calc)

        run['P_Loss_Coeff'] = p_loss_coeff_calc
    except (ValueError, ZeroDivisionError):
        run['P_Loss_Coeff'] = run['P_Loss_Coeff'] or ''

# ----------------------------------------------------------------------------
# 6) WRITE THE SUMMARY CSV FILE
# ----------------------------------------------------------------------------
header = [
    'Condition',
    'Geometry Number',
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'Max Flow',
    'P_d/P_a',
    'HMR',
    'HSR',
    'P_Loss_Coeff',
    'WSS_TE',
    'WSS_LE',
    'WSS_TE_Area',
    'WSS_LE_Area',
    'Rtotal_cor Value',
    'Location',
    'CFR',
    'BMR/HMR',
    'CFR/FFR',
    'WSS_Area_Bifur',
    'WSS_Bif',
    'WSS_LMB',
    'WSS_min',
    'WSS_TE_min',
    'WSS_LE_min',
    'WSS_TE_Area_min',
    'WSS_LE_Area_min',
    'WSS_Area_Bifur_min',
    'v_distal',
    'discord',
    'source',
    # NEW header column
    'R_micro'
]

# Replace 0 values with blank string for select WSS fields
for run in data:
    for key in ['WSS_LMB', 'WSS_min', 'WSS_TE_min', 'WSS_LE_min']:
        try:
            if float(run[key]) == 0.0:
                run[key] = ''
        except (ValueError, TypeError, KeyError):
            pass

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for run in data:
        row = [
            run['Condition'],
            run['Geometry Number'],
            run['Stenosis Percentage'],
            run['Length'],
            run['Width'],
            run['Average Flow'],
            run['Max Flow'],
            run['P_d/P_a'],
            run['HMR'],
            run['HSR'],
            run['P_Loss_Coeff'],
            run['WSS_TE'],
            run['WSS_LE'],
            run['WSS_TE_Area'],
            run['WSS_LE_Area'],
            run['Rtotal_cor Value'],
            run['Location'],
            run['CFR'],
            run['BMR/HMR'],
            run['CFR/FFR'],
            run['WSS_Area_Bifur'],
            run['WSS_Bif'],
            run['WSS_LMB'],
            run['WSS_min'],
            run['WSS_TE_min'],
            run['WSS_LE_min'],
            run['WSS_TE_Area_min'],
            run['WSS_LE_Area_min'],
            run['WSS_Area_Bifur_min'],
            run['v_distal'],
            run['discord'],
            run['source'],
            # NEW column
            run['R_micro']
        ]
        writer.writerow(row)

print(f"Summary file written to {output_file}")

if assumption_made_count > 0:
    print(
        f"Location assumption made for {assumption_made_count} simulation(s): "
        f"neither plaque_lad_ffr.csv nor plaque_lcx_ffr.csv was found. LAD assumed."
    )
