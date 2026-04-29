#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 23:56:31 2025

@author: tejjolly
"""
"""
Changes v_distal extraction to cycle average (from diastole)
"""
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Flags
garcia = True
binary_discord = False
trinary_discord = False
quaternary_discord = True
# Only one of the discord flags should be true

# Function
def find_key_within_tolerance(lookup, target_key, tolerance=0.003):
    for key in lookup.keys():
        if abs(key[0] - target_key[0]) <= tolerance and abs(key[1] - target_key[1]) <= tolerance:
            return key
    return None

# Directories
# Base directory
base_dir = '/Users/tejjolly/Documents/BioSimm/Simulations/'
# Output file path
output_file = os.path.join(base_dir, 'Post_Processing/data/data.csv')
# The Garcia data file (adjust path if needed)
garcia_file = os.path.join(base_dir, 'Post_Processing/data/garcia_data.csv')

# Simulations for manuscript (those that vary HMR)
manuscript_file = os.path.join(base_dir, 'Post_Processing/data/data_manuscript.csv')

### Pressure plot stuff
out_dir  = os.path.join(base_dir, 'Post_Processing/an-2024-00-00--scraper_vis/press_flow_plots')
split_plots = True # Split pressure plots into hyperemic and basal or false: keep combined
vary_styles = False # # True → vary linestyle + marker; False → all solid lines, no markers


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
            r_tot = ''
            location = ''

            # We still read in WSS-related values but do NOT store them if they
            # are just "WSS", "WSS_Avg_Area", or "WSS_Avg_Area_min".
            # This script no longer includes these three in the final output.
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

            v_distal = ''
            R_micro = 0.0
            R_scale = 0.24

            # 1) Set 'r_tot'
            R_scale_found = False
            if hyperemic_status == 'Non-hyperemic':
                R_scale = 0.24
            elif hyperemic_status == 'Hyperemic':
                R_scale = 0.24  # default if file not found
                lpn_file_path = os.path.join(item_path, 'WriteCoronaryLPN.py')
                if os.path.exists(lpn_file_path):
                    with open(lpn_file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            stripped_line = line.strip()
                            if stripped_line.startswith('Rtotal_cor =') and not R_scale_found:
                                match = re.search(r'\*\s*([0-9.]+)', stripped_line)
                                if match:
                                    R_scale = float(match.group(1))
                                    R_scale_found = True
                                else:
                                    print(f"Could not extract 'Rtotal_cor' value from line: {stripped_line}")
                            if 'Rtotal_cor_new' in stripped_line:
                                match_new = re.search(r'\*\s*([0-9.]+)', stripped_line)
                                if match_new:
                                    float_val = float(match_new.group(1))
                                    R_micro = float_val - R_scale
                                break
            r_tot = R_micro + R_scale
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
                wss_file = os.path.join(results_path, 'plaque_lcx_avg_wss_max.csv')   # Not stored in final
                te_wss_file = os.path.join(results_path, 'plaque_lcx_TE_wss_max.csv')
                le_wss_file = os.path.join(results_path, 'plaque_lcx_LE_wss_max.csv')
                high_te_area_file = os.path.join(results_path, 'plaque_high_lcx_TE_wss_area.csv')
                high_le_area_file = os.path.join(results_path, 'plaque_high_lcx_LE_wss_area.csv')
                mb_wss_file = os.path.join(results_path, 'high_mb_wss_area.csv')
                sb_wss_file = os.path.join(results_path, 'high_sb_wss_area.csv')
                avg_high_area_file = os.path.join(results_path, 'plaque_high_lcx_avg_wss_area.csv')  # Not stored in final

                # MIN WSS files
                wss_min_file = os.path.join(results_path, 'plaque_lcx_avg_wss_min.csv')
                te_wss_min_file = os.path.join(results_path, 'plaque_lcx_TE_wss_min.csv')
                le_wss_min_file = os.path.join(results_path, 'plaque_lcx_LE_wss_min.csv')
                te_wss_area_min_file = os.path.join(results_path, 'plaque_lcx_TE_wss_area.csv')
                le_wss_area_min_file = os.path.join(results_path, 'plaque_lcx_LE_wss_area.csv')
                mb_wss_area_min_file = os.path.join(results_path, 'mb_wss_area.csv')
                sb_wss_area_min_file = os.path.join(results_path, 'sb_wss_area.csv')
                avg_area_min_file = os.path.join(results_path, 'plaque_lcx_avg_wss_area.csv')  # Not stored in final

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
                avg_high_area_file = os.path.join(results_path, 'plaque_high_lad_avg_wss_area.csv')  # Not stored in final

                # MIN WSS files
                wss_min_file = os.path.join(results_path, 'plaque_lad_avg_wss_min.csv')
                te_wss_min_file = os.path.join(results_path, 'plaque_lad_TE_wss_min.csv')
                le_wss_min_file = os.path.join(results_path, 'plaque_lad_LE_wss_min.csv')
                te_wss_area_min_file = os.path.join(results_path, 'plaque_lad_TE_wss_area.csv')
                le_wss_area_min_file = os.path.join(results_path, 'plaque_lad_LE_wss_area.csv')
                mb_wss_area_min_file = os.path.join(results_path, 'mb_wss_area.csv')
                sb_wss_area_min_file = os.path.join(results_path, 'sb_wss_area.csv')
                avg_area_min_file = os.path.join(results_path, 'plaque_lad_avg_wss_area.csv')  # Not stored in final

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
                        if len(possible_values) > 0 and possible_values[0].strip():
                            v_temp = possible_values[0].strip()
                        v_distal = v_temp
                    else:
                        print(f"Unexpected format or empty v_distal file: {v_distal_file}")
            else:
                print(f"File not found: {v_distal_file}")

            # Default
            q_distal_tmp = ''
            try:
                v_num = float(v_distal)
                if location == "LAD":
                    q_distal_tmp = v_num * 0.095
                elif location == "LCX":
                    q_distal_tmp = v_num * 0.056
                else:
                    # if somehow unknown, leave blank (or pick a default if you prefer)
                    q_distal_tmp = ''
            except (ValueError, TypeError):
                q_distal_tmp = ''

            # -----------------------------------------------------------------
            # WSS (MAX) files – we read them but do NOT store the plain 'WSS'
            # or 'WSS_Avg_Area' in final output.
            # -----------------------------------------------------------------
            if os.path.exists(wss_file):
                with open(wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        # NOT storing plain 'wss' in final dictionary now
                        pass

            if os.path.exists(te_wss_file):
                with open(te_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {te_wss_file}")

            if os.path.exists(le_wss_file):
                with open(le_wss_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {le_wss_file}")

            # TE/LE "high" area (i.e., MAX area)
            if os.path.exists(high_te_area_file):
                with open(high_te_area_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te_area = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te_area = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {high_te_area_file}")

            if os.path.exists(high_le_area_file):
                with open(high_le_area_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le_area = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le_area = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {high_le_area_file}")

            # Bifur "high" areas (MAX area)
            wss_mb_area = ''
            wss_sb_area = ''
            mb_wss_file_exists = os.path.exists(mb_wss_file)
            sb_wss_file_exists = os.path.exists(sb_wss_file)

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
                        else:
                            print(f"Unexpected format in file: {mb_wss_file}")

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
                        else:
                            print(f"Unexpected format in file: {sb_wss_file}")

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
                        else:
                            print(f"Unexpected format in file: {bif_mb_wss_file}")

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
                        else:
                            print(f"Unexpected format in file: {bif_sb_wss_file}")

            if wss_mb_bif_value != '' and wss_sb_bif_value != '':
                wss_bif_temp = wss_mb_bif_value + wss_sb_bif_value
                if wss_bif_temp == 0.0:
                    wss_bif = ''
                else:
                    wss_bif = wss_bif_temp

            # We won't store plain 'WSS_Avg_Area'
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
                        else:
                            print(f"Unexpected format in file: {lmb_wss_file}")

            # -----------------------------------------------------------------
            # READ MIN VALUES (we still keep WSS_min if you need it)
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
                        else:
                            print(f"Unexpected format in file: {wss_min_file}")

            if os.path.exists(te_wss_min_file):
                with open(te_wss_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te_min = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {te_wss_min_file}")

            if os.path.exists(le_wss_min_file):
                with open(le_wss_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le_min = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {le_wss_min_file}")

            if os.path.exists(te_wss_area_min_file):
                with open(te_wss_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_te_area_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_te_area_min = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {te_wss_area_min_file}")

            if os.path.exists(le_wss_area_min_file):
                with open(le_wss_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        if len(rows[1]) >= 6:
                            wss_le_area_min = rows[1][5]
                        elif len(rows[1]) >= 3:
                            wss_le_area_min = rows[1][2]
                        else:
                            print(f"Unexpected format in file: {le_wss_area_min_file}")

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
                        else:
                            print(f"Unexpected format in file: {mb_wss_area_min_file}")

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
                        else:
                            print(f"Unexpected format in file: {sb_wss_area_min_file}")

            if mb_wss_area_min != '' and sb_wss_area_min != '':
                wss_area_bifur_min = mb_wss_area_min + sb_wss_area_min
            else:
                wss_area_bifur_min = ''

            if os.path.exists(avg_area_min_file):
                with open(avg_area_min_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) >= 2:
                        # Not storing 'WSS_Avg_Area_min' in final dictionary
                        pass

            # 5) Create data dictionary
            data_dict = {
                'Condition': hyperemic_status,
                'Geometry Number': geometry_number,
                'Location': location,
                'Stenosis Percentage': stenosis_percentage,
                'Length': length,
                'Width': width,
                'R_micro': R_micro,
                'R_scale': R_scale,
                'R_total': r_tot,

                'P_d/P_a': pd_pa,
                'CFR': '',
                'CFR/FFR': '',
                'discord': '',

                'HMR': hmr,
                'BMR/HMR': '',

                'HSR': hsr,
                'P_Loss_Coeff': p_loss_coeff,

                'Average Flow': average_flow,
                'Max Flow': max_flow,
                'v_distal': v_distal,
                'Q_distal': q_distal_tmp,

                'WSS_Bif': wss_bif,
                'WSS_LMB': wss_lmb,
                'WSS_TE': wss_te,
                'WSS_LE': wss_le,
                'WSS_Area_Bifur': wss_area_bifur,
                'WSS_TE_Area': wss_te_area,
                'WSS_LE_Area': wss_le_area,
                'WSS_TE_min': wss_te_min,
                'WSS_LE_min': wss_le_min,
                'WSS_TE_Area_min': wss_te_area_min,
                'WSS_LE_Area_min': wss_le_area_min,
                'WSS_Area_Bifur_min': wss_area_bifur_min,
                'WSS_min': wss_min,
                'source': 'mine'
            }

            data.append(data_dict)

# ----------------------------------------------------------------------------
# 2) CALCULATE CFR, BMR/HMR FOR HYPEREMIC RUNS BY MATCHING NON-HYPEREMIC
# ----------------------------------------------------------------------------
# Build a two-level lookup: ifr_lookup[Location][(Stenosis, Length)] -> metrics
ifr_lookup: dict[str, dict[tuple[float, float], dict]] = {}

for run in data:
    if run['Condition'] != 'Non-hyperemic' or run.get('source') != 'mine':
        continue

    loc = str(run.get('Location', '')).strip()
    if not loc:
        # If location is missing, skip this row to avoid cross-vessel matches.
        continue

    # Prepare the numerical key
    try:
        key_num = (
            round(float(run['Stenosis Percentage']), 3),
            round(float(run['Length']), 3),
        )
    except (ValueError, TypeError):
        continue

    # Parse fields, tolerating blanks
    try:
        avg_flow = float(run['Average Flow']) if run['Average Flow'] != '' else None
    except (ValueError, TypeError):
        avg_flow = None
    try:
        max_flow_val = float(run['Max Flow']) if run['Max Flow'] != '' else None
    except (ValueError, TypeError):
        max_flow_val = None
    try:
        hmr_value = float(run['HMR']) if run['HMR'] != '' else None
    except (ValueError, TypeError):
        hmr_value = None
    try:
        v_distal_val = float(run['v_distal']) if run['v_distal'] != '' else None
    except (ValueError, TypeError):
        v_distal_val = None

    ifr_lookup.setdefault(loc, {})[key_num] = {
        'Average Flow': avg_flow,
        'Max Flow': max_flow_val,
        'HMR': hmr_value,
        'v_distal': v_distal_val,
    }

# For each FFR (Hyperemic) run, find a matching iFR (Non-hyperemic) run in the SAME vessel
for run in data:
    if not (run['Condition'] == 'Hyperemic' and run.get('source') == 'mine'):
        # Keep CFR/FFR empty for non-hyperemic and non-mine hyperemic, as before
        if run['Condition'] != 'Hyperemic' or run.get('source') != 'garcia':
            run['CFR/FFR'] = ''
        continue

    loc = str(run.get('Location', '')).strip()
    if not loc:
        # Without a location, we cannot enforce vessel-matching; leave blank
        run['CFR'] = ''
        run['BMR/HMR'] = ''
        run['CFR/FFR'] = ''
        continue

    # Numeric key (stenosis, length)
    try:
        key_num = (
            round(float(run['Stenosis Percentage']), 3),
            round(float(run['Length']), 3),
        )
    except (ValueError, TypeError):
        continue

    sub_lookup = ifr_lookup.get(loc, {})  # Restrict to SAME vessel
    matching_ifr = sub_lookup.get(key_num)

    if not matching_ifr:
        # Fall back to tolerance search within SAME vessel only
        matching_key = find_key_within_tolerance(sub_lookup, key_num, tolerance=0.003)
        if matching_key:
            matching_ifr = sub_lookup.get(matching_key)

    if matching_ifr:
        # a) CFR = v_distal(hyperemic) / v_distal(baseline)
        try:
            v_distal_hyp = float(run['v_distal']) if run['v_distal'] != '' else None
        except (ValueError, TypeError):
            v_distal_hyp = None
        v_distal_base = matching_ifr.get('v_distal', None)

        if (v_distal_hyp is not None) and (v_distal_base is not None) and (v_distal_base != 0):
            run['CFR'] = v_distal_hyp / v_distal_base
        else:
            run['CFR'] = ''

        # b) BMR/HMR = HMR(baseline iFR) / HMR(hyperemic FFR)
        try:
            ffr_hmr = float(run['HMR'])
        except (ValueError, TypeError):
            ffr_hmr = None
        ifr_hmr = matching_ifr.get('HMR', None)

        if (ffr_hmr is not None) and (ifr_hmr is not None) and (ffr_hmr != 0):
            run['BMR/HMR'] = ifr_hmr / ffr_hmr
        else:
            run['BMR/HMR'] = ''
    else:
        run['CFR'] = ''
        run['BMR/HMR'] = ''

    # c) CFR/FFR
    try:
        if run['CFR'] != '' and run['P_d/P_a'] != '':
            cfr_val = float(run['CFR'])
            ffr_val = float(run['P_d/P_a'])
            run['CFR/FFR'] = (cfr_val / ffr_val) if ffr_val != 0 else ''
        else:
            run['CFR/FFR'] = ''
    except (ValueError, TypeError):
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
    if binary_discord:
        if cfr_val >= 2 and ffr_val >= 0.8:
            run['discord'] = 0
        elif cfr_val >= 2 and ffr_val < 0.8:
            run['discord'] = 1
        elif cfr_val < 2 and ffr_val >= 0.8:
            run['discord'] = 1
        elif cfr_val < 2 and ffr_val < 0.8:
            run['discord'] = 0
        else:
            pass
    elif trinary_discord:
        if cfr_val >= 2 and ffr_val >= 0.8:
            run['discord'] = 0
        elif cfr_val >= 2 and ffr_val < 0.8:
            run['discord'] = 1
        elif cfr_val < 2 and ffr_val >= 0.8:
            run['discord'] = 1
        elif cfr_val < 2 and ffr_val < 0.8:
            run['discord'] = 2
        else:
            pass

    elif quaternary_discord:
        if cfr_val >= 2 and ffr_val >= 0.8:
            run['discord'] = 0
        elif cfr_val >= 2 and ffr_val < 0.8:
            run['discord'] = 1
        elif cfr_val < 2 and ffr_val >= 0.8:
            run['discord'] = 2
        elif cfr_val < 2 and ffr_val < 0.8:
            run['discord'] = 3
        else:
            pass

# ----------------------------------------------------------------------------
# 4) APPEND DATA FROM garcia_data.csv
# ----------------------------------------------------------------------------
if garcia:
    print('!!!!!!!!!!!!!!!!ENTERED GARCIA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Find the largest geometry number among your data
    largest_geom = 0
    for d in data:
        try:
            gn = int(d['Geometry Number'])
            if gn > largest_geom:
                largest_geom = gn
        except ValueError:
            pass  # ignore non-numeric geometry numbers if present

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

                # We'll also compute "discord" for Garcia rows:
                discord_value = ''
                try:
                    cfr_val = float(cfr_g)
                    ffr_val = float(ffr_g)
                    if binary_discord:
                        if cfr_val >= 2 and ffr_val >= 0.8:
                            discord_value = 0
                        elif cfr_val >= 2 and ffr_val < 0.8:
                            discord_value = 1
                        elif cfr_val < 2 and ffr_val >= 0.8:
                            discord_value = 1
                        elif cfr_val < 2 and ffr_val < 0.8:
                            discord_value = 0

                    if trinary_discord:
                        if cfr_val >= 2 and ffr_val >= 0.8:
                            discord_value = 0
                        elif cfr_val >= 2 and ffr_val < 0.8:
                            discord_value = 1
                        elif cfr_val < 2 and ffr_val >= 0.8:
                            discord_value = 1
                        elif cfr_val < 2 and ffr_val < 0.8:
                            discord_value = 2

                    elif quaternary_discord:
                        if cfr_val >= 2 and ffr_val >= 0.8:
                            discord_value = 0
                        elif cfr_val >= 2 and ffr_val < 0.8:
                            discord_value = 1
                        elif cfr_val < 2 and ffr_val >= 0.8:
                            discord_value = 2
                        elif cfr_val < 2 and ffr_val < 0.8:
                            discord_value = 3
                except ValueError:
                    pass

                # Create a new "Hyperemic" row with source = garcia
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
                    # REMOVED: 'WSS'
                    'WSS_TE': '',
                    'WSS_LE': '',
                    'WSS_TE_Area': '',
                    'WSS_LE_Area': '',
                    'R_micro': '',
                    'R_scale': '',
                    'R_total': '',
                    'Location': '',
                    'CFR': cfr_g,
                    'BMR/HMR': bmr_hmr_g,
                    'CFR/FFR': cfr_ffr_g,
                    'WSS_Area_Bifur': '',
                    'WSS_Bif': '',
                    # REMOVED: 'WSS_Avg_Area'
                    'WSS_LMB': '',

                    'WSS_min': '',
                    'WSS_TE_min': '',
                    'WSS_LE_min': '',
                    'WSS_TE_Area_min': '',
                    'WSS_LE_Area_min': '',
                    'WSS_Area_Bifur_min': '',
                    # REMOVED: 'WSS_Avg_Area_min'

                    'v_distal': '',
                    'Q_distal': '',
                    'discord': discord_value,
                    'source': 'garcia'
                }

                data.append(garcia_dict)
                next_geom_number += 1
    else:
        print(f"Garcia data file not found: {garcia_file}")

# ----------------------------------------------------------------------------
# 5) CALCULATE THE P_Loss_Coeff IF POSSIBLE (ONLY FOR 'mine')
# ----------------------------------------------------------------------------
rho = 1.06  # g/ml (as requested)

for run in data:
    # If this row is from garcia, do NOT overwrite P_Loss_Coeff
    if run['source'] != 'mine':
        continue

    # Only try if we have valid ratio P_d/P_a, HMR, and v_distal
    try:
        pd_pa_val = float(run['P_d/P_a'])   # e.g. FFR or iFR
        v_d = float(run['v_distal'])
        hmr_val = float(run['HMR'])

        # If any are zero, skip
        if pd_pa_val == 0 or v_d == 0:
            continue

        # P_d = (HMR) * (v_d)
        P_d = hmr_val * v_d
        P_d *= 1333

        # P_a = P_d / (P_d/P_a)
        P_a = P_d / pd_pa_val

        # Numerator
        delta_p = P_a - P_d

        # Denominator
        denom = 0.5 * rho * (v_d ** 2)

        # Pressure loss coefficient (log scale)
        p_loss_coeff_calc = (delta_p) / denom
        p_loss_coeff_calc = np.log10(p_loss_coeff_calc)

        run['P_Loss_Coeff'] = p_loss_coeff_calc  # Overwrite any old value
    except (ValueError, ZeroDivisionError):
        run['P_Loss_Coeff'] = run['P_Loss_Coeff'] or ''

# ----------------------------------------------------------------------------
# 6) WRITE THE SUMMARY CSV FILE (excluding WSS, WSS_Avg_Area, and WSS_Avg_Area_min)
# ----------------------------------------------------------------------------
header = [
    'Condition',
    'Geometry Number',
    'Location',
    'Stenosis Percentage',
    'Length',
    'Width',

    'R_scale',
    'R_micro',
    'R_total',

    'P_d/P_a',
    'CFR',
    'CFR/FFR',
    'discord',

    'HMR',
    'BMR/HMR',

    'HSR',
    'P_Loss_Coeff',

    'Average Flow',
    'Max Flow',
    'v_distal',
    'Q_distal',


    'WSS_LMB',
    'WSS_Bif',
    'WSS_LE',
    'WSS_TE',

    'WSS_min',
    'WSS_LE_min',
    'WSS_TE_min',

    'WSS_Area_Bifur',
    'WSS_LE_Area',
    'WSS_TE_Area',

    'WSS_Area_Bifur_min',
    'WSS_LE_Area_min',
    'WSS_TE_Area_min',

    'source'
]

# Replace 0 values with blank string for select WSS fields
for run in data:
    for key in ['WSS_LMB', 'WSS_min', 'WSS_TE_min', 'WSS_LE_min']:
        try:
            if float(run[key]) == 0.0:
                run[key] = ''
        except (ValueError, TypeError, KeyError):
            pass  # Leave as-is if not a number or not present


with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for run in data:
        row = [
            run['Condition'],
            run['Geometry Number'],
            run['Location'],
            run['Stenosis Percentage'],
            run['Length'],
            run['Width'],

            run['R_scale'],
            run['R_micro'],
            run['R_total'],


            run['P_d/P_a'],
            run['CFR'],
            run['CFR/FFR'],
            run['discord'],

            run['HMR'],
            run['BMR/HMR'],

            run['HSR'],
            run['P_Loss_Coeff'],

            run['Average Flow'],
            run['Max Flow'],
            run['v_distal'],
            run['Q_distal'],

            run['WSS_LMB'],
            run['WSS_Bif'],
            run['WSS_LE'],
            run['WSS_TE'],

            run['WSS_min'],
            run['WSS_LE_min'],
            run['WSS_TE_min'],

            run['WSS_Area_Bifur'],
            run['WSS_LE_Area'],
            run['WSS_TE_Area'],

            run['WSS_Area_Bifur_min'],
            run['WSS_LE_Area_min'],
            run['WSS_TE_Area_min'],

            run['source']
        ]
        writer.writerow(row)

print(f"Summary file written to {output_file}")

if assumption_made_count > 0:
    print(
        f"Location assumption made for {assumption_made_count} simulation(s): "
        f"neither plaque_lad_ffr.csv nor plaque_lcx_ffr.csv was found. LAD assumed."
    )

# ----------------------------------------------------------------------------
# WRITE SELECTED ROWS TO data_manuscript.csv (config via two lists)
# ----------------------------------------------------------------------------

# Edit these two lists as needed; strings or ints are both OK.
MANU_NH = ["1", "13", "18", "26", "37",
           # "48", "49", "50", "51", # LCX
           "52", "53", "54", "55"]
MANU_H  = [
    "1", "13", "18", "26", "31", "32", "33", "34", "35", "36",
    "38", "39", "40", "41", "42", "43", "44", "45", "46", "47",
    # "52","53","54","55","56","57","58","59", # LCX
    "117", "118", "119", "120", "121", "122", "123", "124",
    "125", "126", "127", "128", "129", "130", "131", "132"]

# Build the exact writing order you want:
manuscript_pairs = (
    [("Non-hyperemic", str(g)) for g in MANU_NH] +
    [("Hyperemic",     str(g)) for g in MANU_H]
)

# Build quick index: (Condition, Geometry Number) -> list of runs
index = {}
for run in data:
    key = (str(run.get('Condition', '')).strip(), str(run.get('Geometry Number', '')).strip())
    index.setdefault(key, []).append(run)

missing = []  # keep track of (cond, geom) we couldn't find

with open(manuscript_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # reuse your existing header

    # Write rows in the exact order (and with duplicates) as provided above
    for cond, geom in manuscript_pairs:
        key = (cond, geom)
        if key not in index or len(index[key]) == 0:
            missing.append(key)
            continue

        # If multiple runs exist for the same (cond, geom), pick the first.
        run = index[key][0]

        row = [
            run.get('Condition', ''),
            run.get('Geometry Number', ''),
            run.get('Location', ''),
            run.get('Stenosis Percentage', ''),
            run.get('Length', ''),
            run.get('Width', ''),

            run.get('R_scale', ''),
            run.get('R_micro', ''),
            run.get('R_total', ''),

            run.get('P_d/P_a', ''),
            run.get('CFR', ''),
            run.get('CFR/FFR', ''),
            run.get('discord', ''),

            run.get('HMR', ''),
            run.get('BMR/HMR', ''),

            run.get('HSR', ''),
            run.get('P_Loss_Coeff', ''),

            run.get('Average Flow', ''),
            run.get('Max Flow', ''),
            run.get('v_distal', ''),
            run.get('Q_distal', ''),

            run.get('WSS_LMB', ''),
            run.get('WSS_Bif', ''),
            run.get('WSS_LE', ''),
            run.get('WSS_TE', ''),

            run.get('WSS_min', ''),
            run.get('WSS_LE_min', ''),
            run.get('WSS_TE_min', ''),

            run.get('WSS_Area_Bifur', ''),
            run.get('WSS_LE_Area', ''),
            run.get('WSS_TE_Area', ''),

            run.get('WSS_Area_Bifur_min', ''),
            run.get('WSS_LE_Area_min', ''),
            run.get('WSS_TE_Area_min', ''),

            run.get('source', '')
        ]
        writer.writerow(row)

print(f"Manuscript subset written to {manuscript_file}")
if missing:
    print("WARNING: The following (Condition, Geometry) pairs were not found and were skipped:")
    for m in missing:
        print(f"  {m}")

"""PRESSURE PLOTTING BLOCK"""
plot_size = (8,4)
line_width = 1
cond2master = {"Non-hyperemic": "master_ifr", "Hyperemic": "master_ffr"}
def read_pressure_series(geom_dir, normalize_time_zero=True):
    """
    Returns time (s) from 'step' * 0.001 and inlet (last column) pressure (mmHg).
    If normalize_time_zero=True, time starts at 0 at the first row.
    """
    press_path = os.path.join(geom_dir, 'results-processed-new', 'all_results-pressure.txt')
    if not os.path.exists(press_path):
        return None, None

    df = pd.read_csv(press_path, sep=r"\s+")
    if 'step' not in df.columns:
        # Fallback: synthesize time if 'step' missing
        t = np.arange(len(df)) * 0.001
    else:
        step = df['step'].to_numpy(dtype=float)
        t = step * 0.001   # real seconds
        if normalize_time_zero and len(t) > 0:
            t = t - t[0]   # start at 0 s

        # keep 'step' for time; drop it from signals
        df = df.drop(columns=['step'])

    # inlet is the last column
    p_inlet = (df.iloc[:, -1].to_numpy(dtype=float)) / 1333.0  # mmHg
    return t, p_inlet

# Style cycles (only used if vary_styles=True)
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', 'd', '^', 'v', '<', '>', 'x', '+', '*']

missing = []

def make_pressure_plot(cond_filter=None, filename_suffix="", add_legend=True, title=False):
    fig, ax = plt.subplots(figsize=plot_size)
    style_idx = 0
    for cond, geom in manuscript_pairs:
        if cond_filter and cond != cond_filter:
            continue
        master = cond2master[cond]
        geom_dir = os.path.join(base_dir, master, f'Geometry_{geom}')
        t, p = read_pressure_series(geom_dir, normalize_time_zero=False)
        if t is None:
            missing.append((cond, geom))
            continue

        label = f"{'NH' if cond=='Non-hyperemic' else 'H'}-G{geom}"

        if vary_styles:
            ls = linestyles[style_idx % len(linestyles)]
            mk = markers[style_idx % len(markers)]
            style_idx += 1
            ax.plot(t, p, linestyle=ls, marker=mk, markevery=2, linewidth=1.4, label=label)
        else:
            ax.plot(t, p, linewidth=line_width, label=label)  # solid, no markers

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pressure [mmHg]')

    if title:
        if cond_filter:
            ax.set_title(f"Pressure vs Time ({cond_filter} simulations)")
        else:
            ax.set_title("Pressure vs Time")
    # ax.grid(True, alpha=0.25)
    if add_legend:
        ax.legend(ncol=2, fontsize=8, frameon=False)
    out_path = os.path.join(out_dir, f'inlet_pressure{filename_suffix}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {out_path}")

if split_plots:
    make_pressure_plot(cond_filter="Non-hyperemic", filename_suffix="_non_hyperemic", add_legend=False)
    make_pressure_plot(cond_filter="Hyperemic", filename_suffix="_hyperemic", add_legend=False)
else:
    make_pressure_plot(cond_filter=None, filename_suffix="", add_legend=False)

if missing:
    print("WARN: Missing pressure files for:", missing)

# ===== FLOW (cm^3/s) PLOTS: inlet = last column, time from 'step' * 0.001 =====

def read_flow_series(geom_dir, normalize_time_zero=True, flip_sign=True):
    """
    Returns time (s) from 'step' * 0.001 and inlet (last column) flow (cm^3/s).
    If normalize_time_zero=True, time starts at 0 at the first row.
    flip_sign=True if your file convention uses negative outflow and you want to flip it.
    """
    flow_path = os.path.join(geom_dir, 'results-processed-new', 'all_results-flows.txt')
    if not os.path.exists(flow_path):
        return None, None

    df = pd.read_csv(flow_path, sep=r"\s+")
    if 'step' not in df.columns:
        t = np.arange(len(df)) * 0.001
    else:
        step = df['step'].to_numpy(dtype=float)
        t = step * 0.001
        if normalize_time_zero and len(t) > 0:
            t = t - t[0]
        df = df.drop(columns=['step'])

    q_inlet = df.iloc[:, -1].to_numpy(dtype=float)  # cm^3/s
    if flip_sign:
        q_inlet = -q_inlet
    return t, q_inlet

flow_missing = []

def make_flow_plot(cond_filter=None, filename_suffix="", add_legend=True, flip_sign=True, title=False):
    fig, ax = plt.subplots(figsize=plot_size)
    style_idx = 0
    for cond, geom in manuscript_pairs:
        if cond_filter and cond != cond_filter:
            continue
        master = cond2master[cond]
        geom_dir = os.path.join(base_dir, master, f'Geometry_{geom}')
        t, q = read_flow_series(geom_dir, normalize_time_zero=False, flip_sign=flip_sign)
        if t is None:
            flow_missing.append((cond, geom))
            continue

        label = f"{'NH' if cond=='Non-hyperemic' else 'H'}-G{geom}"

        if vary_styles:
            ls = linestyles[style_idx % len(linestyles)]
            mk = markers[style_idx % len(markers)]
            style_idx += 1
            ax.plot(t, q, linestyle=ls, marker=mk, markevery=2, linewidth=1.4, label=label)
        else:
            ax.plot(t, q, linewidth=line_width, label=label)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Flow [cm³/s]')
    if title:
        if cond_filter:
            ax.set_title(f"Flow vs Time ({cond_filter} simulations)")
        else:
            ax.set_title("Flow vs Time")
    if add_legend:
        ax.legend(ncol=2, fontsize=8, frameon=False)
    out_path = os.path.join(out_dir, f'inlet_flow{filename_suffix}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {out_path}")

# Use the same split behavior as your pressure plots.
# Set flip_sign=True if your file stores outflow as negative and you want positive upward.
if split_plots:
    make_flow_plot(cond_filter="Non-hyperemic", filename_suffix="_non_hyperemic", add_legend=False, flip_sign=True)
    make_flow_plot(cond_filter="Hyperemic", filename_suffix="_hyperemic", add_legend=False, flip_sign=True)
else:
    make_flow_plot(cond_filter=None, filename_suffix="", add_legend=False, flip_sign=True)

if flow_missing:
    print("WARN: Missing flow files for:", flow_missing)


# ===== COMBINED PLOTS: Pressure (solid, left y-axis) + Flow (dotted, right y-axis) =====

combo_missing = []

def make_combo_plot(cond_filter=None, filename_suffix="", add_legend=True, flip_sign=True):
    fig, ax1 = plt.subplots(figsize=plot_size)
    ax2 = ax1.twinx()  # secondary y-axis for flow

    style_idx = 0
    for cond, geom in manuscript_pairs:
        if cond_filter and cond != cond_filter:
            continue
        master = cond2master[cond]
        geom_dir = os.path.join(base_dir, master, f'Geometry_{geom}')

        # inlet pressure
        t_p, p = read_pressure_series(geom_dir, normalize_time_zero=False)
        # inlet flow
        t_q, q = read_flow_series(geom_dir, normalize_time_zero=False, flip_sign=flip_sign)

        if t_p is None or t_q is None:
            combo_missing.append((cond, geom))
            continue

        label = f"{'NH' if cond=='Non-hyperemic' else 'H'}-G{geom}"

        if vary_styles:
            ls = linestyles[style_idx % len(linestyles)]
            mk = markers[style_idx % len(markers)]
            style_idx += 1
            # Pressure
            ax1.plot(t_p, p, linestyle=ls, marker=mk, markevery=2,
                     linewidth=1.4, label=f"{label} (P)")
            # Flow
            ax2.plot(t_q, q, linestyle=":", alpha=0.5,
                     linewidth=1.4, label=f"{label} (Q)")
        else:
            # Pressure
            ax1.plot(t_p, p, linewidth=1.6, label=f"{label} (P)")
            # Flow
            ax2.plot(t_q, q, linestyle=":", alpha=0.5,
                     linewidth=1.4, label=f"{label} (Q)")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Pressure [mmHg]")
    ax2.set_ylabel("Flow [cm³/s]")

    if cond_filter:
        ax1.set_title(f"Pressure + Flow vs Time ({cond_filter} simulations)")
    else:
        ax1.set_title("Pressure + Flow vs Time")

    if add_legend:
        # Merge legends from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1+handles2, labels1+labels2, ncol=2, fontsize=7, frameon=False)

    out_path = os.path.join(out_dir, f'inlet_pressure_and_flow{filename_suffix}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {out_path}")

# Use same split flag logic as before
if split_plots:
    make_combo_plot(cond_filter="Non-hyperemic", filename_suffix="_non_hyperemic", add_legend=False, flip_sign=True)
    make_combo_plot(cond_filter="Hyperemic", filename_suffix="_hyperemic", add_legend=True, flip_sign=True)
else:
    make_combo_plot(cond_filter=None, filename_suffix="", add_legend=False, flip_sign=True)

if combo_missing:
    print("WARN: Missing pressure/flow files for:", combo_missing)
