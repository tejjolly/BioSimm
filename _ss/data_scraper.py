import os
import csv
import re

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
            hms = ''
            rtotal_cor_value = ''

            # Set 'rtotal_cor_value' based on conditions
            if hyperemic_status == 'Non-hyperemic':
                # For all 'master_ifr' runs, set value to 1.0
                rtotal_cor_value = 1.0
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
                                flows.append(float(flow_value))
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
                        hms = rows[2][0]
                    else:
                        print(f"Unexpected format in file: {hmr_file}")
            else:
                print(f"File not found: {hmr_file}")

            # Append the data to the list
            data.append([
                hyperemic_status,
                geometry_number,
                stenosis_percentage,
                length,
                width,
                average_flow,
                pd_pa,
                hmr,
                hms,
                rtotal_cor_value  # New column added here
            ])

# Write the data to CSV file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow([
        'Condition',
        'Geometry Number',
        'Stenosis Percentage',
        'Length',
        'Width',
        'Average Flow',
        'P_d/P_a',
        'HMR',
        'HMS',
        'Rtotal_cor Value'  # New column added here
    ])
    # Write data
    writer.writerows(data)

print(f"Summary file written to {output_file}")
