#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############################################################
# 1) LOAD DATA
############################################################
df = pd.read_csv("summary.csv")

# Separate iFR (non‐hyperemic) vs FFR (hyperemic)
df_ifr = df[df['Condition'] == 'Non-hyperemic'].copy()
df_ffr = df[df['Condition'] == 'Hyperemic'].copy()

def make_key(row):
    """Round (Stenosis, Length) to 3 decimals, or return None on error."""
    try:
        sten = round(float(row['Stenosis Percentage']), 3)
        leng = round(float(row['Length']), 3)
        return (sten, leng)
    except:
        return None

df_ifr['merge_key'] = df_ifr.apply(make_key, axis=1)
df_ffr['merge_key'] = df_ffr.apply(make_key, axis=1)

df_ifr = df_ifr.dropna(subset=['merge_key'])
df_ffr = df_ffr.dropna(subset=['merge_key'])

# Build a dictionary of iFR runs for matching
ifr_dict = {}
for idx, row in df_ifr.iterrows():
    k = row['merge_key']
    if k not in ifr_dict:
        ifr_dict[k] = []
    ifr_dict[k].append(idx)

def find_key_within_tolerance(lookup_keys, target_key, tolerance=0.003):
    """Return the first key from 'lookup_keys' that is within tolerance of target_key."""
    for k in lookup_keys:
        if abs(k[0] - target_key[0]) <= tolerance and abs(k[1] - target_key[1]) <= tolerance:
            return k
    return None

pairs = []
for idx_ffr, row_ffr in df_ffr.iterrows():
    k_ffr = row_ffr['merge_key']
    if k_ffr in ifr_dict:
        # direct match
        for idx_ifr in ifr_dict[k_ffr]:
            pairs.append((idx_ffr, idx_ifr))
    else:
        # fallback tolerance
        possible_k = find_key_within_tolerance(ifr_dict.keys(), k_ffr, tolerance=0.003)
        if possible_k is not None:
            for idx_ifr in ifr_dict[possible_k]:
                pairs.append((idx_ffr, idx_ifr))

merged = pd.DataFrame({
    'ffr_index': [p[0] for p in pairs],
    'ifr_index': [p[1] for p in pairs]
})
merged.drop_duplicates(inplace=True)

print("Number of matched (FFR,iFR) pairs:", len(merged))

# 2) Map columns from each original df
for col in df_ffr.columns:
    merged[f"{col}_ffr"] = merged['ffr_index'].map(df_ffr[col])
for col in df_ifr.columns:
    merged[f"{col}_ifr"] = merged['ifr_index'].map(df_ifr[col])

############################################################
# 3) PREPARE FIELDS FOR eqn(7) & eqn(B)
############################################################

# Equation (7):
# Observed: CFR/FFR   => CFR_ffr / P_d/P_a_ffr
# Predicted: (1/FFR) + (BMR/IMR) - 1
# BMR = HMR_ifr, IMR=HMR_ffr

merged['CFR_ffr'] = pd.to_numeric(merged['CFR_ffr'], errors='coerce')
merged['FFR_val'] = pd.to_numeric(merged['P_d/P_a_ffr'], errors='coerce')
merged['BMR'] = pd.to_numeric(merged['HMR_ifr'], errors='coerce')
merged['IMR'] = pd.to_numeric(merged['HMR_ffr'], errors='coerce')

merged['Obs_eqn7'] = merged['CFR_ffr'] / merged['FFR_val']

def eqn7_pred(row):
    try:
        ffrv = row['FFR_val']
        bmr  = row['BMR']
        imr  = row['IMR']
        if ffrv and ffrv!=0 and imr and imr!=0:
            return (1.0/ffrv) + (bmr/imr) - 1.0
    except:
        pass
    return np.nan

merged['Pred_eqn7'] = merged.apply(eqn7_pred, axis=1)

# Equation (B):
# Observed_B = FFR_val / NHi
# => NHi = P_d/P_a_ifr
# eqnB => (FFR/NHi) = (IMR*BSR + BMR)/(BMR*HSR + IMR)
merged['NHi_val'] = pd.to_numeric(merged['P_d/P_a_ifr'], errors='coerce')
merged['Obs_eqnB'] = merged['FFR_val'] / merged['NHi_val']

merged['BSR'] = pd.to_numeric(merged['HSR_ifr'], errors='coerce')
merged['HSR'] = pd.to_numeric(merged['HSR_ffr'], errors='coerce')

def eqnB_pred(row):
    try:
        bmr = row['BMR']
        imr = row['IMR']
        bsr = row['BSR']
        hsr = row['HSR']
        num = imr*bsr + bmr
        den = bmr*hsr + imr
        if den != 0:
            return num/den
    except:
        pass
    return np.nan

merged['Pred_eqnB'] = merged.apply(eqnB_pred, axis=1)

############################################################
# 4) CREATE *SEPARATE* SUBSETS FOR eqn(7) & eqn(B)
############################################################
# eqn7_data => only require Obs_eqn7, Pred_eqn7 to be non-null
eqn7_data = merged.dropna(subset=['Obs_eqn7','Pred_eqn7'])

# eqnB_data => only require Obs_eqnB, Pred_eqnB to be non-null
eqnB_data = merged.dropna(subset=['Obs_eqnB','Pred_eqnB'])

print(f"Equation(7) has {eqn7_data.shape[0]} rows after dropping invalid for eqn7.")
print(f"Equation(B) has {eqnB_data.shape[0]} rows after dropping invalid for eqnB.")

############################################################
# 5) ERROR METRICS (MSE) FOR EACH
############################################################
def mse(a, b):
    return np.mean((a - b)**2)

mse_eqn7 = mse(eqn7_data['Obs_eqn7'], eqn7_data['Pred_eqn7']) if eqn7_data.shape[0]>0 else np.nan
mse_eqnB = mse(eqnB_data['Obs_eqnB'], eqnB_data['Pred_eqnB']) if eqnB_data.shape[0]>0 else np.nan

print(f"Eqn(7) MSE: {mse_eqn7:.5f}")
print(f"Eqn(B)  MSE: {mse_eqnB:.5f}")

############################################################
# 6) PLOT eqn(7) AND eqn(B) SEPARATELY
############################################################

# eqn(7)
if eqn7_data.shape[0] > 0:
    plt.figure(figsize=(5,4))
    plt.scatter(eqn7_data['Obs_eqn7'], eqn7_data['Pred_eqn7'],
                color='k', edgecolor='k', alpha=1)
    # identity line (adjust your range as needed)
    minval = min(eqn7_data['Obs_eqn7'].min(), eqn7_data['Pred_eqn7'].min())-0.1
    maxval = max(eqn7_data['Obs_eqn7'].max(), eqn7_data['Pred_eqn7'].max())+0.1
    plt.plot([minval, maxval], [minval, maxval], 'k--')
    plt.xlabel("Observed: CFR/FFR")
    plt.ylabel("Predicted: 1/FFR + BMR/IMR - 1")
    plt.title(f"Eqn(7) MSE={mse_eqn7:.3f}")
    plt.tight_layout()
    plt.show()
else:
    print("No valid data for eqn(7).")

# eqn(B)
if eqnB_data.shape[0] > 0:
    plt.figure(figsize=(5,4))
    plt.scatter(eqnB_data['Pred_eqnB'],eqnB_data['Obs_eqnB'], 
                color='tab:blue', edgecolor='k', alpha=1)
    minvalB = min(eqnB_data['Obs_eqnB'].min(), eqnB_data['Pred_eqnB'].min())-0.1
    maxvalB = max(eqnB_data['Obs_eqnB'].max(), eqnB_data['Pred_eqnB'].max())+0.1
    plt.plot([0, 2.5], [.5, 1.1], 'k--')
    plt.xlabel("Calculated FFR/NHi")
    plt.ylabel("Observed FFR/NHi")
    plt.title(f"Eqn. (B), from Molfetta et al., MSE={mse_eqnB:.3f}")
    plt.tight_layout()
    plt.show()
else:
    print("No valid data for eqn(B).")
    
    # Suppose we've already created eqn7_data with columns:
#  'Obs_eqn7' (observed), 'Pred_eqn7' (predicted), and e.g. 'Geometry Number_ffr'
#  as in your merged DataFrame from the script above.

threshold = 1  # or whatever you consider "large" squared error

if eqn7_data.shape[0] > 0:
    plt.figure(figsize=(5,4))
    plt.scatter(eqn7_data['Pred_eqn7'], eqn7_data['Obs_eqn7'],
                color='tab:blue', edgecolor='k', alpha=0.8)

    # Identity line
    minval = min(eqn7_data['Obs_eqn7'].min(), eqn7_data['Pred_eqn7'].min()) - 0.1
    maxval = max(eqn7_data['Obs_eqn7'].max(), eqn7_data['Pred_eqn7'].max()) + 0.1
    plt.plot([minval, maxval], [minval, maxval], 'k--')

    # Now label points that exceed 'threshold' in squared error
    for i, row in eqn7_data.iterrows():
        obs = row['Obs_eqn7']
        pred = row['Pred_eqn7']
        sq_err = abs(obs - pred)
        if sq_err > threshold:
            # Print info in console
            geom_ffr = row.get('Geometry Number_ffr','?')
            print(f"Large eqn(7) error: Geom={geom_ffr}, Obs={obs:.3f}, Pred={pred:.3f}, SE={sq_err:.3f}")
            # Place a text label near the point
            plt.text(pred + 0.08, obs, str(f'#{geom_ffr}'), color='black', fontsize=8)
    
    plt.xlabel("Calculated: CFR/FFR")
    plt.ylabel("Observed: CFR/FFR")
    plt.title("Eqn. (7) from Garcia (2019)")
    plt.tight_layout()
    plt.show()
else:
    print("No valid data for eqn(7).")

