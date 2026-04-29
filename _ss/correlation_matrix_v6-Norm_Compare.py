#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:52:41 2025

@author: tejjolly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:43:19 2025

@author: tejjolly
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from scipy.stats import pearsonr

### Subset for WSS + Target Variables ###
df_full = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary.csv')

n_bins = 5
cmap = plt.get_cmap('Reds', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

columns_of_interest = [
    # WSS variables in the order you specified:
    'WSS_LMB',
    'WSS', 
    'WSS_TE', 
    'WSS_LE', 
    'WSS_Bif', 
    'WSS_Avg_Area',
    'WSS_TE_Area', 
    'WSS_LE_Area', 
    'WSS_Area_Bifur',
    # target variables in the order you specified:
    'HMR', 
    'BMR/HMR', 
    'iFR', 
    'FFR', 
    'CFR', 
    'CFR/FFR'
]

# Filter out columns that might not exist in df_full
existing_cols = [col for col in columns_of_interest if col in df_full.columns]

df_compare = df_full.copy()

# To avoid division by zero, replace 0 with NaN in WSS_LMB
df_compare["WSS_LMB"] = df_compare["WSS_LMB"].replace(0, np.nan)

# If you prefer to remove rows that still have NaN in WSS_LMB:
df_compare = df_compare.dropna(subset=["WSS_LMB"])

# Create normalized columns for comparison
df_compare["N_WSS"]     = df_compare["WSS"]     / df_compare["WSS_LMB"]
df_compare["N_WSS_TE"]  = df_compare["WSS_TE"]  / df_compare["WSS_LMB"]
df_compare["N_WSS_LE"]  = df_compare["WSS_LE"]  / df_compare["WSS_LMB"]
df_compare["N_WSS_Bif"] = df_compare["WSS_Bif"] / df_compare["WSS_LMB"]

# -------------------------------------
# 3) Build a pairwise plot comparing both
# -------------------------------------
# We'll select the original (non-normalized) columns plus the new (normalized) columns.
cols_for_plot = [
    "WSS", "WSS_TE", "WSS_LE", "WSS_Bif", 
    "N_WSS", "N_WSS_TE", "N_WSS_LE", "N_WSS_Bif"
]

# Pairplot will produce a matrix of scatterplots/histograms 
# so you can visually compare how the distributions and pairwise relationships 
# differ between non-normalized and normalized columns.
sns.pairplot(df_compare[cols_for_plot])
plt.suptitle("Pairwise Comparison: Non-normalized vs. Normalized (N_)", y=1.02)
plt.show()

