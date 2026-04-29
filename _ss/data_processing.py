#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:18:28 2024

@author: tejjolly
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load the data
df_full = pd.read_csv('summary.csv')
df_full['Condition'] = df_full['Condition'].astype('category').cat.codes

df = df_full.drop(columns=['Condition','Geometry Number'])
# Rename columns permanently in the DataFrame
df.rename(columns={
    'P_d/P_a': 'FFR',
    'Stenosis Percentage': 'Stenosis',
    'Average Flow': 'Flow',
    'Rtotal_cor Value': 'R. mult.',
}, inplace=True)


# Display the first few rows
print(df.head())
# Get a summary of the DataFrame
print(df.info())
# Check for missing values
print(df.isnull().sum())



# Compute the correlation matrix
corr_matrix = df.corr().abs()

# Heat map
n_bins = 4
cmap = plt.get_cmap('coolwarm', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
plt.figure(figsize=(12, 10))
ax = sns.heatmap(corr_matrix, annot=True, cmap=cmap, norm=norm)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.show()

pairplot = sns.pairplot(df)
pairplot.savefig("pairplot_output.png")
plt.show()
