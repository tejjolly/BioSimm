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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
# corr_matrix = df.corr()
corr_matrix = df.corr().abs()
mask = np.eye(corr_matrix.shape[0], dtype=bool)
mask |= corr_matrix.isnull()

# Heat map
n_bins = 10
# cmap = plt.get_cmap('coolwarm', n_bins)
cmap = plt.get_cmap('Reds', n_bins)
levels = np.linspace(0, 1, n_bins + 1)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
plt.figure(figsize=(12, 10))
ax = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, norm=norm,cbar_kws={"shrink": 0.8})
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

for (i, j), value in np.ndenumerate(mask):
    if value:  # Only color the masked cells
        # ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white'))
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
        

plt.show()

# pairplot = sns.pairplot(df)
# pairplot.savefig("pairplot_output.png")
# plt.show()


# target = 'CFR/FFR'
# features = df.drop(columns=[target])
# X = features
# y = df[target]
# print('y:',y)


# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Apply PCA
# pca = PCA(n_components=0.95)  # Retain 95% variance
# X_pca = pca.fit_transform(X_scaled)

# print(f'Number of components selected: {pca.n_components_}')

