#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example HDBSCAN clustering script for summary3.csv

Created on Thu Feb 13 02:27:39 2025

@author: yourname
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

# If not installed: pip install hdbscan
import hdbscan


# 1. LOAD DATA
# ----------------------------------------------------
df = pd.read_csv('./summary2.csv')

# 🔧 Filter only rows where Condition == "Hyperemic" and source == "mine"
df = df[(df["Condition"] == "Hyperemic") & (df["source"] == "mine")]


# 2. PREPARE THE NUMERIC DATA (EXCLUDING COLUMNS)
# ----------------------------------------------------
exclude_cols = ["Geometry_Number", "discord", "P_Loss_Coeff"]  # or others if needed
df_numeric = df.select_dtypes(include=[np.number]).copy()
df_numeric.drop(columns=[col for col in exclude_cols if col in df_numeric.columns],
                inplace=True, errors='ignore')


# 3. HANDLE MISSING VALUES
# ----------------------------------------------------
# Drop columns that are entirely NaN:
df_numeric.dropna(axis=1, how='all', inplace=True)

# Use SimpleImputer to fill remaining NaNs with mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df_numeric.values)
df_numeric = pd.DataFrame(X_imputed, columns=df_numeric.columns)


# 4. SCALE / NORMALIZE FEATURES
# ----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric.values)


# 5. CLUSTER USING HDBSCAN
# ----------------------------------------------------
# A common starting point:
# - min_cluster_size: smallest cluster size to allow
# - min_samples: controls how conservative the clustering is
#   (often the same as min_cluster_size, but you can tune)
# - metric: default 'euclidean', but can try others
# - cluster_selection_epsilon: might be useful in some contexts
# - cluster_selection_method="eom" or "leaf"
#
# You’ll likely tune these parameters to see which
# yields the most interpretable and stable results.

clusterer = hdbscan.HDBSCAN(min_cluster_size=5,
                            min_samples=1,
                            metric='euclidean',
                            cluster_selection_method='eom')
cluster_labels = clusterer.fit_predict(X_scaled)

# NOTE: HDBSCAN assigns `-1` to noise points (points not assigned to any cluster).


# 6. ASSESS CLUSTER QUALITY (SILHOUETTE SCORE)
# ----------------------------------------------------
# Silhouette score is normally used for "hard" cluster assignments, but HDBSCAN
# can label outliers/noise as -1.
# A common approach is to calculate silhouette only on the points labeled in a real cluster.
# i.e., exclude noise points with label == -1.

non_noise_mask = (cluster_labels != -1)
if non_noise_mask.sum() > 1:  # Enough points to score
    sil_score_hdbscan = silhouette_score(X_scaled[non_noise_mask],
                                         cluster_labels[non_noise_mask])
    print("Silhouette score (excluding noise points):", sil_score_hdbscan)
else:
    print("Not enough non-noise points to compute silhouette score.")


# 7. SUMMARIZE CLUSTERS
# ----------------------------------------------------
df_clustered = df_numeric.copy()
df_clustered["Cluster"] = cluster_labels  # same length as df_numeric

# Show how many points ended up in each cluster (including noise)
cluster_counts = df_clustered["Cluster"].value_counts(dropna=False).sort_index()
print("\nCluster counts (including noise = -1):")
print(cluster_counts)

# Calculate mean ± std for each cluster *except noise*
clusters_no_noise = df_clustered[df_clustered["Cluster"] != -1]
cluster_stats = clusters_no_noise.groupby("Cluster").agg(["mean", "std"])
print("\nCluster means and std dev (excluding noise):\n", cluster_stats)


# 8. (OPTIONAL) GET CLUSTER PROBABILITIES (SOFT CLUSTERING)
# ---------------------------------------------------------
# HDBSCAN can provide membership strengths (probabilities) for each cluster:
if hasattr(clusterer, 'probabilities_'):
    cluster_probabilities = clusterer.probabilities_
    df_clustered["Cluster_Probability"] = cluster_probabilities
    print("\nFirst few cluster probabilities (soft assignments):")
    print(df_clustered["Cluster_Probability"].head())


# 9. (OPTIONAL) PCA VISUALIZATION
# ----------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=20)
plt.title("PCA Projection with HDBSCAN Clusters (-1 = Noise)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# 10. (OPTIONAL) RELATE TO KNOWN LABELS, e.g. 'discord'
# ----------------------------------------------------
if "discord" in df.columns:
    # Merge cluster labels back to original data (if needed)
    # The length is the same, so you can simply assign:
    df["Cluster"] = cluster_labels
    cluster_discord_table = pd.crosstab(df["Cluster"], df["discord"])
    print("\nCluster vs. Discord Cross-Tabulation:\n", cluster_discord_table)
