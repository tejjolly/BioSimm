#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 02:27:39 2025

@author: tejjolly
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# 1. LOAD DATA
# ----------------------------------------------------
df = pd.read_csv('./summary3.csv')

# 🔧 Filter only rows where Condition == "Hyperemic" and source == "mine"
df = df[(df["Condition"] == "Hyperemic") & (df["source"] == "mine")]

# 🔧 Exclude ID and label columns from clustering
exclude_cols = ["Geometry_Number", "discord"]
df_numeric = df.select_dtypes(include=[np.number]).copy()
df_numeric.drop(columns=[col for col in exclude_cols if col in df_numeric.columns], inplace=True)


# 2. HANDLE MISSING VALUES / OUTLIERS
# ----------------------------------------------------
# Simple approach: drop rows with missing values (or you can fill them)
# df_numeric.dropna(inplace=True)

# With this:
df_numeric.dropna(axis=1, how='all', inplace=True)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df_numeric.values)
df_numeric = pd.DataFrame(X_imputed, columns=df_numeric.columns)

# Then scale

# 3. SCALE / NORMALIZE FEATURES
# ----------------------------------------------------
scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df_numeric.values)  # Numpy array
X_scaled = scaler.fit_transform(X_imputed)


# 4. DETERMINE BEST k (ELBOW + SILHOUETTE)
# ----------------------------------------------------
K_values = range(2, 11)
inertia_list = []
silhouette_list = []

for k in K_values:
    kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans_model.fit(X_scaled)
    inertia_list.append(kmeans_model.inertia_)

    labels = kmeans_model.labels_
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_list.append(sil_score)

# 5. PLOT ELBOW (WCSS/INERTIA) AND SILHOUETTE SCORES
# ----------------------------------------------------
# Plot 1: Elbow
plt.figure()
plt.plot(K_values, inertia_list, marker='o')
plt.title("Elbow Plot: Within-Cluster Sum of Squares vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.show()

# Plot 2: Silhouette
plt.figure()
plt.plot(K_values, silhouette_list, marker='o')
plt.title("Silhouette Scores vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# Choose the best k (you might pick based on elbow or peak silhouette)
best_k = 6  # <-- For example, or decide programmatically

# 6. FINAL KMEANS WITH best_k
# ----------------------------------------------------
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# 7. SUMMARIZE CLUSTERS
# ----------------------------------------------------
df_clustered = df_numeric.copy()
df_clustered["Cluster"] = cluster_labels

# Count how many samples in each cluster
cluster_counts = df_clustered["Cluster"].value_counts().sort_index()

# Calculate mean ± std for each cluster
cluster_stats = df_clustered.groupby("Cluster").agg(["mean", "std"])

# Display cluster counts
print("Cluster counts:")
print(cluster_counts)
print("\nCluster means and std dev:\n", cluster_stats)

# 8. CENTROIDS
# ----------------------------------------------------
centroids = kmeans_final.cluster_centers_
# (These are in scaled space. You could inverse-transform if you want the original scale.)
original_space_centroids = scaler.inverse_transform(centroids)
centroids_df = pd.DataFrame(original_space_centroids, columns=df_numeric.columns)
print("\nCluster Centroids (unscaled):\n", centroids_df)

# 9. OPTIONAL: PCA FOR VISUAL CLUSTER MAP
# ----------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels)
plt.title("PCA Projection with K-Means Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 10. OPTIONAL: RELATE TO KNOWN LABELS
# ----------------------------------------------------
# If you have a label like "discord" or "Condition" in the original df,
# you can see how it intersects with your clusters. For example:
if "discord" in df.columns:
    # Merge cluster labels back to original data (if needed)
    df["Cluster"] = cluster_labels
    # Now check distribution:
    cluster_discord_table = pd.crosstab(df["Cluster"], df["discord"])
    print("\nCluster vs. Discord Cross-Tabulation:\n", cluster_discord_table)
