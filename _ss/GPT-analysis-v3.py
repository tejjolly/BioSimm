import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


##############################################################################
# 1) LOAD AND PREPARE YOUR DATA
##############################################################################
df = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/summary.csv')

# Example columns
global_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'P_d/P_a',
    'HMR',
    'HSR',
    'WSS',
    # 'Rtotal_cor Value'
    # Excluding 'CFR' and 'BMR/HMR' for the global PCA
]

focused_cols = [
    'Stenosis Percentage',
    'Length',
    'Width',
    'Average Flow',
    'P_d/P_a',
    'HMR',
    'HSR',
    'WSS',
    # 'Rtotal_cor Value',
    'CFR',
    'BMR/HMR'
]

# GLOBAL DATASET: keep all (Hyperemic + Non-hyperemic) rows that have no NaNs
df_global = df.dropna(subset=global_cols + ['Condition'])

# FOCUSED DATASET: only Hyperemic, including ratio cols
df_hyp = df[df['Condition'] == 'Hyperemic'].dropna(subset=focused_cols)

##############################################################################
# 2) HELPER FUNCTION: PCA + PLOTTING 2D (PC combos)
##############################################################################
def run_pca_get_coords(dataframe, feature_cols, n_components=3):
    """
    Fits PCA on 'feature_cols' in 'dataframe' (after standardizing),
    returns (pc_coords, explained_var, pca_model).
    """
    X = dataframe[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n_components)
    pc_coords = pca.fit_transform(X_scaled)  # shape: (n_samples, n_components)
    explained = pca.explained_variance_ratio_
    
    return pc_coords, explained, pca

def plot_pca_2d_combos(pc_coords, explained_var, color, 
                       pc_labels=None, discrete=False, 
                       discrete_label_map=None, colorbar_label=None,
                       fig_title=""):
    """
    Creates 2D PCA scatter plots for the three principal component pairs.
    Ensures uniform subplot sizes, equal aspect ratios, and fixed overall figure size.
    """
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    # import matplotlib.ticker as ticker  # Import ticker for formatting
    
    # Default labels for PCs
    if pc_labels is None:
        pc_labels = [
            f'PC1 ({explained_var[0]*100:.1f}% var)',
            f'PC2 ({explained_var[1]*100:.1f}% var)',
            f'PC3 ({explained_var[2]*100:.1f}% var)'
        ]
    
    combos = [(0,1), (0,2), (1,2)]  # (PC1 vs PC2), (PC1 vs PC3), (PC2 vs PC3)

    # Set a fixed figure size to maintain layout consistency
    fig_width = 20  # Adjusted to fit both colorbars and legends
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 5))
    fig.suptitle(fig_title, fontsize=16)

    # Set equal axis limits for consistent subplot sizes
    min_lim = np.min(pc_coords) - 0.5
    max_lim = np.max(pc_coords) + 0.5

    # If continuous, create a colormap normalization
    cmap = cm.viridis
    norm = None
    if not discrete:
        cvals = np.array(color, dtype=float)
        norm = plt.Normalize(vmin=cvals.min(), vmax=cvals.max())

    # Create scatter plots
    for ax, (i, j) in zip(axes, combos):
        x = pc_coords[:, i]
        y = pc_coords[:, j]

        if discrete:
            sc = ax.scatter(x, y, c=color, alpha=0.7, edgecolor='k')
        else:
            sc = ax.scatter(x, y, c=color, cmap=cmap, norm=norm, alpha=0.7, edgecolor='k')

        ax.set_xlabel(pc_labels[i])
        ax.set_ylabel(pc_labels[j])
        ax.set_title(f"{pc_labels[i]} vs. {pc_labels[j]}")
        ax.grid(True)

        # Apply uniform axis limits and aspect ratio
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.set_aspect('equal', adjustable='datalim')

    if discrete:
        # Add a fixed-width legend to maintain figure size consistency
        unique_vals = np.unique(color)
        legend_patches = [mpatches.Patch(color=val, label=discrete_label_map[val]) 
                          for val in unique_vals if val in discrete_label_map]
        
        # Create legend on the right, outside the last subplot
        legend_ax = fig.add_axes([0.90, 0.3, 0.05, 0.4])  # Colorbar placement
        legend_ax.axis("off")  # Hide the axis
        legend_ax.legend(handles=legend_patches, title="Condition", loc="center")
    else:
        # Create a fixed-position colorbar to ensure layout consistency
        # fig.subplots_adjust(right=10)  # Make space for the colorbar
        cbar_ax = fig.add_axes([0.9171, 0.2, 0.015, 0.6])  # Colorbar placement
        cbar = fig.colorbar(sc, cax=cbar_ax)

        # **Set colorbar format to always show two decimal places**
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if colorbar_label:
            cbar.set_label(colorbar_label)

    # Adjust subplot spacing to make figures visually identical
    fig.subplots_adjust(left=0.08, right=0.88, top=0.85, bottom=0.15, wspace=0.2)
    plt.show()


##############################################################################
# 3) GLOBAL PCA (Discrete color by Condition)
##############################################################################
pc_coords_global, expl_global, pca_global = run_pca_get_coords(df_global, global_cols, n_components=3)
print("\n[GLOBAL PCA] Explained variance ratio:", expl_global)
print("Cumulative:", np.cumsum(expl_global))

# Create a discrete color mapping for Condition
# This time we'll make the color array strings like 'red' or 'blue',
# and also prepare a discrete_label_map that translates 'red' -> 'Hyperemic', etc.
color_map = {'Hyperemic': 'red', 'Non-hyperemic': 'blue'}

# Convert condition to color strings
global_color_array = df_global['Condition'].map(color_map).values


# Prepare the dictionary to label them in the legend
discrete_label_map = {
    'red': 'Hyperemic',
    'blue': 'Non-hyperemic'
}

plot_pca_2d_combos(
    pc_coords=pc_coords_global,
    explained_var=expl_global,
    color=global_color_array,
    pc_labels=None,  # let the function auto-label them
    discrete=True,   # we have discrete categories
    discrete_label_map=discrete_label_map,
    colorbar_label=None,
    fig_title="PCA, colored by (non-)hyperemic"
)
##############################################################################
# 4) GLOBAL PCA (Continuous color by Stenosis Percentage)
##############################################################################
# Convert condition to color strings
global_color_array = df_global['Stenosis Percentage'].values

plot_pca_2d_combos(
    pc_coords=pc_coords_global,
    explained_var=expl_global,
    color=global_color_array,
    pc_labels=None,  # let the function auto-label them
    discrete=False,   # we have discrete categories
    discrete_label_map=None,
    colorbar_label="Stenosis Percentage",
    fig_title="PCA, colored by stenosis"
)

##############################################################################
# 5) FOCUSED PCA (Hyperemic-only, continuous color by Stenosis Percentage)
##############################################################################
pc_coords_hyp, expl_hyp, pca_hyp = run_pca_get_coords(df_hyp, focused_cols, n_components=3)
print("\n[HYPEREMIC-ONLY PCA] Explained variance ratio:", expl_hyp)
print("Cumulative:", np.cumsum(expl_hyp))

# We'll color by "Stenosis Percentage" (continuous)
color_array_hyp = df_hyp['Stenosis Percentage'].values  # numeric

plot_pca_2d_combos(
    pc_coords=pc_coords_hyp,
    explained_var=expl_hyp,
    color=color_array_hyp,
    pc_labels=None,   # auto-label
    discrete=False,   # numeric
    discrete_label_map=None,
    colorbar_label="Stenosis Percentage",
    fig_title="PCA, hyperemic only, colored by stenosis"
)

# ##############################################################################
# # 4.2) GLOBAL PCA (Continuous color by HMR)
# ##############################################################################
# pc_coords_global, expl_global, pca_global = run_pca_get_coords(df_global, global_cols, n_components=3)
# print("\n[GLOBAL PCA] Explained variance ratio:", expl_global)
# print("Cumulative:", np.cumsum(expl_global))

# # Convert condition to color strings
# global_color_array = df_global['HMR'].values

# plot_pca_2d_combos(
#     pc_coords=pc_coords_global,
#     explained_var=expl_global,
#     color=global_color_array,
#     pc_labels=None,  # let the function auto-label them
#     discrete=False,   # we have discrete categories
#     discrete_label_map=None,
#     colorbar_label="HMR",
#     fig_title="PCA, colored by HMR"
# )

# ##############################################################################
# # 5.2) FOCUSED PCA (Hyperemic-only, continuous color by Stenosis Percentage)
# ##############################################################################
# pc_coords_hyp, expl_hyp, pca_hyp = run_pca_get_coords(df_hyp, focused_cols, n_components=3)
# print("\n[HYPEREMIC-ONLY PCA] Explained variance ratio:", expl_hyp)
# print("Cumulative:", np.cumsum(expl_hyp))

# # We'll color by "Stenosis Percentage" (continuous)
# color_array_hyp = df_hyp['HMR'].values  # numeric

# plot_pca_2d_combos(
#     pc_coords=pc_coords_hyp,
#     explained_var=expl_hyp,
#     color=color_array_hyp,
#     pc_labels=None,   # auto-label
#     discrete=False,   # numeric
#     discrete_label_map=None,
#     colorbar_label="HMR",
#     fig_title="PCA, hyperemic only, colored by HMR"
# )

##############################################################################
# 6) INSPECT LOADINGS IF DESIRED
##############################################################################
# Global loadings
loading_df_global = pd.DataFrame(
    pca_global.components_.T,
    index=global_cols,
    columns=[f"PC{i+1}" for i in range(3)]
)
print("\n[GLOBAL PCA Loadings]")
print(loading_df_global)

# Focused (Hyperemic) loadings
loading_df_hyp = pd.DataFrame(
    pca_hyp.components_.T,
    index=focused_cols,
    columns=[f"PC{i+1}" for i in range(3)]
)
print("\n[HYPEREMIC-ONLY PCA Loadings]")
print(loading_df_hyp)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Suppose you have pc_coords_global from your existing PCA:
X_pca = pc_coords_global  # shape (n_samples, 3)

# 1) Perform Hierarchical Clustering (Ward's method)
Z = linkage(X_pca, method='average', metric='euclidean')

# 2) Plot the dendrogram
plt.figure(figsize=(10, 6))
dn = dendrogram(
    Z,
    above_threshold_color='grey',  # color branches that merge above certain distance
    orientation='top',             # dendrogram up top, leaves at bottom
    distance_sort='descending',    # optional: tries to arrange merges by distance
    show_leaf_counts=True          # show how many points in each leaf cluster
)
plt.title("Hierarchical Clustering Dendrogram ('Average' linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Distance (Euclidian)")
plt.grid(True, which='major', axis='y')  # Enable only major y-grid lines
plt.grid(False, which='both', axis='x')  # Disable all x-grid lines
plt.grid(False, which='minor', axis='y')  # Disable minor y-grid lines
plt.show()

# 3) Decide on a number of clusters or a distance cutoff
#    For example, we want 3 clusters:
num_clusters = 3
labels_hier = fcluster(Z, t=num_clusters, criterion='maxclust')
labels_hier_zero_based = labels_hier - 1  # optional shift if you prefer 0-based

all_cluster_colors = ['red', 'blue', 'green', 'pink', 'orange', 'black', 'gray']
color_array_hier = [all_cluster_colors[label] for label in labels_hier_zero_based]
label_map_hier = {
    all_cluster_colors[i]: f"Cluster {i+1}" for i in range(num_clusters)
}



plot_pca_2d_combos(
    pc_coords=X_pca,
    explained_var=expl_global,
    color=color_array_hier,     # color array of strings like 'red','blue','green'
    pc_labels=None,
    discrete=True,  
    discrete_label_map=label_map_hier,  # color->label mapping
    colorbar_label="HierClusterID",
    fig_title=f"Hierarchical Clustering (k={num_clusters}) on PCA"
)

