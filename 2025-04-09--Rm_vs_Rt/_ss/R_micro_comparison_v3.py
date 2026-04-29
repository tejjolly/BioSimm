#!/usr/bin/env python3

import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

SUMMARY_FILE = "summary2.csv"

NUMERIC_COLS = [
    "Average Flow", "Max Flow", "P_d/P_a", "HMR", "HSR",
    "CFR", "BMR/HMR", "CFR/FFR", "P_Loss_Coeff",
    "WSS_TE", "WSS_LE", "WSS_TE_Area", "WSS_LE_Area",
    "WSS_Area_Bifur", "WSS_Bif", "WSS_LMB",
    "WSS_min", "WSS_TE_min", "WSS_LE_min",
    "WSS_TE_Area_min", "WSS_LE_Area_min", "WSS_Area_Bifur_min",
    "v_distal"
]

def safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def is_zero(x, tol=1e-12):
    return x is not None and abs(x) < tol

def main():
    with open(SUMMARY_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    group_dict = {}
    for row in data:
        key = (row["Stenosis Percentage"], row["Length"], row["R_total"])
        group_dict.setdefault(key, []).append(row)

    ratio_rows = []
    for key, rows in group_dict.items():
        zero_micro_runs = []
        non_zero_micro_runs = []
        for r in rows:
            r_m = safe_float(r.get("R_micro", ""))
            if r_m is not None and not is_zero(r_m):
                non_zero_micro_runs.append(r)
            else:
                zero_micro_runs.append(r)

        for nz in non_zero_micro_runs:
            nz_rmicro_val = safe_float(nz["R_micro"])
            for z in zero_micro_runs:
                ratio_dict = {
                    "Stenosis": key[0],
                    "Length": key[1],
                    "R_total": key[2],
                    "R_micro_nonzero": nz_rmicro_val,
                    "R_micro_zero": 0.0,
                    "Geometry (NZ)": nz.get("Geometry Number", ""),
                    "Geometry (Z)": z.get("Geometry Number", "")
                }

                for col in NUMERIC_COLS:
                    val_nz = safe_float(nz.get(col, None))
                    val_z = safe_float(z.get(col, None))
                    ratio_dict[col] = (
                        (val_nz / val_z)
                        if val_nz is not None
                           and val_z
                           and not is_zero(val_z)
                        else None
                    )


                ratio_rows.append(ratio_dict)

    df_ratios = pd.DataFrame(ratio_rows)

    reorder_cols = [
        "Stenosis", "Length", "R_total",
        "R_micro_nonzero", "R_micro_zero",
        "Geometry (NZ)", "Geometry (Z)"
    ] + NUMERIC_COLS
    # Keep only columns that exist in df_ratios:
    df_ratios = df_ratios[[c for c in reorder_cols if c in df_ratios.columns]]

    df_ratios["Stenosis"] = pd.to_numeric(df_ratios["Stenosis"], errors="coerce")
    df_ratios["R_micro_nonzero"] = pd.to_numeric(df_ratios["R_micro_nonzero"], errors="coerce")

    # Sort descending by Stenosis, then by R_micro_nonzero
    df_ratios = df_ratios.sort_values(by=["Stenosis", "R_micro_nonzero"],
                                      ascending=[False, False]).reset_index(drop=True)

    # Print table in the console
    print("\n==== TABLE OF RATIOS (non-zero R_micro ÷ zero R_micro) ====\n")
    print(df_ratios.to_string(index=False))

    # ---- HEATMAP ----
    numeric_data = df_ratios[NUMERIC_COLS].fillna(0.0).copy()
    ratio_matrix = numeric_data.values

    # Replace zeros with NaN so those cells appear black (or masked) in the heatmap
    ratio_matrix_masked = np.where(ratio_matrix == 0.0, np.nan, ratio_matrix)

    base_cmap = plt.get_cmap("bwr")
    cmap = base_cmap
    cmap.set_bad(color='black')

    # Track boundaries where Stenosis changes (for dividing lines)
    stenosis_values = df_ratios["Stenosis"].tolist()
    divider_rows = []
    for i in range(1, len(stenosis_values)):
        if stenosis_values[i] != stenosis_values[i - 1]:
            divider_rows.append(i)

    fig, ax = plt.subplots(figsize=(24, 10), dpi=100, constrained_layout=True)

    # Create the heatmap
    cax = ax.imshow(ratio_matrix_masked,
                    aspect="auto", vmin=0.5, vmax=1.5, cmap=cmap)

    # 1) Identify row blocks for each Stenosis
    df_ratios["Stenosis"] = df_ratios["Stenosis"].astype(float)
    unique_sten = df_ratios["Stenosis"].unique()

    # We'll collect:
    # - The center of each Stenosis block (for major tick placement)
    # - The row-by-row positions (for minor tick labels)
    sten_centers = []
    sten_labels = []
    # y_minor_ticks = np.arange(len(df_ratios))  # one tick per row
    y_minor_ticks = np.arange(len(df_ratios)) + 0.01  # small offset to avoid exact overlap

    y_minor_labels = []

    start_idx = 0
    for s in unique_sten:
        block = df_ratios[df_ratios["Stenosis"] == s]
        size = len(block)
        center = start_idx + size / 2 - 0.5
        sten_centers.append(center)
        sten_labels.append(f"Sten: {s*100:.0f}%: ")

        # For each row in this block, label the R_micro (or something else)
        for i, row in block.iterrows():
            r_m = row["R_micro_nonzero"]
            # For example, label with R_micro_nonzero
            y_minor_labels.append(f"R_m={r_m:.2f}")
        start_idx += size

    # 2) Set the major y-ticks (one per Stenosis block)
    ax.set_yticks(sten_centers)
    ax.set_yticklabels(sten_labels, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', which='major', pad=75)  # increase pad to move labels left
    # 3) Add a secondary axis for the minor row labels
    # ax2 = ax.twinx()
    # ax2.set_ylim(ax.get_ylim())  # Match vertical range
    # ax2.set_yticks(y_minor_ticks)
    # ax2.set_yticklabels(y_minor_labels, fontsize=10)

    # Set both major and minor y-ticks on the same axis
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.set_yticklabels(y_minor_labels, fontsize=12, fontweight='bold', minor=True)

    # 4) Draw dividing lines between Stenosis blocks
    divider_rows = []
    for i in range(1, len(df_ratios)):
        if df_ratios["Stenosis"].iat[i] != df_ratios["Stenosis"].iat[i - 1]:
            divider_rows.append(i)
    for div_y in divider_rows:
        ax.axhline(div_y - 0.5, color='black', linewidth=1.2)

    # 5) Add numeric values to each cell (if not NaN)
    for i in range(ratio_matrix_masked.shape[0]):
        for j in range(ratio_matrix_masked.shape[1]):
            val = ratio_matrix_masked[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold")

    # 6) Finish with colorbar, x-tick labels, and show
    cbar = fig.colorbar(cax)
    cbar.set_label("Ratio (NZ / Z)", fontsize=14, labelpad=10)  # label + spacing
    cbar.ax.tick_params(labelsize=12)  # increase tick font size

    ax.set_xticks(range(len(NUMERIC_COLS)))
    ax.set_xticklabels(NUMERIC_COLS, rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    ax.set_title("Heatmap of (non-zero R_micro ÷ zero R_micro) Ratios")

    plt.show()

if __name__ == "__main__":
    main()
