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
                    ratio_dict[col] = (val_nz / val_z) if val_nz is not None and val_z and not is_zero(val_z) else None

                ratio_rows.append(ratio_dict)

    df_ratios = pd.DataFrame(ratio_rows)

    reorder_cols = [
        "Stenosis", "Length", "R_total",
        "R_micro_nonzero", "R_micro_zero",
        "Geometry (NZ)", "Geometry (Z)"
    ] + NUMERIC_COLS
    df_ratios = df_ratios[[c for c in reorder_cols if c in df_ratios.columns]]

    df_ratios["Stenosis"] = pd.to_numeric(df_ratios["Stenosis"], errors="coerce")
    df_ratios["R_micro_nonzero"] = pd.to_numeric(df_ratios["R_micro_nonzero"], errors="coerce")
    df_ratios = df_ratios.sort_values(by=["Stenosis", "R_micro_nonzero"], ascending=[False, False]).reset_index(drop=True)

    print("\n==== TABLE OF RATIOS (non-zero R_micro ÷ zero R_micro) ====\n")
    print(df_ratios.to_string(index=False))

    # ---- HEATMAP ----
    numeric_data = df_ratios[NUMERIC_COLS].fillna(0.0).copy()
    ratio_matrix = numeric_data.values

    ratio_matrix_masked = np.where(ratio_matrix == 0.0, np.nan, ratio_matrix)

    base_cmap = cm.get_cmap("bwr")
    cmap = base_cmap
    cmap.set_bad(color='black')
    # Track boundaries where Stenosis changes
    stenosis_values = df_ratios["Stenosis"].tolist()
    divider_rows = []
    for i in range(1, len(stenosis_values)):
        if stenosis_values[i] != stenosis_values[i - 1]:
            divider_rows.append(i)

    fig, ax = plt.subplots(figsize=(24, 10), dpi=300, constrained_layout=True)
    cax = ax.imshow(ratio_matrix_masked, aspect="auto", vmin=0.5, vmax=1.5, cmap=cmap)
    for div_y in divider_rows:
        ax.plot(
            [-0.6, len(NUMERIC_COLS) - 0.5],
            [div_y - 0.5, div_y - 0.5],
            color='black',
            linewidth=1.5,
            clip_on=False
        )

    fig.colorbar(cax)

    label_font_size = 11
    matrix_font_size = 12
    ax.set_xticks(range(len(NUMERIC_COLS)))
    ax.set_xticklabels(NUMERIC_COLS, rotation=60, fontsize=label_font_size)
    ax.set_yticks(range(len(df_ratios)))

    row_labels = [
        f"Sten: {float(df_ratios.iloc[i]['Stenosis']):0.1f}, "
        f"{float(df_ratios.iloc[i]['R_micro_nonzero']):0.2f}/"
        f"{float(df_ratios.iloc[i]['R_total']):0.2f}"
        for i in range(len(df_ratios))
    ]
    ax.set_yticklabels(row_labels, fontsize=label_font_size)

    ax.set_title("Heatmap of (non-zero R_micro ÷ zero R_micro) Ratios", fontsize = matrix_font_size)

    for i in range(ratio_matrix.shape[0]):
        for j in range(ratio_matrix.shape[1]):
            val = ratio_matrix[i, j]
            if not is_zero(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12, fontweight="bold")

    plt.show()

if __name__ == "__main__":
    main()
