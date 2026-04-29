#!/usr/bin/env python3

import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

SUMMARY_FILE = "summary2.csv"

NUMERIC_COLS = [
    "Average Flow", "Max Flow", "v_distal","P_d/P_a", "HSR",
    "P_Loss_Coeff", "HMR", "BMR/HMR","CFR","CFR/FFR",
    "WSS_Bif", "WSS_LE", "WSS_TE", "WSS_Area_Bifur",
    "WSS_TE_min", "WSS_LE_min",
    "WSS_LE_Area", "WSS_TE_Area",
    "WSS_Area_Bifur_min", "WSS_TE_Area_min", "WSS_LE_Area_min"
]

DISPLAY_LABELS = {
    "P_d/P_a": "FFR",
    "v_distal": "Distal Velocity"
}

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

    ############################################################################
    # 1) Normalize selected columns by WSS_LMB, set near-zero to 0.0, handle 0% case
    ############################################################################
    to_normalize = [
        "WSS_Bif", "WSS_LE", "WSS_TE",
        "WSS_Area_Bifur", "WSS_TE_min", "WSS_LE_min"
    ]

    group_dict = {}

    for row in data:
        # Normalize by WSS_LMB first
        wss_lmb = safe_float(row.get("WSS_LMB"))
        if wss_lmb and not is_zero(wss_lmb):
            for col in to_normalize:
                val = safe_float(row.get(col))
                if val is not None:
                    row[col] = str(val / wss_lmb)

        # Then do any thresholding or 'N/A' logic
        stenosis_val = safe_float(row.get("Stenosis Percentage"))

        for col in NUMERIC_COLS:
            val = safe_float(row.get(col))
            # Set near-zero values to 0.0
            if val is not None and val < 0.001:
                row[col] = "0.0"
            # If 0% stenosis and LE/TE in name, set to NaN
            if stenosis_val < 0.01 and ("LE" in col or "TE" in col):
                row[col] = np.nan

        # Group by (Stenosis, Length, R_total)
        key = (row["Stenosis Percentage"], row["Length"], row["R_total"])
        group_dict.setdefault(key, []).append(row)

    ############################################################################
    # 2) Build df_ratios but also store raw original vs. scaled
    ############################################################################
    ratio_rows = []
    for key, rows_for_key in group_dict.items():
        zero_micro_runs = []
        non_zero_micro_runs = []
        for r in rows_for_key:
            r_m = safe_float(r.get("R_micro", ""))
            # Decide if it's a zero-micro run or a non-zero-micro run
            if r_m is not None and not is_zero(r_m):
                non_zero_micro_runs.append(r)
            else:
                zero_micro_runs.append(r)

        # Pair each non-zero run with each zero run
        for nz in non_zero_micro_runs:
            nz_rmicro_val = safe_float(nz.get("R_micro"))
            for z in zero_micro_runs:
                ratio_dict = {
                    "Stenosis": key[0],
                    "Length": key[1],
                    "R_total": key[2],
                    "R_micro_nonzero": nz_rmicro_val,  # e.g. 0.38
                    "R_micro_zero": 0.0,
                    "Geometry (NZ)": nz.get("Geometry Number", ""),
                    "Geometry (Z)": z.get("Geometry Number", "")
                }

                # For each numeric col, store BOTH raw values + ratio
                for col in NUMERIC_COLS:
                    val_nz = safe_float(nz.get(col, None))
                    val_z = safe_float(z.get(col, None))

                    # (a) Store raw values
                    ratio_dict[col + "_orig"]   = val_z
                    ratio_dict[col + "_scaled"] = val_nz

                    # (b) Optionally store ratio
                    if (val_nz is None or is_zero(val_nz)) and (val_z is None or is_zero(val_z)):
                        ratio_dict[col] = "0/0"
                    elif val_z is not None and is_zero(val_z) and not is_zero(val_nz):
                        ratio_dict[col] = "∞"
                    elif val_nz is not None and val_z and not is_zero(val_z):
                        ratio_dict[col] = val_nz / val_z
                    else:
                        ratio_dict[col] = None

                ratio_rows.append(ratio_dict)

    df_ratios = pd.DataFrame(ratio_rows)

    # Reorder columns
    reorder_cols = (
        ["Stenosis", "Length", "R_total",
         "R_micro_nonzero", "R_micro_zero",
         "Geometry (NZ)", "Geometry (Z)"]
        + NUMERIC_COLS
        + [c + "_orig" for c in NUMERIC_COLS]
        + [c + "_scaled" for c in NUMERIC_COLS]
    )
    # Keep only columns that actually exist
    reorder_cols = [c for c in reorder_cols if c in df_ratios.columns]
    df_ratios = df_ratios[reorder_cols]

    # Convert Stenosis, R_micro_nonzero, etc. to numeric
    df_ratios["Stenosis"] = pd.to_numeric(df_ratios["Stenosis"], errors="coerce")
    df_ratios["R_micro_nonzero"] = pd.to_numeric(df_ratios["R_micro_nonzero"], errors="coerce")

    # Sort
    df_ratios = df_ratios.sort_values(
        by=["Stenosis", "R_micro_nonzero"], ascending=[False, False]
    ).reset_index(drop=True)

    print("\n==== TABLE OF RAW + RATIOS ====\n")
    print(df_ratios.head(20).to_string(index=False))

    ############################################################################
    # 3) Build the bar charts from raw columns (not from the ratio)
    ############################################################################

    # Convert R_total to numeric
    df_ratios["R_total"] = pd.to_numeric(df_ratios["R_total"], errors="coerce")

    # Drop rows where R_total might still be NaN
    df_ratios.dropna(subset=["R_total"], inplace=True)

    COLUMNS_OF_INTEREST = ["WSS_TE_min", "WSS_LE_min", "WSS_LE_Area", "WSS_TE_Area"]

    # Exclude 0% stenosis if you wish
    df_nonzero_sten = df_ratios[df_ratios["Stenosis"] > 0.1].copy()

    # We'll plot the bar chart from the columns _orig and _scaled
    # E.g. "WSS_TE_min_orig" vs "WSS_TE_min_scaled"
    fig, axes = plt.subplots(
        nrows=1, ncols=len(COLUMNS_OF_INTEREST),
        figsize=(16, 5), tight_layout=True
    )

    for ax, base_col in zip(axes, COLUMNS_OF_INTEREST):
        col_orig = base_col + "_orig"
        col_scaled = base_col + "_scaled"

        # Group by R_total and average if multiple rows
        group_data = df_nonzero_sten.groupby("R_total")[[col_orig, col_scaled]].mean().reset_index()

        x_vals = group_data["R_total"]
        y_orig = group_data[col_orig]
        y_scaled = group_data[col_scaled]

        width = 0.35
        ax.bar(x_vals - width/2, y_orig,   width=width, label="Original")
        ax.bar(x_vals + width/2, y_scaled, width=width, label="Scaled")

        ax.set_title(base_col, fontsize=12, fontweight="bold")
        ax.set_xlabel("R_total")
        ax.set_ylabel("Value")
        ax.legend()

    plt.suptitle("Original vs. r_micro-Scaled Values by R_total", fontsize=14, fontweight="bold")
    plt.show()

if __name__ == "__main__":
    main()
