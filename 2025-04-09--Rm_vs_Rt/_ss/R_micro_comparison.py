#!/usr/bin/env python3

import csv
import math
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_FILE = "summary2.csv"

# The numeric columns we want to compare via ratio (non-zero R_micro ÷ zero R_micro):
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
    """Convert string to float; return None if blank or invalid."""
    try:
        val = float(s)
        return val
    except (TypeError, ValueError):
        return None

def is_zero(x, tol=1e-12):
    """Check if x is effectively zero within tolerance."""
    if x is None:
        return False
    return abs(x) < tol

def main():
    # ----------------------
    # 1) Read summary2.csv
    # ----------------------
    with open(SUMMARY_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # ----------------------
    # 2) Group runs by (stenosis, length, R_total)
    #    so we can find zero vs non-zero R_micro pairs
    # ----------------------
    # We'll build a dict keyed by (stenosis, length, r_total)
    # where each value is a list of row dicts that share those 3 features.
    group_dict = {}
    for row in data:
        stenosis = row["Stenosis Percentage"]
        length   = row["Length"]
        r_total  = row["R_total"]
        key = (stenosis, length, r_total)

        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append(row)

    # ----------------------
    # 3) For each group, find runs with R_micro=0 and R_micro!=0
    #    Pair them up and compute ratio of their columns
    # ----------------------
    ratio_rows = []  # each element will become one row in our final table DataFrame
    for key, rows in group_dict.items():
        # Split into zero vs non-zero
        zero_micro_runs = []
        non_zero_micro_runs = []
        for r in rows:
            r_m = safe_float(r.get("R_micro", ""))
            if r_m is not None and not is_zero(r_m):
                non_zero_micro_runs.append(r)
            else:
                zero_micro_runs.append(r)

        # For each non-zero run, pair it with each zero run
        # Typically you'd expect 1:1, but let's do a double loop in case there's more.
        for nz in non_zero_micro_runs:
            nz_rmicro_val = safe_float(nz["R_micro"])
            for z in zero_micro_runs:
                # We'll build a new row representing the ratio for (nz ÷ z)
                ratio_dict = {}
                ratio_dict["Stenosis"] = key[0]
                ratio_dict["Length"]   = key[1]
                ratio_dict["R_total"]  = key[2]
                ratio_dict["R_micro_nonzero"] = nz_rmicro_val
                ratio_dict["R_micro_zero"]    = 0.0

                # Optionally also record e.g. geometry numbers
                ratio_dict["Geometry (NZ)"] = nz.get("Geometry Number", "")
                ratio_dict["Geometry (Z)"]  = z.get("Geometry Number", "")

                # Compute ratio for each numeric col
                for col in NUMERIC_COLS:
                    val_nz = safe_float(nz.get(col, None))
                    val_z  = safe_float(z.get(col, None))
                    # We'll store e.g. col+"_ratio"
                    col_ratio = None
                    if val_nz is not None and val_z is not None and not is_zero(val_z):
                        col_ratio = val_nz / val_z
                    ratio_dict[col] = col_ratio
                ratio_rows.append(ratio_dict)

    # ----------------------
    # 4) Convert ratio_rows to a DataFrame
    # ----------------------
    df_ratios = pd.DataFrame(ratio_rows)

    # Reorder columns for convenience (place numeric columns last or as you like)
    # We'll start with some ID columns, then numeric columns
    reorder_cols = [
        "Stenosis", "Length", "R_total",
        "R_micro_nonzero", "R_micro_zero",
        "Geometry (NZ)", "Geometry (Z)"
    ] + NUMERIC_COLS

    # Filter to only columns that exist
    final_cols = [c for c in reorder_cols if c in df_ratios.columns]
    df_ratios = df_ratios[final_cols]

    print("\n==== TABLE OF RATIOS (non-zero R_micro ÷ zero R_micro) ====\n")
    print(df_ratios.to_string(index=False))

    # ----------------------
    # 5) Make a heatmap
    #    We'll do a single heatmap of the ratio values for all pairs
    #    across the numeric columns. Rows = run pairs, Cols = numeric columns.
    # ----------------------
    # We'll ignore the ID columns and just use the numeric ratio columns
    numeric_data = df_ratios[NUMERIC_COLS]  # shape = (#pairs x #features)

    # Some of these ratio cells could be None. Let's fill them with 0 or NaN so imshow can handle them.
    numeric_data = numeric_data.fillna(0.0)

    # Convert to actual NumPy array
    ratio_matrix = numeric_data.values

    # Plot with matplotlib
    fig, ax = plt.subplots()
    cax = ax.imshow(ratio_matrix, aspect="auto")
    fig.colorbar(cax)

    # Label ticks
    ax.set_xticks(range(len(NUMERIC_COLS)))
    ax.set_xticklabels(NUMERIC_COLS, rotation=90)
    ax.set_yticks(range(len(df_ratios)))
    # Optionally build row labels that show some identification, e.g. geometry
    row_labels = [
        f"Sten: {float(df_ratios.iloc[i]['Stenosis']):0.1f}, {float(df_ratios.iloc[i]['R_micro_nonzero']):0.2f}/{float(df_ratios.iloc[i]['R_total']):0.2f}"
        for i in range(len(df_ratios))
    ]
    ax.set_yticklabels(row_labels)

    ax.set_title("Heatmap of (non-zero R_micro ÷ zero R_micro) Ratios")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
