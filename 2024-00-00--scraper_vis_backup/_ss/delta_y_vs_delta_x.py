#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# =============================
# CONFIG / FLAGS
# =============================
manuscript_data = True

# choose which vessels to include
all_flag = False
LAD_flag = True
LCX_flag = True

# flag if Δx/x_prev <= this
FRAC_STEP_THRESH = 0.30

# Optional: ignore tiny numerical jitter in x steps
MIN_ABS_DX = 0.0   # set e.g. 1e-6 if you want

# Toggle detailed per-step printing
PRINT_INDIVIDUAL_STEPS = True

# =============================
# 1) READ DATA
# =============================
data_file = "../../data/data_manuscript.csv" if manuscript_data else "../data/data.csv"
df = pd.read_csv(data_file)

# -----------------------------
# 2) CLEAN / NUMERIC COLS
# -----------------------------
cols_to_num = [
    "CFR", "P_d/P_a", "BMR/HMR", "R_total",
    "Stenosis Percentage", "Length", "HMR", "HSR", "P_Loss_Coeff",
    "Q_distal", "v_distal"
]
for col in cols_to_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# OPTIONAL: define BMR if missing
if "BMR" not in df.columns and ("BMR/HMR" in df.columns) and ("HMR" in df.columns):
    df["BMR"] = df["BMR/HMR"] * df["HMR"]

# create iFR / FFR convenience columns (not strictly needed for this task)
if "Condition" in df.columns and "P_d/P_a" in df.columns:
    df["iFR"] = np.where(df["Condition"] == "Non-hyperemic", df["P_d/P_a"], np.nan)
    df["FFR"] = np.where(df["Condition"] == "Hyperemic", df["P_d/P_a"], np.nan)

# =============================
# 3) LOCATION FILTER (same logic as your plotting script)
# =============================
location_filter = (
    ["LAD", "LCX"] if all_flag or (LAD_flag and LCX_flag)
    else "LAD" if LAD_flag
    else "LCX" if LCX_flag
    else None
)

# =============================
# 4) GROUPING HELPER + CREATE "Stenosis Group"
# =============================
def stenosis_group(val, decimal_places=2, tolerance=0.02):
    """
    Round `val` to `decimal_places` and group it if within ±tolerance of that rounded value.
    Returns NaN if out of range, so it won't be grouped.
    """
    if pd.isna(val):
        return np.nan
    rounded_val = round(val, decimal_places)
    return rounded_val if abs(val - rounded_val) <= tolerance else np.nan

if "Stenosis Percentage" not in df.columns:
    raise KeyError("Your CSV does not contain 'Stenosis Percentage', cannot form 'Stenosis Group'.")

df["Stenosis Group"] = df["Stenosis Percentage"].apply(stenosis_group)

# =============================
# 5) CORE FUNCTION: INDIVIDUAL PRINTING + SUMMARY AVERAGES
# =============================
def report_small_x_steps_in_series(
    data,
    *,
    x_col,
    y_col,
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=None,
    show_singletons=False,
    frac_step_thresh=0.25,
    min_abs_dx=0.0,
    print_individual_steps=True,
):
    # pretty print names (your request)
    def pretty(col):
        if col == "Q_distal":
            return "Q"
        if col == "P_d/P_a":
            return "FRR"
        return col

    XNAME = pretty(x_col)
    YNAME = pretty(y_col)

    df_plot = data.copy()

    # ---- optional location filter ----
    if location_filter is not None:
        if location_col not in df_plot.columns:
            raise KeyError(f"location_col='{location_col}' not in dataframe columns.")
        if isinstance(location_filter, (list, tuple, set)):
            df_plot = df_plot[df_plot[location_col].isin(location_filter)]
        else:
            df_plot = df_plot[df_plot[location_col] == location_filter]

    # ---- required columns check ----
    needed = [x_col, y_col] + list(series_cols)
    missing = [c for c in needed if c not in df_plot.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ---- drop NA in x/y and in grouping keys ----
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna()]
    for c in series_cols:
        df_plot = df_plot[df_plot[c].notna()]

    if len(df_plot) == 0:
        print("No rows to scan after filters / NA drops.")
        return

    # ---- optionally remove singleton groups ----
    if not show_singletons:
        counts = df_plot.groupby(list(series_cols)).size()
        valid_groups = counts[counts > 1].index
        if len(valid_groups) == 0:
            print("All groups are singletons after filtering; nothing to scan.")
            return
        df_plot = (
            df_plot.set_index(list(series_cols))
                  .loc[valid_groups]
                  .reset_index()
        )

    # ---- print which series exist ----
    series_to_scan = (
        df_plot[list(series_cols)]
        .drop_duplicates()
        .sort_values(list(series_cols))
    )
    print("\nUNIQUE SERIES THAT WILL BE SCANNED "
          f"(total = {len(series_to_scan)})")
    print(series_to_scan.to_string(index=False))

    groups = df_plot.groupby(list(series_cols), dropna=False)

    total_steps = 0
    flagged_steps = 0

    all_flagged_dy = []
    all_flagged_frac = []
    all_flagged_dx = []
    all_flagged_dy_rel = []

    print(f"\nScanning: flag if (Δ{XNAME} / {XNAME}_prev) <= {frac_step_thresh:.2f} "
          f"and |Δ{XNAME}| >= {min_abs_dx}\n")

    for series_key, gdf in groups:
        if len(gdf) < 2:
            continue

        gdf_sorted = gdf.sort_values(by=x_col).reset_index(drop=True)

        x = gdf_sorted[x_col].to_numpy(dtype=float)
        y = gdf_sorted[y_col].to_numpy(dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        x_prev = x[:-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = dx / x_prev

        valid = np.isfinite(frac) & (x_prev != 0.0) & (np.abs(dx) >= float(min_abs_dx))
        hits = valid & (frac <= float(frac_step_thresh))
        y_prev = y[:-1]

        total_steps += (len(gdf_sorted) - 1)

        if not np.any(hits):
            continue

        flagged_steps += int(np.sum(hits))
        all_flagged_dy.extend(dy[hits].tolist())
        all_flagged_frac.extend(frac[hits].tolist())
        all_flagged_dx.extend(dx[hits].tolist())
        valid_rel = hits & (y_prev != 0.0)
        all_flagged_dy_rel.extend((dy[valid_rel] / y_prev[valid_rel]).tolist())

        # ---- per-step printing (kept) ----
        if print_individual_steps:
            print(f"Series {series_cols} = {series_key} (n={len(gdf_sorted)})")
            for i in np.where(hits)[0]:
                i0, i1 = i, i + 1
                rel_str = ""
                if y[i0] != 0:
                    rel_str = f"  (Δ{YNAME}/{YNAME}_prev={dy[i] / y[i0]:.3f})"
                print(
                    f"  step {i0}->{i1}: "
                    f"{XNAME}: {x[i0]:.6g} → {x[i1]:.6g}  "
                    f"Δ{XNAME}={dx[i]:.6g}  (Δ{XNAME}/{XNAME}_prev={frac[i]:.3f})   "
                    f"{YNAME}: {y[i0]:.6g} → {y[i1]:.6g}  Δ{YNAME}={dy[i]:.6g}"
                    f"{rel_str}"
                )
            print("")

    # ---- summary stats ----
    print("========================================")
    print(f"SUMMARY over FLAGGED steps (Δ{XNAME}/{XNAME}_prev <= {frac_step_thresh:.2f})")
    print("========================================")
    print(f"Total consecutive steps evaluated: {total_steps}")
    print(f"Flagged steps: {flagged_steps}")

    if flagged_steps == 0:
        print("No flagged steps; no averages to report.")
        return

    arr_dy = np.array(all_flagged_dy, dtype=float)
    dy_mean = float(np.mean(arr_dy))
    dy_std = float(np.std(arr_dy, ddof=1)) if flagged_steps > 1 else float("nan")

    if np.isnan(dy_std):
        print(f"mean Δ{YNAME}: {dy_mean:.6g}   std: NA (n=1)")
    else:
        print(f"mean Δ{YNAME}: {dy_mean:.6g}   std: {dy_std:.6g}")

    arr_dy_rel = np.array(all_flagged_dy_rel, dtype=float)

    if arr_dy_rel.size == 0:
        print(f"mean Δ{YNAME}/{YNAME}_prev: NA")
    else:
        mean_rel = float(np.mean(arr_dy_rel))
        std_rel = float(np.std(arr_dy_rel, ddof=1)) if arr_dy_rel.size > 1 else float("nan")

        if np.isnan(std_rel):
            print(f"mean Δ{YNAME}/{YNAME}_prev: {mean_rel:.4f}   std: NA (n=1)")
        else:
            print(f"mean Δ{YNAME}/{YNAME}_prev: {mean_rel:.4f}   std: {std_rel:.4f}")

    # (Optional but usually useful sanity)
    arr_frac = np.array(all_flagged_frac, dtype=float)
    arr_dx = np.array(all_flagged_dx, dtype=float)

    frac_mean = float(np.mean(arr_frac))
    frac_std = float(np.std(arr_frac, ddof=1)) if flagged_steps > 1 else float('nan')
    dx_mean = float(np.mean(arr_dx))
    dx_std = float(np.std(arr_dx, ddof=1)) if flagged_steps > 1 else float('nan')

    if np.isnan(frac_std):
        print(f"mean (Δ{XNAME}/{XNAME}_prev): {frac_mean:.6g}   std: NA (n=1)")
    else:
        print(f"mean (Δ{XNAME}/{XNAME}_prev): {frac_mean:.6g}   std: {frac_std:.6g}")

    if np.isnan(dx_std):
        print(f"mean Δ{XNAME}: {dx_mean:.6g}   std: NA (n=1)")
    else:
        print(f"mean Δ{XNAME}: {dx_mean:.6g}   std: {dx_std:.6g}")


# =============================
# 6) APPLY TO YOUR CASE (plot26 semantics)
# =============================
required_for_filter = ["Condition", "HMR", "v_distal", "P_d/P_a", "Q_distal", "HSR", "Length", "Stenosis Group", "Location"]
missing = [c for c in required_for_filter if c not in df.columns]
if missing:
    raise KeyError(f"CSV missing columns required for this analysis: {missing}")

df_filtered_third = df[
    (df["Condition"] == "Hyperemic") &
    df["HMR"].notna() &
    df["v_distal"].notna() &
    df["P_d/P_a"].notna() &
    df["Q_distal"].notna() &
    df["HSR"].notna() &
    df["Length"].notna() &
    df["Stenosis Group"].notna()
].copy()

report_small_x_steps_in_series(
    df_filtered_third,
    x_col="Q_distal",
    y_col="P_d/P_a",
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=location_filter,
    show_singletons=False,
    frac_step_thresh=FRAC_STEP_THRESH,
    min_abs_dx=MIN_ABS_DX,
    print_individual_steps=PRINT_INDIVIDUAL_STEPS,
)