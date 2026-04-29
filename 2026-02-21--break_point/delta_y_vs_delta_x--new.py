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

# # flag if Δx/x_prev <= this
# FRAC_STEP_THRESH = 0.30

# # Optional: ignore tiny numerical jitter in x steps
# MIN_ABS_DX = 0.0   # set e.g. 1e-6 if you want

# # Toggle detailed per-step printing
# PRINT_INDIVIDUAL_STEPS = True

# =============================
# 1) READ DATA
# =============================
data_file = "../data/data_manuscript.csv" if manuscript_data else "../data/data.csv"
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

def summarize_slopes_across_specified_steps(
    data,
    *,
    x_col,
    y_col,
    step_pairs,  # e.g. [(1,2), (3,4)]
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=None,
    show_singletons=False,
    order_col=None,
    one_indexed_steps=True,
    min_abs_dx=0.0,
    print_individual=False,
    # NEW:
    min_abs_ref_slope=0.0,   # skip % change if |ref slope| < this (avoid blowups)
):
    def pretty(col):
        if col == "Q_distal":
            return "Q"
        if col == "P_d/P_a":
            return "FRR"
        return col

    XNAME = pretty(x_col)
    YNAME = pretty(y_col)
    order_col = x_col if order_col is None else order_col

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
    needed = [x_col, y_col, order_col] + list(series_cols)
    missing = [c for c in needed if c not in df_plot.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ---- drop NA ----
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[order_col].notna()]
    for c in series_cols:
        df_plot = df_plot[df_plot[c].notna()]

    if len(df_plot) == 0:
        print("No rows to scan after filters / NA drops.")
        return None

    # ---- optionally remove singleton groups ----
    if not show_singletons:
        counts = df_plot.groupby(list(series_cols)).size()
        valid_groups = counts[counts > 1].index
        if len(valid_groups) == 0:
            print("All groups are singletons after filtering; nothing to analyze.")
            return None
        df_plot = (
            df_plot.set_index(list(series_cols))
                  .loc[valid_groups]
                  .reset_index()
        )

    # normalize step pairs to 0-index (and preserve user-given order)
    step_pairs_0 = []
    pair_labels = []
    for (i, j) in step_pairs:
        if one_indexed_steps:
            i0, j0 = int(i) - 1, int(j) - 1
            pair_label = f"{int(i)}→{int(j)}"
        else:
            i0, j0 = int(i), int(j)
            pair_label = f"{int(i)}→{int(j)}"
        if i0 == j0:
            raise ValueError(f"Invalid step pair {(i,j)}: i and j must be different.")
        step_pairs_0.append((i0, j0))
        pair_labels.append(pair_label)

    groups = df_plot.groupby(list(series_cols), dropna=False)

    rows = []
    for series_key, gdf in groups:
        if len(gdf) < 2:
            continue

        gdf_sorted = gdf.sort_values(by=order_col).reset_index(drop=True)

        x = gdf_sorted[x_col].to_numpy(dtype=float)
        y = gdf_sorted[y_col].to_numpy(dtype=float)

        n = len(gdf_sorted)
        for (i0, j0), pair_label in zip(step_pairs_0, pair_labels):
            if i0 < 0 or j0 < 0 or i0 >= n or j0 >= n:
                continue

            dx = x[j0] - x[i0]
            dy = y[j0] - y[i0]

            if not np.isfinite(dx) or not np.isfinite(dy):
                continue
            if abs(dx) < float(min_abs_dx):
                continue

            slope = dy / dx

            rows.append({
                "pair": pair_label,
                "pair_order": pair_labels.index(pair_label),  # preserves your order
                "series_key": series_key,
                "n_in_series": n,
                "i": i0, "j": j0,
                "x_i": x[i0], "x_j": x[j0],
                "y_i": y[i0], "y_j": y[j0],
                "dx": dx, "dy": dy,
                "slope_dy_dx": slope,
            })

            if print_individual:
                print(f"Series {series_cols}={series_key} (n={n})  pair {pair_label}: "
                      f"{XNAME} {x[i0]:.6g}→{x[j0]:.6g} (Δ{XNAME}={dx:.6g}), "
                      f"{YNAME} {y[i0]:.6g}→{y[j0]:.6g} (Δ{YNAME}={dy:.6g}), "
                      f"Δ{YNAME}/Δ{XNAME}={slope:.6g}")

    if len(rows) == 0:
        print("No slopes computed (check step_pairs, filtering, or min_abs_dx).")
        return None

    out = pd.DataFrame(rows)

    # ---- slope summary ----
    print("\n========================================")
    print(f"SUMMARY: Δ{YNAME}/Δ{XNAME} across specified step pairs")
    print("========================================")

    mean_by_pair = {}
    std_by_pair = {}
    n_by_pair = {}

    for pair in pair_labels:  # preserve order exactly as provided
        sdf = out[out["pair"] == pair]
        arr = sdf["slope_dy_dx"].to_numpy(dtype=float)
        n = arr.size
        if n == 0:
            mean_by_pair[pair] = np.nan
            std_by_pair[pair] = np.nan
            n_by_pair[pair] = 0
            print(f"pair {pair}: mean=NA   std=NA   (n=0)")
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        mean_by_pair[pair] = mean
        std_by_pair[pair] = std
        n_by_pair[pair] = n

        if np.isnan(std):
            print(f"pair {pair}: mean={mean:.6g}   std=NA (n=1)")
        else:
            print(f"pair {pair}: mean={mean:.6g}   std={std:.6g}   (n={n})")

    # ---- NEW: percent change in slope between adjacent pairs ----
    if len(pair_labels) >= 2:
        print("\n========================================")
        print("PERCENT CHANGE IN SLOPE BETWEEN PAIRS")
        print("========================================")
        print("Reported as:")
        print("  (1) series-wise % change: mean±std across series with both slopes")
        print("  (2) % change of means: 100*(mean_new - mean_ref)/mean_ref\n")

        # pivot to get per-series slopes per pair
        piv = out.pivot_table(index="series_key", columns="pair", values="slope_dy_dx", aggfunc="mean")

        def pct_change(new, ref):
            return 100.0 * (new - ref) / ref

        # adjacent comparisons in the order you provided
        for a, b in zip(pair_labels[:-1], pair_labels[1:]):
            if a not in piv.columns or b not in piv.columns:
                continue

            ref = piv[a]
            new = piv[b]
            valid = ref.notna() & new.notna() & np.isfinite(ref) & np.isfinite(new) & (np.abs(ref) >= float(min_abs_ref_slope))

            arr = pct_change(new[valid].to_numpy(dtype=float), ref[valid].to_numpy(dtype=float))
            n = arr.size

            # % change of means (based on pooled means you printed above)
            mean_ref = mean_by_pair.get(a, np.nan)
            mean_new = mean_by_pair.get(b, np.nan)
            means_pct = np.nan
            if np.isfinite(mean_ref) and np.isfinite(mean_new) and (abs(mean_ref) >= float(min_abs_ref_slope)):
                means_pct = pct_change(mean_new, mean_ref)

            if n == 0:
                print(f"{a} → {b}: series-wise %Δ=NA (n=0);   %Δ(means)={means_pct:.3g}" if np.isfinite(means_pct)
                      else f"{a} → {b}: series-wise %Δ=NA (n=0);   %Δ(means)=NA")
                continue

            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if n > 1 else float("nan")

            if np.isnan(s):
                line1 = f"{a} → {b}: series-wise %Δ={m:.3g}% (std=NA, n=1)"
            else:
                line1 = f"{a} → {b}: series-wise %Δ={m:.3g}% ± {s:.3g}% (n={n})"

            if np.isfinite(means_pct):
                print(f"{line1};   %Δ(means)={means_pct:.3g}%")
            else:
                print(f"{line1};   %Δ(means)=NA")

        # optional: first → last
        a, b = pair_labels[0], pair_labels[-1]
        if a in piv.columns and b in piv.columns and a != b:
            ref = piv[a]
            new = piv[b]
            valid = ref.notna() & new.notna() & np.isfinite(ref) & np.isfinite(new) & (np.abs(ref) >= float(min_abs_ref_slope))
            arr = 100.0 * (new[valid].to_numpy(dtype=float) - ref[valid].to_numpy(dtype=float)) / ref[valid].to_numpy(dtype=float)
            n = arr.size

            mean_ref = mean_by_pair.get(a, np.nan)
            mean_new = mean_by_pair.get(b, np.nan)
            means_pct = np.nan
            if np.isfinite(mean_ref) and np.isfinite(mean_new) and (abs(mean_ref) >= float(min_abs_ref_slope)):
                means_pct = 100.0 * (mean_new - mean_ref) / mean_ref

            if n > 0:
                m = float(np.mean(arr))
                s = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
                if np.isnan(s):
                    line1 = f"{a} → {b} (first→last): series-wise %Δ={m:.3g}% (std=NA, n=1)"
                else:
                    line1 = f"{a} → {b} (first→last): series-wise %Δ={m:.3g}% ± {s:.3g}% (n={n})"
                if np.isfinite(means_pct):
                    print(f"{line1};   %Δ(means)={means_pct:.3g}%")
                else:
                    print(f"{line1};   %Δ(means)=NA")

    return out

slopes_df = summarize_slopes_across_specified_steps(
    df_filtered_third,
    x_col="HMR",
    y_col="P_d/P_a",                     # hyperemic Pd/Pa = FFR
    step_pairs=[(1, 2), (2,3), (3, 4)],
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=location_filter,
    show_singletons=False,
    order_col="HMR",                      # explicit (default would already be x_col)
    one_indexed_steps=True,
    min_abs_dx=0.0,
    print_individual=False,               # True if you want per-series printing
)

slopes_df = summarize_slopes_across_specified_steps(
    df_filtered_third,
    x_col="HMR",
    y_col="CFR",                     # hyperemic Pd/Pa = FFR
    step_pairs=[(1, 2), (2,3), (3, 4)],
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=location_filter,
    show_singletons=False,
    order_col="HMR",                      # explicit (default would already be x_col)
    one_indexed_steps=True,
    min_abs_dx=0.0,
    print_individual=False,               # True if you want per-series printing
)