#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# =============================
# CONFIG / FLAGS
# =============================
manuscript_data = True

# choose which vessels to include
all_flag = True
LAD_flag = False
LCX_flag = False

# flag if Δx/x_prev <= this
FRAC_STEP_THRESH = 0.30

# Optional: ignore tiny numerical jitter in x steps
MIN_ABS_DX = 0.0   # set e.g. 1e-6 if you want

# Toggle detailed per-step printing
PRINT_INDIVIDUAL_STEPS = True

# Toggle Part 2 printing (Δy/Δx for every consecutive pair)
PRINT_SLOPES = True

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
# 5) PART 1: FLAGGED STEPS (individual printing + summary averages)
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

    # (Optional sanity)
    arr_frac = np.array(all_flagged_frac, dtype=float)
    arr_dx = np.array(all_flagged_dx, dtype=float)

    frac_mean = float(np.mean(arr_frac))
    frac_std = float(np.std(arr_frac, ddof=1)) if flagged_steps > 1 else float("nan")
    dx_mean = float(np.mean(arr_dx))
    dx_std = float(np.std(arr_dx, ddof=1)) if flagged_steps > 1 else float("nan")

    if np.isnan(frac_std):
        print(f"mean (Δ{XNAME}/{XNAME}_prev): {frac_mean:.6g}   std: NA (n=1)")
    else:
        print(f"mean (Δ{XNAME}/{XNAME}_prev): {frac_mean:.6g}   std: {frac_std:.6g}")

    if np.isnan(dx_std):
        print(f"mean Δ{XNAME}: {dx_mean:.6g}   std: NA (n=1)")
    else:
        print(f"mean Δ{XNAME}: {dx_mean:.6g}   std: {dx_std:.6g}")


# =============================
# 6) PART 2: RELATIVE CONSECUTIVE CHANGES
#     WITH STEP-WISE AVERAGES
# =============================
def report_relative_consecutive_changes(
    data,
    *,
    x_col,
    y_col,
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=None,
    show_singletons=False,
    min_abs_dx=0.0,
    print_individual=True,
    include_no_stenosis=False,
):
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
        if isinstance(location_filter, (list, tuple, set)):
            df_plot = df_plot[df_plot[location_col].isin(location_filter)]
        else:
            df_plot = df_plot[df_plot[location_col] == location_filter]

    # ---- NEW: optionally exclude no-stenosis cases ----
    if not include_no_stenosis:
        if "Stenosis Group" not in df_plot.columns:
            raise KeyError("Stenosis Group column required to exclude no-stenosis cases.")
        df_plot = df_plot[df_plot["Stenosis Group"] != 0.0]

    # ---- drop NA ----
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna()]
    for c in series_cols:
        df_plot = df_plot[df_plot[c].notna()]

    if not show_singletons:
        counts = df_plot.groupby(list(series_cols)).size()
        valid = counts[counts > 1].index
        if len(valid) == 0:
            print("No multi-point series for relative-change analysis.")
            return
        df_plot = (
            df_plot.set_index(list(series_cols))
                   .loc[valid]
                   .reset_index()
        )

    groups = df_plot.groupby(list(series_cols), dropna=False)

    # storage by step index: i -> list of values
    step_frac_dx = {}   # Δx/x_prev
    step_frac_dy = {}   # Δy/y_prev

    print("\n========================================")
    print("PART 2: RELATIVE CONSECUTIVE CHANGES")
    print(f"        (Δ{XNAME}/{XNAME}_prev , Δ{YNAME}/{YNAME}_prev)")
    print("========================================\n")

    for series_key, gdf in groups:
        if len(gdf) < 2:
            continue

        gdf = gdf.sort_values(by=x_col).reset_index(drop=True)

        x = gdf[x_col].to_numpy(float)
        y = gdf[y_col].to_numpy(float)

        dx = np.diff(x)
        dy = np.diff(y)
        x_prev = x[:-1]
        y_prev = y[:-1]

        valid = (
            (x_prev != 0.0) &
            (y_prev != 0.0) &
            np.isfinite(dx) &
            np.isfinite(dy) &
            (np.abs(dx) >= float(min_abs_dx))
        )

        if not np.any(valid):
            continue

        if print_individual:
            print(f"Series {series_cols} = {series_key}")

        idx = np.where(valid)[0]
        for i in idx:
            fx = dx[i] / x_prev[i]
            fy = dy[i] / y_prev[i]

            step_frac_dx.setdefault(i, []).append(fx)
            step_frac_dy.setdefault(i, []).append(fy)

            if print_individual:
                print(
                    f"  {i}->{i+1}: "
                    f"Δ{XNAME}/{XNAME}_prev={fx: .3f}, "
                    f"Δ{YNAME}/{YNAME}_prev={fy: .3f}"
                )

        if print_individual:
            print("")

    # ---- STEP-WISE AVERAGES ----
    print("\n-------- STEP-WISE AVERAGES (across series) --------")
    if len(step_frac_dx) == 0:
        print("No valid relative steps to summarize.")
        return

    for i in sorted(step_frac_dx.keys()):
        fx = np.array(step_frac_dx[i], dtype=float)
        fy = np.array(step_frac_dy[i], dtype=float)

        n = fx.size
        fx_mean = fx.mean()
        fx_std = fx.std(ddof=1) if n > 1 else np.nan
        fy_mean = fy.mean()
        fy_std = fy.std(ddof=1) if n > 1 else np.nan

        print(
            f"step {i}->{i+1}:  "
            f"Δ{XNAME}/{XNAME}_prev = {fx_mean:+.3f} ± {fx_std:.3f},  "
            f"Δ{YNAME}/{YNAME}_prev = {fy_mean:+.3f} ± {fy_std:.3f}  "
            f"(n={n})"
        )

def find_cutoff_piecewise_constant(steps_df, min_per_side=6, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    df = steps_df.dropna(subset=["hmr_mid","sens","series"]).copy()

    # Candidate cutoffs from interior values (avoid extreme cutpoints)
    cand = np.unique(df["hmr_mid"].values)
    cand = cand[(cand > np.quantile(cand, 0.15)) & (cand < np.quantile(cand, 0.85))]

    def best_sse(df_in):
        best = (np.inf, None, None, None)  # sse, c, mu_lo, mu_hi
        for c in cand:
            lo = df_in[df_in["hmr_mid"] <= c]["sens"].values
            hi = df_in[df_in["hmr_mid"] >  c]["sens"].values
            if (lo.size < min_per_side) or (hi.size < min_per_side):
                continue
            mu_lo = lo.mean()
            mu_hi = hi.mean()
            sse = ((lo - mu_lo)**2).sum() + ((hi - mu_hi)**2).sum()
            if sse < best[0]:
                best = (sse, c, mu_lo, mu_hi)
        return best

    # Observed best cutoff
    sse_obs, c_obs, mu_lo_obs, mu_hi_obs = best_sse(df)
    if c_obs is None:
        raise RuntimeError("No viable cutoff found; relax min_per_side or check data.")

    # Baseline (no cutoff): one mean
    mu_all = df["sens"].mean()
    sse_null = ((df["sens"].values - mu_all)**2).sum()
    improvement_obs = sse_null - sse_obs  # larger => better evidence of cutoff

    # Permute sens within series (keeps series structure)
    improvements = []
    for _ in range(n_perm):
        dfp = df.copy()
        for s, idx in dfp.groupby("series").groups.items():
            dfp.loc[idx, "sens"] = rng.permutation(dfp.loc[idx, "sens"].values)
        sse_p, _, _, _ = best_sse(dfp)
        improvement_p = sse_null - sse_p
        improvements.append(improvement_p)

    improvements = np.array(improvements)
    p_value = (np.sum(improvements >= improvement_obs) + 1) / (n_perm + 1)

    return {
        "cutoff_hmr": float(c_obs),
        "mean_sens_low": float(mu_lo_obs),
        "mean_sens_high": float(mu_hi_obs),
        "delta": float(mu_hi_obs - mu_lo_obs),
        "improvement": float(improvement_obs),
        "p_value_perm": float(p_value),
        "n_steps": int(len(df)),
    }

def build_steps_df_for_cutoff(
    data,
    *,
    x_col,
    y_col,
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=None,
    include_no_stenosis=True,
    use_loglog=True,
):
    """
    Returns a per-step dataframe with columns:
      - series: tuple key for the series
      - step: 0,1,2,...
      - hmr_mid: geometric midpoint of x over that step (name kept for compatibility)
      - sens: per-step sensitivity (log-log elasticity by default)
    NOTE: 'hmr_mid' is just the midpoint of x_col, even if x_col != 'HMR'.
    """
    dfp = data.copy()

    # optional location filter (same semantics as your other funcs)
    if location_filter is not None:
        if isinstance(location_filter, (list, tuple, set)):
            dfp = dfp[dfp[location_col].isin(location_filter)]
        else:
            dfp = dfp[dfp[location_col] == location_filter]

    # optional exclude no-stenosis
    if not include_no_stenosis:
        if "Stenosis Group" not in dfp.columns:
            raise KeyError("Stenosis Group column required to exclude no-stenosis cases.")
        dfp = dfp[dfp["Stenosis Group"] != 0.0]

    # drop NA in x/y and keys
    needed = [x_col, y_col] + list(series_cols)
    missing = [c for c in needed if c not in dfp.columns]
    if missing:
        raise KeyError(f"Missing required columns for step building: {missing}")

    dfp = dfp[dfp[x_col].notna() & dfp[y_col].notna()]
    for c in series_cols:
        dfp = dfp[dfp[c].notna()]

    rows = []
    for key, g in dfp.groupby(list(series_cols), dropna=False):
        g = g.sort_values(by=x_col).reset_index(drop=True)
        if len(g) < 2:
            continue

        x = g[x_col].to_numpy(float)
        y = g[y_col].to_numpy(float)

        for i in range(len(g) - 1):
            x0, x1 = x[i], x[i+1]
            y0, y1 = y[i], y[i+1]

            # require positive values for log-log; also avoid zero division
            if use_loglog:
                if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0:
                    continue
                dlogx = np.log(x1 / x0)
                if dlogx == 0:
                    continue
                sens = np.log(y1 / y0) / dlogx
                x_mid = np.sqrt(x0 * x1)  # geometric midpoint (natural on log scale)
            else:
                # fallback: your ratio-of-fractions approximation
                if x0 == 0 or y0 == 0:
                    continue
                sens = ((y1 - y0) / y0) / ((x1 - x0) / x0)
                x_mid = 0.5 * (x0 + x1)

            rows.append({"series": key, "step": i, "hmr_mid": x_mid, "sens": sens})

    return pd.DataFrame(rows)


# =============================
# 7) APPLY TO YOUR CASE (plot26 semantics)
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

# Part 1: flagged steps
# report_small_x_steps_in_series(
#     df_filtered_third,
#     x_col="HMR",
#     y_col="P_d/P_a",
#     series_cols=("Stenosis Group", "Length", "Location"),
#     location_col="Location",
#     location_filter=location_filter,
#     show_singletons=False,
#     frac_step_thresh=FRAC_STEP_THRESH,
#     min_abs_dx=MIN_ABS_DX,
#     print_individual_steps=PRINT_INDIVIDUAL_STEPS,
# )

# report_relative_consecutive_changes(
#     df_filtered_third,
#     x_col="Q_distal",
#     y_col="P_d/P_a",
#     series_cols=("Stenosis Group", "Length", "Location"),
#     location_col="Location",
#     location_filter=location_filter,
#     show_singletons=False,
#     min_abs_dx=MIN_ABS_DX,
#     print_individual=True,
#     include_no_stenosis=True
# )

# Build per-step sensitivity table for cutoff search
# IMPORTANT: x_col here determines what cutoff you're finding.
# If you want an HMR cutoff, use x_col="HMR".
steps_df = build_steps_df_for_cutoff(
    df_filtered_third,
    x_col="HMR",          # <-- change to "Q_distal" only if you want a FLOW cutoff instead
    y_col="P_d/P_a",
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=location_filter,
    include_no_stenosis=False,
    use_loglog=False,
)

print("\nsteps_df sanity:")
print(steps_df[["hmr_mid","sens"]].describe())
print("n steps total =", len(steps_df))

result = find_cutoff_piecewise_constant(steps_df, min_per_side=6, n_perm=5000, seed=1)
print(result)


