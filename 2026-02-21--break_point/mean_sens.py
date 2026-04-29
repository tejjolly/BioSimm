#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe

try:
    import ruptures as rpt
except ImportError as e:
    raise ImportError("ruptures is not installed. Install with: pip install ruptures") from e


# =============================
# CONFIG / FLAGS
# =============================
manuscript_data = True

# choose which vessels to include
all_flag = False
LAD_flag = True
LCX_flag = True

USE_LOGLOG = True

# ---- Analysis choices ----
X_COL = "HMR"          # cutoff is on this variable
Y_COL_ANALYSIS = "CFR" # sensitivity computed for this variable (e.g., "P_d/P_a" for FFR, "CFR" for CFR)
Y_COL_PLOT = Y_COL_ANALYSIS     # plotted on y-axis (often same as Y_COL_ANALYSIS)
COLOR_COL = "FFR"      # point coloring for plot

INCLUDE_NO_STENOSIS_SENS = True   # exclude astenotic series for sensitivity/cutoff
INCLUDE_NO_STENOSIS_PLOT = True    # you can still plot astenotic points if you want

# ---- Change-point inference ----
MIN_PER_SIDE = 1
N_PERM = 5000
SEED = 1

# ---- Plot styling ----
CMAP_NAME = "BuPu"
HSR_boundaries = np.linspace(0.1, 1.3, 5)
SAVEFIG = True
OUTDIR = "images"
FIGSIZE = (8.5, 5)

# Optional: label points with Geometry Number (if present)
LABELS = False

plt.rcParams.update({
    'font.size': 20,  # Increase base font size
    'axes.labelsize': 18,  # Axis label font size
    'axes.titlesize': 18,  # Title font size
    'xtick.labelsize': 18,  # X-tick label font size
    'ytick.labelsize': 18,  # Y-tick label font size
    'legend.fontsize': 18,  # Legend font size
    'figure.dpi': 600  # Higher DPI for clearer text in smaller figure
})


# =============================
# 1) READ DATA
# =============================
data_file = "../data/data_manuscript.csv" if manuscript_data else "../data/break_test.csv"
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

# convenience columns
if "Condition" in df.columns and "P_d/P_a" in df.columns:
    df["iFR"] = np.where(df["Condition"] == "Non-hyperemic", df["P_d/P_a"], np.nan)
    df["FFR"] = np.where(df["Condition"] == "Hyperemic", df["P_d/P_a"], np.nan)

# =============================
# 3) LOCATION FILTER
# =============================
location_filter = (
    ["LAD", "LCX"] if all_flag or (LAD_flag and LCX_flag)
    else "LAD" if LAD_flag
    else "LCX" if LCX_flag
    else None
)

# =============================
# 4) STENOSIS GROUP
# =============================
def stenosis_group(val, decimal_places=2, tolerance=0.02):
    if pd.isna(val):
        return np.nan
    rounded_val = round(val, decimal_places)
    return rounded_val if abs(val - rounded_val) <= tolerance else np.nan

if "Stenosis Percentage" not in df.columns:
    raise KeyError("Your CSV does not contain 'Stenosis Percentage'.")

df["Stenosis Group"] = df["Stenosis Percentage"].apply(stenosis_group)

# =============================
# 5) PLOT HELPERS (match your style)
# =============================
def snap_length_style(val, tol=0.15):
    # 1.2 cm → ':', 2.5 cm → '--'  (your original)
    if pd.isna(val):
        return "solid"
    if np.isclose(val, 1.2, atol=tol):
        return ":"
    if np.isclose(val, 2.5, atol=tol):
        return "--"
    return "solid"

def vessel_line_color(loc):
    # keep neutral (you had grey for both)
    if loc in ("LAD", "LCX"):
        return "grey"
    return "gray"

def stenosis_marker(sten_val):
    if pd.isna(sten_val): return "x"
    if sten_val < 0.10: return "o"
    if 0.40 <= sten_val <= 0.50: return "s"
    if 0.55 <= sten_val <= 0.65: return "^"
    return "x"

# =============================
# 6) BUILD STEPS DF (midpoints + sens)
# =============================
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
      - series: tuple key for the series (series_cols)
      - step: 0,1,2,...
      - hmr_mid: geometric midpoint of x over that step (name kept for compatibility)
      - y_mid: arithmetic midpoint of y over that step (for plotting midpoints)
      - sens: per-step sensitivity (log–log elasticity by default)
    NOTE: 'hmr_mid' is the midpoint of x_col even if x_col != 'HMR'.
    """
    dfp = data.copy()

    # optional location filter
    if location_filter is not None:
        if location_col not in dfp.columns:
            raise KeyError(f"location_col='{location_col}' not in dataframe columns.")
        if isinstance(location_filter, (list, tuple, set)):
            dfp = dfp[dfp[location_col].isin(location_filter)]
        else:
            dfp = dfp[dfp[location_col] == location_filter]

    # optional exclude no-stenosis
    if not include_no_stenosis:
        if "Stenosis Group" not in dfp.columns:
            raise KeyError("Stenosis Group column required to exclude no-stenosis cases.")
        dfp = dfp[dfp["Stenosis Group"] != 0.0]

    # required columns
    needed = [x_col, y_col] + list(series_cols)
    missing = [c for c in needed if c not in dfp.columns]
    if missing:
        raise KeyError(f"Missing required columns for step building: {missing}")

    # drop NA
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
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]

            # midpoint for plotting
            x_mid = float(np.sqrt(x0 * x1)) if (x0 > 0 and x1 > 0) else float(0.5 * (x0 + x1))
            y_mid = float(0.5 * (y0 + y1))

            if use_loglog:
                # requires positive values
                if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0:
                    continue
                dlogx = np.log(x1 / x0)
                if dlogx == 0:
                    continue
                sens = float(np.log(y1 / y0) / dlogx)
            else:
                # ratio-of-fractions approximation
                if x0 == 0 or y0 == 0:
                    continue
                sens = float(((y1 - y0) / y0) / ((x1 - x0) / x0))

            rows.append({"series": key, "step": i, "hmr_mid": x_mid, "y_mid": y_mid, "sens": sens})

    return pd.DataFrame(rows)

# =============================
# 7) RUPTURES CHANGE-POINT + BLOCKED PERMUTATION P
# =============================
def find_cutoff_piecewise_constant_ruptures(
    steps_df,
    *,
    min_per_side=6,
    n_perm=5000,
    seed=0,
    model="l2",
):
    """
    One breakpoint in mean(sens) across sorted hmr_mid using ruptures Binseg + L2.
    P-value: within-series permutation on sens (blocked permutation).
    """
    rng = np.random.default_rng(seed)

    req = {"hmr_mid", "sens", "series"}
    missing = [c for c in req if c not in steps_df.columns]
    if missing:
        raise KeyError(f"steps_df missing required columns: {missing}")

    dfc = steps_df.dropna(subset=["hmr_mid", "sens", "series"]).copy()
    if len(dfc) < 2 * min_per_side + 1:
        raise RuntimeError(f"Not enough step observations ({len(dfc)}) for min_per_side={min_per_side}.")

    # sort by x-midpoint
    dfc = dfc.sort_values("hmr_mid").reset_index(drop=True)
    x = dfc["hmr_mid"].to_numpy(float)
    y = dfc["sens"].to_numpy(float)

    def _segment_once(y_vec):
        signal = y_vec.reshape(-1, 1)
        algo = rpt.Binseg(model=model, min_size=int(min_per_side), jump=1).fit(signal)
        bkpts = algo.predict(n_bkps=1)  # [t, n]
        t = int(bkpts[0])

        n = len(y_vec)
        if not (min_per_side <= t <= n - min_per_side):
            raise RuntimeError(f"Invalid breakpoint t={t} for n={n}, min_per_side={min_per_side}.")

        mu_lo = float(y_vec[:t].mean())
        mu_hi = float(y_vec[t:].mean())
        sse = float(((y_vec[:t] - mu_lo) ** 2).sum() + ((y_vec[t:] - mu_hi) ** 2).sum())

        # cutoff between x[t-1] and x[t] (geometric boundary)
        cutoff = float(np.sqrt(x[t - 1] * x[t]))
        return t, cutoff, mu_lo, mu_hi, sse

    # observed
    t_obs, cutoff_obs, mu_lo_obs, mu_hi_obs, sse_obs = _segment_once(y)

    # null
    mu_all = float(y.mean())
    sse_null = float(((y - mu_all) ** 2).sum())
    improvement_obs = sse_null - sse_obs

    # permute within series (in sorted df space)
    series_to_idx = dfc.groupby("series").indices
    improvements = np.empty(int(n_perm), dtype=float)

    for k in range(int(n_perm)):
        y_perm = y.copy()
        for _, idx in series_to_idx.items():
            y_perm[idx] = rng.permutation(y_perm[idx])
        _, _, _, _, sse_p = _segment_once(y_perm)
        improvements[k] = sse_null - sse_p

    p_value = float((np.sum(improvements >= improvement_obs) + 1) / (len(improvements) + 1))

    return {
        "cutoff": float(cutoff_obs),
        "mean_sens_low": float(mu_lo_obs),
        "mean_sens_high": float(mu_hi_obs),
        "delta": float(mu_hi_obs - mu_lo_obs),
        "improvement": float(improvement_obs),
        "p_value_perm": float(p_value),
        "n_steps": int(len(dfc)),
        "break_index": int(t_obs),
    }

# =============================
# 8) PLOT: points + midpoints + breakpoint
# =============================
def plot_with_midpoints_and_cutoff(
    df_data,
    steps_df,
    result,
    *,
    x_col,
    y_col,
    color_col,
    location_col="Location",
    location_filter=None,
    figsize=(8.5, 5),
    cmap_name="BuPu",
    custom_boundaries=None,
    color_label="",
    connect_series=True,
    alpha_scatter=0.8,
    s_scatter=60,
    labels=False,
    savefig=False,
    outdir="images",
    fname="plot_with_cutoff",
    add_threshold=None,
):
    plt.figure(figsize=figsize)

    df_plot = df_data.copy()

    # optional location filter
    if location_filter is not None:
        if isinstance(location_filter, (list, tuple, set)):
            df_plot = df_plot[df_plot[location_col].isin(location_filter)]
        else:
            df_plot = df_plot[df_plot[location_col] == location_filter]

    # drop NA for plotting
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna() & df_plot[color_col].notna()]
    df_plot = df_plot[df_plot["Stenosis Group"].notna() & df_plot["Length"].notna() & df_plot[location_col].notna()]

    # color norm
    if custom_boundaries is None:
        cmin, cmax = df_plot[color_col].min(), df_plot[color_col].max()
        custom_boundaries = np.linspace(cmin, cmax, 6)

    norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
    cmap = plt.get_cmap(cmap_name)

    groups = df_plot.groupby(["Stenosis Group", "Length", location_col], dropna=False)

    first_scatter = None
    for (sten_val, length_val, loc), gdf in groups:
        marker_style = stenosis_marker(sten_val)
        linestyle = snap_length_style(length_val, tol=0.15)
        line_color = vessel_line_color(loc)

        sc = plt.scatter(
            gdf[x_col], gdf[y_col],
            c=gdf[color_col], cmap=cmap, norm=norm,
            edgecolor="k", alpha=alpha_scatter, s=s_scatter,
            marker=marker_style, zorder=3
        )
        if first_scatter is None:
            first_scatter = sc

        if labels and ("Geometry Number" in gdf.columns):
            for xi, yi, lab in zip(gdf[x_col], gdf[y_col], gdf["Geometry Number"]):
                plt.text(
                    xi, yi, str(lab), fontsize=8, ha="right", va="top",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
                )

        if connect_series and len(gdf) > 1:
            gdf_sorted = gdf.sort_values(by=x_col)
            plt.plot(
                gdf_sorted[x_col], gdf_sorted[y_col],
                linestyle=linestyle, color=line_color,
                alpha=0.8, linewidth=2.0, zorder=1
            )

    # colorbar
    if first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label)

    # thresholds
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get("axis", "y")
            value = tdict.get("value", 0.0)
            style = tdict.get("style", "--")
            c = tdict.get("color", "gray")
            w = tdict.get("width", 2.5)
            if axis_type == "y":
                plt.axhline(y=value, color=c, linestyle=style, linewidth=w, zorder=2)
            else:
                plt.axvline(x=value, color=c, linestyle=style, linewidth=w, zorder=2)

    # midpoints (red circles)
    if steps_df is not None and len(steps_df) > 0:
        plt.scatter(
            steps_df["hmr_mid"].to_numpy(float),
            steps_df["y_mid"].to_numpy(float),
            s=80,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=5
        )

    # breakpoint (vertical dashed red)
    if result is not None and ("cutoff" in result):
        plt.axvline(
            x=float(result["cutoff"]),
            color="red",
            linestyle="--",
            linewidth=2.5,
            zorder=2
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(False)
    plt.tight_layout()

    if savefig:
        plt.savefig(f"{outdir}/{fname}.png", dpi=600, transparent=True, bbox_inches="tight")
        plt.savefig(f"{outdir}/{fname}.svg", transparent=True, bbox_inches="tight")
        print(f"saved → {outdir}/{fname}.png/.svg")

    plt.show()
    plt.close()

import itertools
import numpy as np

def find_cutoff_piecewise_constant_exact_circular_shift(
    steps_df,
    *,
    min_per_side=6,
    seed=0,          # unused, kept for signature similarity
):
    """
    One breakpoint in mean(sens) across sorted hmr_mid (piecewise-constant mean, L2 loss).
    Exact p-value under a circular-shift-within-series null.

    Null: for each series, circularly shift its sens values along its own x-order positions
          (as they appear in the globally x-sorted array). Enumerate all combinations.

    Feasible when product_i m_i is moderate (here: 3^9 = 19683).
    """
    req = {"hmr_mid", "sens", "series"}
    missing = [c for c in req if c not in steps_df.columns]
    if missing:
        raise KeyError(f"steps_df missing required columns: {missing}")

    dfc = steps_df.dropna(subset=["hmr_mid", "sens", "series"]).copy()
    if len(dfc) < 2 * min_per_side + 1:
        raise RuntimeError(f"Not enough step observations ({len(dfc)}) for min_per_side={min_per_side}.")

    # sort by x-midpoint
    dfc = dfc.sort_values("hmr_mid").reset_index(drop=True)
    x = dfc["hmr_mid"].to_numpy(float)
    y = dfc["sens"].to_numpy(float)
    n = len(y)

    # --- helpers: best 1-break SSE by scanning all valid t ---
    def best_two_segment_fit(y_vec):
        mu_all = float(y_vec.mean())
        sse_null = float(((y_vec - mu_all) ** 2).sum())

        best_sse = np.inf
        best_t = None
        best_mu_lo = None
        best_mu_hi = None

        t_min = int(min_per_side)
        t_max = int(n - min_per_side)
        for t in range(t_min, t_max + 1):
            mu_lo = float(y_vec[:t].mean())
            mu_hi = float(y_vec[t:].mean())
            sse = float(((y_vec[:t] - mu_lo) ** 2).sum() + ((y_vec[t:] - mu_hi) ** 2).sum())
            if sse < best_sse:
                best_sse = sse
                best_t = t
                best_mu_lo = mu_lo
                best_mu_hi = mu_hi

        if best_t is None:
            raise RuntimeError("No valid breakpoint found (check min_per_side vs n).")

        improvement = sse_null - best_sse
        cutoff = float(np.sqrt(x[best_t - 1] * x[best_t]))  # geometric boundary
        return best_t, cutoff, best_mu_lo, best_mu_hi, sse_null, best_sse, improvement

    # observed statistic (search over t)
    t_obs, cutoff_obs, mu_lo_obs, mu_hi_obs, sse_null, sse_obs, improvement_obs = best_two_segment_fit(y)

    # --- exact circular-shift null ---
    # Get indices (in dfc order) for each series; sort them to define the within-series x-order
    series_to_idx = {k: np.sort(v) for k, v in dfc.groupby("series").indices.items()}

    # Precompute all circular shifts per series
    rolled_by_series = {}
    shift_ranges = []
    series_keys = []

    for k, idx in series_to_idx.items():
        idx = np.asarray(idx, dtype=int)
        m = len(idx)
        if m < 2:
            continue  # shouldn't happen; but harmless
        y_sub = y[idx].copy()

        rolled = [np.roll(y_sub, shift=s) for s in range(m)]
        rolled_by_series[k] = (idx, rolled)
        shift_ranges.append(range(m))
        series_keys.append(k)

    # total null configurations
    N_null = 1
    for r in shift_ranges:
        N_null *= len(list(r))

    extreme = 0
    max_null_improvement = -np.inf

    # Enumerate all combinations of circular shifts across series
    for shifts in itertools.product(*shift_ranges):
        y_perm = y.copy()
        for k, s in zip(series_keys, shifts):
            idx, rolled = rolled_by_series[k]
            y_perm[idx] = rolled[s]

        # re-fit breakpoint on permuted y (search over t again)
        _, _, _, _, _, _, imp = best_two_segment_fit(y_perm)

        if imp >= improvement_obs:
            extreme += 1
        if imp > max_null_improvement:
            max_null_improvement = imp

    # exact p-value (with +1 correction for continuity, matching your MC style)
    p_value = float((extreme + 1) / (N_null + 1))

    return {
        "cutoff": float(cutoff_obs),
        "mean_sens_low": float(mu_lo_obs),
        "mean_sens_high": float(mu_hi_obs),
        "delta": float(mu_hi_obs - mu_lo_obs),
        "improvement": float(improvement_obs),
        "p_value_shift_exact": float(p_value),
        "n_null": int(N_null),
        "extreme": int(extreme),
        "max_null_improvement": float(max_null_improvement),
        "n_steps": int(n),
        "break_index": int(t_obs),
    }


# =============================
# 9) APPLY (filter -> steps -> cutoff -> plot)
# =============================
required = [
    "Condition", "Length", "Stenosis Group", "Location",
    X_COL, Y_COL_ANALYSIS, Y_COL_PLOT, COLOR_COL
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"CSV missing required columns: {missing}")

# base filter (hyperemic only for FFR/CFR-type analyses; adjust if needed)
df_base = df[
    (df["Condition"] == "Hyperemic") &
    df[X_COL].notna() &
    df[Y_COL_ANALYSIS].notna() &
    df[Y_COL_PLOT].notna() &
    df[COLOR_COL].notna() &
    df["Length"].notna() &
    df["Stenosis Group"].notna() &
    df["Location"].notna()
].copy()

# apply plotting-only stenosis inclusion choice
if not INCLUDE_NO_STENOSIS_PLOT:
    df_plot = df_base[df_base["Stenosis Group"] != 0.0].copy()
else:
    df_plot = df_base.copy()

# build steps_df for the sensitivity/cutoff (may exclude astenotic cases)
steps_df = build_steps_df_for_cutoff(
    df_base,
    x_col=X_COL,
    y_col=Y_COL_ANALYSIS,
    series_cols=("Stenosis Group", "Length", "Location"),
    location_col="Location",
    location_filter=location_filter,
    include_no_stenosis=INCLUDE_NO_STENOSIS_SENS,
    use_loglog=USE_LOGLOG,
)

print("\nsteps_df sanity:")
print(steps_df[["hmr_mid", "sens"]].describe())
print("n steps total =", len(steps_df))

result = find_cutoff_piecewise_constant_exact_circular_shift(
    steps_df,
    min_per_side=MIN_PER_SIDE,
)
print("\ncutoff result:")
print(result)
# plot with overlays using the SAME steps_df and result
fname = f"{Y_COL_PLOT}_vs_{X_COL}_col_{COLOR_COL}_with_cutoff"
plot_with_midpoints_and_cutoff(
    df_plot,
    steps_df=steps_df.rename(columns={"y_mid": "y_mid"}),  # explicit, no-op
    result=result,
    x_col=X_COL,
    y_col=Y_COL_PLOT,
    color_col=COLOR_COL,
    location_col="Location",
    location_filter=location_filter,
    figsize=FIGSIZE,
    cmap_name=CMAP_NAME,
    custom_boundaries=HSR_boundaries,
    color_label="HSR [mmHg/cm/s]" if COLOR_COL == "HSR" else COLOR_COL,
    connect_series=True,
    alpha_scatter=0.8,
    s_scatter=60,
    labels=LABELS,
    savefig=SAVEFIG,
    outdir=OUTDIR,
    fname=fname,
    # add_threshold=[{'axis': 'y', 'value': 2.0}]
)

# dfc = steps_df.dropna(subset=["hmr_mid","sens","series"]).copy()
# dfc = dfc.sort_values("hmr_mid").reset_index(drop=True)
# dfc["pos"] = np.arange(len(dfc))
#
# def interleave_stats(g):
#     pos = np.sort(g["pos"].to_numpy())
#     gaps = np.diff(pos)
#     n_blocks = 1 + np.sum(gaps > 1)          # how many separated chunks in global order
#     span = pos[-1] - pos[0]                 # how wide across the global order
#     return pd.Series({
#         "n_steps": len(pos),
#         "n_blocks": int(n_blocks),
#         "span_in_positions": int(span),
#         "min_hmr_mid": float(g["hmr_mid"].min()),
#         "max_hmr_mid": float(g["hmr_mid"].max()),
#     })
#
# diag = dfc.groupby("series").apply(interleave_stats).sort_values(
#     ["n_blocks","span_in_positions"], ascending=[True, True]
# )
#
# print(diag)
# print("\nFraction of series with n_blocks == 1:", (diag["n_blocks"] == 1).mean())
