#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global (pooled) breakpoint detection for y ~ x across multiple series.

FE segmented model (one global breakpoint b shared by all series):
  y_{s,i} = alpha_s + beta1 * x_{s,i} + beta2 * (x_{s,i} - b)_+ + eps_{s,i}

What this script outputs (and nothing else):
  - b_hat (breakpoint)
  - bootstrap p-value for "no breakpoint" (NULL residual bootstrap; sup-F)
  - 95% bootstrap CI for b_hat (ALT residual bootstrap; re-estimate b each draw)

Plotting style + scatter function are left intact.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe


# =============================================================================
# USER FLAGS / STYLE (as provided)
# =============================================================================
manuscript_data = False
test_data = True

all_flag = False
LAD_flag = True
LCX_flag = False

location_filter = (['LAD', 'LCX'] if all_flag or (LAD_flag and LCX_flag)
                   else 'LAD' if LAD_flag
                   else 'LCX' if LCX_flag
                   else None)

original_settings = False
if original_settings:
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 8,
        'figure.dpi': 600
    })
else:
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.dpi': 600
    })


# =============================================================================
# CONFIG FOR BREAKPOINT MODEL
# =============================================================================
X_COL = "HMR"
Y_COL = "FFR"
COLOR_COL = "HMR"
COLOR_CMAP = "BuPu"

SERIES_COLS = ["Location", "Stenosis Group", "Length"]  # defines a "series"
MIN_POINTS_PER_SERIES = 3

# Bootstrap settings
N_BOOT = 999
BOOT_SEED = 7

# Optional filter (KEEP THIS)
STENOSIS_MIN = 0.05

# Plot settings
OUTDIR = "images"
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# DATA LOAD / CLEAN
# =============================================================================
if manuscript_data:
    data_file = "../data/data_manuscript.csv"
elif test_data:
    data_file = "../data/break_test.csv"
else:
    data_file = "../data/data.csv"

df = pd.read_csv(data_file)

cols_to_num = [
    "CFR", "P_d/P_a", "BMR/HMR", "R_total",
    "Stenosis Percentage", "Length", "HMR", "HSR",
    "P_Loss_Coeff", "Q_distal", "v_distal"
]
for col in cols_to_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional BMR
if "BMR" not in df.columns and ("BMR/HMR" in df.columns) and ("HMR" in df.columns):
    df["BMR"] = df["BMR/HMR"] * df["HMR"]

# Create iFR / FFR
if "Condition" in df.columns and "P_d/P_a" in df.columns:
    df["iFR"] = np.where(df["Condition"] == "Non-hyperemic", df["P_d/P_a"], np.nan)
    df["FFR"] = np.where(df["Condition"] == "Hyperemic", df["P_d/P_a"], np.nan)


# =============================================================================
# HELPERS (your style)
# =============================================================================
def stenosis_group(val, decimal_places=2, tolerance=0.02):
    """
    Round `val` to `decimal_places` and group it if within ±tolerance of that rounded value.
    Returns NaN if out of range, so it won't be grouped.
    """
    if pd.isna(val):
        return np.nan
    rounded_val = round(val, decimal_places)
    if abs(val - rounded_val) <= tolerance:
        return rounded_val
    else:
        return np.nan

def snap_length_style(val, tol=0.15):
    """Return linestyle for a length value: 1.2 cm → ':' , 2.5 cm → '--'."""
    if pd.isna(val):
        return "solid"
    if np.isclose(val, 1.2, atol=tol):
        return ":"
    if np.isclose(val, 2.5, atol=tol):
        return "--"
    return "solid"

def vessel_line_color(loc):
    """LAD→grey, LCX→grey, else gray."""
    if loc == "LAD": return "grey"
    if loc == "LCX": return "grey"
    return "gray"

def stenosis_marker(sten_val):
    """0%→circle; ~45%→square; ~60%→triangle; else 'x'."""
    if pd.isna(sten_val): return "x"
    if sten_val < 0.10: return "o"
    if 0.40 <= sten_val <= 0.50: return "s"
    if 0.55 <= sten_val <= 0.65: return "^"
    return "x"

# Stenosis Group column
if "Stenosis Percentage" in df.columns:
    df["Stenosis Group"] = df["Stenosis Percentage"].apply(stenosis_group)
else:
    df["Stenosis Group"] = np.nan


# =============================================================================
# PLOTTING (kept intact; includes overlay_fn hook)
# =============================================================================
def make_smart_scatter(
        data, x_col, y_col, color_col,
        x_label, y_label, title,
        cmap_name="BuPu",
        custom_boundaries=None,
        color_label="",
        add_threshold=None,
        alpha_scatter=1,
        s_scatter=60,
        connect_stenosis_groups=False,
        show_singletons=True,
        savefig=False,
        dpi=600,
        dir="images",
        labels=False,
        location_col="Location",
        location_filter=None,
        figsize=(8.5, 5),
        external_data_source=None,
        source_col="source",
        internal_source_name="mine",
        external_alpha=0.4,
        overlay_fn=None,
):
    plt.figure(figsize=figsize)
    df_plot = data.copy()

    # Optional filter by location
    if location_filter is not None and location_col in df_plot.columns:
        if isinstance(location_filter, (list, tuple, set)):
            df_plot = df_plot[df_plot[location_col].isin(location_filter)]
        else:
            df_plot = df_plot[df_plot[location_col] == location_filter]

    # Always require x/y
    df_plot = df_plot[df_plot[x_col].notna() & df_plot[y_col].notna()]

    # External overlay mode (kept)
    external_mode = external_data_source is not None
    if external_mode:
        if source_col not in df_plot.columns:
            raise ValueError(f"external_data_source was provided, but '{source_col}' is not a column.")
        ext_sources = [external_data_source] if isinstance(external_data_source, str) else list(external_data_source)
        df_plot = df_plot[df_plot[source_col].isin([internal_source_name] + ext_sources)]
        df_internal = df_plot[df_plot[source_col] == internal_source_name].copy()
        df_external = df_plot[df_plot[source_col].isin(ext_sources)].copy()

        df_internal = df_internal[df_internal[color_col].notna()]
        df_external_col   = df_external[df_external[color_col].notna()].copy()
        df_external_nocol = df_external[df_external[color_col].isna()].copy()

        df_for_color = df_internal if len(df_internal) > 0 else df_external_col
    else:
        df_plot = df_plot[df_plot[color_col].notna()]
        df_for_color = df_plot

    if len(df_for_color) == 0:
        print(f"{title}: 0 points (after filtering).")
        plt.close()
        return

    # Singleton filtering
    if not show_singletons:
        if external_mode:
            if ("Stenosis Group" in df_internal.columns) and ("Length" in df_internal.columns):
                counts = df_internal.groupby(["Stenosis Group", "Length"]).size()
                valid_groups = counts[counts > 1].index
                df_internal = (df_internal.set_index(["Stenosis Group", "Length"])
                               .loc[valid_groups].reset_index())
        else:
            counts = df_plot.groupby(["Stenosis Group", "Length"]).size()
            valid_groups = counts[counts > 1].index
            df_plot = (df_plot.set_index(["Stenosis Group", "Length"])
                       .loc[valid_groups].reset_index())

    # Color scaling
    if custom_boundaries is None:
        cmin = df_for_color[color_col].min()
        cmax = df_for_color[color_col].max()
        custom_boundaries = np.linspace(cmin, cmax, 6)

    norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
    cmap = plt.get_cmap(cmap_name)

    first_scatter = None

    # External overlay
    if external_mode:
        if len(df_external_col) > 0:
            plt.scatter(df_external_col[x_col], df_external_col[y_col],
                        c=df_external_col[color_col], cmap=cmap, norm=norm,
                        edgecolor="gray", linewidths=1.0,
                        alpha=external_alpha, s=s_scatter, marker="o", zorder=1)
        if len(df_external_nocol) > 0:
            plt.scatter(df_external_nocol[x_col], df_external_nocol[y_col],
                        color="lightgray", edgecolor="gray", linewidths=1.0,
                        alpha=external_alpha, s=s_scatter, marker="o", zorder=1)

        groups = (df_internal.groupby(["Stenosis Group", "Length", location_col])
                  if (("Stenosis Group" in df_internal.columns) and ("Length" in df_internal.columns)
                      and (location_col in df_internal.columns))
                  else [])
    else:
        groups = (df_plot.groupby(["Stenosis Group", "Length", location_col])
                  if location_col in df_plot.columns else [])

    # Grouped/internal points
    for (sten_val, length_val, loc), gdf in groups:
        if pd.isna(sten_val) or pd.isna(length_val):
            continue

        marker_style = "o" if external_mode else stenosis_marker(sten_val)
        linestyle = snap_length_style(length_val, tol=0.15)
        line_color = vessel_line_color(loc)

        sc = plt.scatter(
            gdf[x_col], gdf[y_col],
            c=gdf[color_col], cmap=cmap, norm=norm,
            edgecolor="black", alpha=alpha_scatter, s=s_scatter,
            marker=marker_style, zorder=3
        )

        if labels and ("Geometry Number" in gdf.columns):
            for xi, yi, lab in zip(gdf[x_col], gdf[y_col], gdf["Geometry Number"]):
                plt.text(xi, yi, str(lab), fontsize=8, ha="right", va="top",
                         path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

        if first_scatter is None:
            first_scatter = sc

        if connect_stenosis_groups and (len(gdf) > 1):
            gdf_sorted = gdf.sort_values(by=x_col)
            plt.plot(gdf_sorted[x_col], gdf_sorted[y_col],
                     linestyle=linestyle, color=line_color,
                     alpha=0.8, linewidth=2.0, zorder=2)

    # Colorbar
    if first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        if color_label:
            cbar.set_label(color_label)

    # Threshold lines
    if add_threshold:
        for tdict in add_threshold:
            axis_type = tdict.get("axis", "y")
            value = tdict.get("value", 0.0)
            style = tdict.get("style", "--")
            c = tdict.get("color", "gray")
            w = tdict.get("width", 2.5)
            if axis_type == "y":
                plt.axhline(y=value, color=c, linestyle=style, linewidth=w)
            else:
                plt.axvline(x=value, color=c, linestyle=style, linewidth=w)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(False)
    plt.tight_layout()

    # Overlay
    if overlay_fn is not None:
        overlay_fn(plt.gca(), df_plot)

    # Save
    if savefig:
        fname = str(savefig) if savefig is not True else f"{y_col}_vs_{x_col}_col_{color_col}"
        plt.savefig(f"{dir}/{fname}.png", dpi=dpi, transparent=True, bbox_inches="tight")
        plt.savefig(f"{dir}/{fname}.svg", transparent=True, bbox_inches="tight")
        print(f"saved → {dir}/{fname}.png/.svg")

    plt.show()
    plt.close()


# =============================================================================
# GLOBAL BREAKPOINT CORE (kept; trimmed to essentials)
# =============================================================================
def _make_series_id(df_in, series_cols):
    df_out = df_in.copy()
    for c in series_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out["series_id"] = df_out[series_cols].astype(str).agg(" | ".join, axis=1)
    return df_out

def _within_demean(arr, groups):
    s = pd.Series(arr)
    g = pd.Series(groups)
    means = s.groupby(g).transform("mean").to_numpy()
    return arr - means

def _ols_rss(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid.T @ resid)
    return beta, rss

def _fit_fe_piecewise_rss_F(df_in, x_col, y_col, group_col, b):
    x = df_in[x_col].to_numpy()
    y = df_in[y_col].to_numpy()
    g = df_in[group_col].to_numpy()

    hinge = np.clip(x - b, 0.0, None)

    # within transform removes alpha_g
    y_t = _within_demean(y, g)
    x_t = _within_demean(x, g)
    h_t = _within_demean(hinge, g)

    # Null: y ~ beta*x
    X0 = x_t[:, None]
    beta0, rss0 = _ols_rss(y_t, X0)

    # Alt: y ~ beta1*x + beta2*hinge
    X1 = np.column_stack([x_t, h_t])
    beta1, rss1 = _ols_rss(y_t, X1)

    n = len(y)
    G = pd.Series(g).nunique()
    k0, k1 = 1, 2
    df_den = n - G - k1
    if df_den <= 0:
        return rss0, rss1, np.nan, beta0, beta1

    num = (rss0 - rss1) / (k1 - k0)
    den = rss1 / df_den
    F = num / den if den > 0 else np.nan
    return rss0, rss1, F, beta0, beta1

def _candidate_breakpoints(x, trim_q=0.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.array([])

    lo = np.quantile(x, trim_q)
    hi = np.quantile(x, 1 - trim_q)

    xu = np.unique(np.sort(x))
    mids = 0.5 * (xu[:-1] + xu[1:])
    mids = mids[(mids > lo) & (mids < hi)]
    return mids

def estimate_global_breakpoint(df_in, x_col, y_col, group_col="series_id", trim_q=0.0):
    x = df_in[x_col].to_numpy()
    cands = _candidate_breakpoints(x, trim_q=trim_q)
    if cands.size == 0:
        raise RuntimeError("No candidate breakpoints found (too few unique x).")

    best = {"b_hat": None, "supF": -np.inf, "rss0": None, "rss1": None,
            "beta0": None, "beta1": None}

    for b in cands:
        rss0, rss1, F, beta0, beta1 = _fit_fe_piecewise_rss_F(df_in, x_col, y_col, group_col, b)
        if np.isfinite(F) and F > best["supF"]:
            best.update({"b_hat": float(b), "supF": float(F),
                         "rss0": float(rss0), "rss1": float(rss1),
                         "beta0": beta0.copy(), "beta1": beta1.copy()})

    # --- ADD THIS GUARD (prevents supF=-inf poisoning the bootstrap) ---
    if best["b_hat"] is None or (not np.isfinite(best["supF"])):
        best.update({
            "b_hat": np.nan,
            "supF": np.nan,
            "rss0": np.nan,
            "rss1": np.nan,
            "beta0": None,
            "beta1": None,
        })

    return best

def _fit_null_fe(df_in, x_col, y_col, group_col="series_id"):
    x = df_in[x_col].to_numpy()
    y = df_in[y_col].to_numpy()
    g = df_in[group_col].to_numpy()

    # within slope beta
    y_t = _within_demean(y, g)
    x_t = _within_demean(x, g)
    beta, _ = _ols_rss(y_t, x_t[:, None])
    beta = float(beta[0])

    # alpha_g from group means
    df_tmp = df_in[[group_col, x_col, y_col]].copy()
    mu_y = df_tmp.groupby(group_col)[y_col].mean()
    mu_x = df_tmp.groupby(group_col)[x_col].mean()
    alpha = mu_y - beta * mu_x

    # residuals
    yhat = alpha.loc[df_tmp[group_col]].to_numpy() + beta * x
    resid = y - yhat
    return beta, alpha, resid

def bootstrap_supF_pvalue(df_in, x_col, y_col, group_col="series_id",
                          trim_q=0.0, n_boot=999, seed=0):
    rng = np.random.default_rng(seed)

    # observed
    obs = estimate_global_breakpoint(df_in, x_col, y_col, group_col=group_col, trim_q=trim_q)
    supF_obs = obs["supF"]
    b_obs = obs["b_hat"]

    # null fit
    beta_hat, alpha_hat, resid = _fit_null_fe(df_in, x_col, y_col, group_col=group_col)

    df_boot = df_in[[group_col, x_col, y_col]].copy()
    df_boot["resid"] = resid
    groups = df_boot[group_col].to_numpy()
    unique_groups = pd.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in unique_groups}

    supF_star = np.empty(n_boot, float)

    x = df_boot[x_col].to_numpy()
    alpha_vec = alpha_hat.loc[df_boot[group_col]].to_numpy()
    yhat_null = alpha_vec + beta_hat * x

    for r in range(n_boot):
        e_star = np.empty_like(resid)
        for g in unique_groups:
            idx = idx_by_g[g]
            e_g = resid[idx]
            w_g = rng.choice([-1.0, 1.0], size=len(idx), replace=True)  # Rademacher
            e_star[idx] = e_g * w_g

        y_star = yhat_null + e_star

        df_r = df_in.copy()
        df_r[y_col] = y_star

        est_r = estimate_global_breakpoint(df_r, x_col, y_col, group_col=group_col, trim_q=trim_q)
        F_r = est_r.get("supF", np.nan)
        supF_star[r] = float(F_r) if np.isfinite(F_r) else np.nan

    if not np.isfinite(supF_obs):
        raise RuntimeError(
            "Observed supF is not finite (scan failed). "
            "Try ensuring enough unique x values or relaxing filtering."
        )

    valid = np.isfinite(supF_star)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        raise RuntimeError(
            "All bootstrap draws produced non-finite supF. "
            "Your scan is degenerate under resampling."
        )

    if n_valid < 0.8 * n_boot:
        print(f"WARNING: only {n_valid}/{n_boot} bootstrap draws had finite supF. "
              "p-value may be unreliable; consider wild bootstrap.")

    p = (1.0 + np.sum(supF_star[valid] >= supF_obs)) / (n_valid + 1.0)

    return {
        "p_value": float(p),
        "supF_obs": float(supF_obs),
        "b_obs": float(b_obs) if b_obs is not None else np.nan,
        "obs_detail": obs,
        "n_boot_valid": int(n_valid),
    }

def _fit_alt_fe(df_in, x_col, y_col, group_col, b, beta1=None, beta2=None):
    x = df_in[x_col].to_numpy()
    y = df_in[y_col].to_numpy()
    g = df_in[group_col].to_numpy()
    hinge = np.clip(x - b, 0.0, None)

    # If betas not provided, estimate via within transformation
    if beta1 is None or beta2 is None:
        y_t = _within_demean(y, g)
        x_t = _within_demean(x, g)
        h_t = _within_demean(hinge, g)
        betas, _ = _ols_rss(y_t, np.column_stack([x_t, h_t]))
        beta1 = float(betas[0])
        beta2 = float(betas[1])

    df_tmp = df_in[[group_col, x_col, y_col]].copy()
    df_tmp["_hinge"] = hinge
    mu_y = df_tmp.groupby(group_col)[y_col].mean()
    mu_x = df_tmp.groupby(group_col)[x_col].mean()
    mu_h = df_tmp.groupby(group_col)["_hinge"].mean()
    alpha = mu_y - beta1 * mu_x - beta2 * mu_h

    yhat = alpha.loc[df_tmp[group_col]].to_numpy() + beta1 * x + beta2 * hinge
    resid = y - yhat
    return beta1, beta2, alpha, resid, yhat

def bootstrap_breakpoint_ci(df_in, x_col, y_col, group_col="series_id",
                            trim_q=0.0, n_boot=999, seed=0):
    rng = np.random.default_rng(seed)

    obs = estimate_global_breakpoint(df_in, x_col, y_col, group_col=group_col, trim_q=trim_q)
    b_obs = obs["b_hat"]
    if b_obs is None or not np.isfinite(b_obs):
        raise RuntimeError("Could not estimate observed breakpoint for CI bootstrap.")

    beta1_obs = float(obs["beta1"][0])
    beta2_obs = float(obs["beta1"][1])

    beta1_obs, beta2_obs, _, resid_alt, yhat_alt = _fit_alt_fe(
        df_in, x_col, y_col, group_col, b_obs, beta1=beta1_obs, beta2=beta2_obs
    )

    groups = df_in[group_col].to_numpy()
    unique_groups = pd.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in unique_groups}

    b_star = np.empty(n_boot, float)

    for r in range(n_boot):
        e_star = np.empty_like(resid_alt)
        for g in unique_groups:
            idx = idx_by_g[g]
            e_star[idx] = rng.choice(resid_alt[idx], size=len(idx), replace=True)

        y_star = yhat_alt + e_star
        df_r = df_in.copy()
        df_r[y_col] = y_star

        est_r = estimate_global_breakpoint(df_r, x_col, y_col, group_col=group_col, trim_q=trim_q)
        b_star[r] = est_r["b_hat"] if est_r["b_hat"] is not None else np.nan

    return {"b_obs": float(b_obs), "b_boot": b_star, "obs_detail": obs}

def add_centered_response(df_in, x_col, y_col, group_col, b_hat, beta1, beta2):
    dfp = df_in.copy()
    hinge = np.clip(dfp[x_col].to_numpy() - b_hat, 0.0, None)
    dfp["_hinge"] = hinge

    mu_y = dfp.groupby(group_col)[y_col].mean()
    mu_x = dfp.groupby(group_col)[x_col].mean()
    mu_h = dfp.groupby(group_col)["_hinge"].mean()
    alpha = mu_y - beta1 * mu_x - beta2 * mu_h

    dfp["y_centered"] = dfp[y_col] - alpha.loc[dfp[group_col]].to_numpy()
    return dfp


# =============================================================================
# RUN ANALYSIS
# =============================================================================
def main():
    d = df.copy()

    # Location filter
    if location_filter is not None and "Location" in d.columns:
        if isinstance(location_filter, (list, tuple, set)):
            d = d[d["Location"].isin(location_filter)]
        else:
            d = d[d["Location"] == location_filter]

    # KEEP stenosis filter
    if STENOSIS_MIN is not None and "Stenosis Percentage" in d.columns:
        d = d[d["Stenosis Percentage"].notna() & (d["Stenosis Percentage"] >= STENOSIS_MIN)]

    # Require X/Y
    if X_COL not in d.columns:
        raise KeyError(f"Missing x column: {X_COL}")
    if Y_COL not in d.columns:
        raise KeyError(f"Missing y column: {Y_COL} (did you construct FFR from Condition/P_d/P_a?)")
    d = d[d[X_COL].notna() & d[Y_COL].notna()].copy()

    # Build series IDs
    d = _make_series_id(d, SERIES_COLS)

    # Drop series with too few points
    counts = d.groupby("series_id").size()
    keep_series = counts[counts >= MIN_POINTS_PER_SERIES].index
    d = d[d["series_id"].isin(keep_series)].copy()

    if len(d) < 10:
        raise RuntimeError("Too few points after filtering to run breakpoint detection.")

    # 1) Breakpoint + NULL bootstrap p-value
    boot_null = bootstrap_supF_pvalue(
        d, x_col=X_COL, y_col=Y_COL, group_col="series_id",
        trim_q=0.0, n_boot=N_BOOT, seed=BOOT_SEED
    )
    b_hat = boot_null["b_obs"]
    pval = boot_null["p_value"]

    # 2) ALT bootstrap 95% CI for breakpoint
    boot_ci = bootstrap_breakpoint_ci(
        d, x_col=X_COL, y_col=Y_COL, group_col="series_id",
        trim_q=0.0, n_boot=N_BOOT, seed=BOOT_SEED + 1000
    )
    b_boot = np.asarray(boot_ci["b_boot"], float)
    b_boot = b_boot[np.isfinite(b_boot)]
    if b_boot.size < 20:
        b_ci = (np.nan, np.nan)
    else:
        b_ci = (float(np.quantile(b_boot, 0.025)), float(np.quantile(b_boot, 0.975)))

    # Print ONLY what you asked for
    print(f"b_hat: {b_hat:.6g}")
    print(f"p_boot (H0: no breakpoint): {pval:.6g}")
    print(f"95% CI for b_hat: [{b_ci[0]:.6g}, {b_ci[1]:.6g}]")

    # --- Plot (kept as before; uses within-betas at b_hat for overlay) ---
    obs = boot_null["obs_detail"]
    beta1 = float(obs["beta1"][0])
    beta2 = float(obs["beta1"][1])

    d_plot = add_centered_response(d, X_COL, Y_COL, "series_id", b_hat, beta1, beta2)

    # Ensure color column exists for plotting
    if COLOR_COL not in d_plot.columns:
        if "Stenosis Percentage" in d_plot.columns:
            color_col_plot = "Stenosis Percentage"
        else:
            d_plot["__color"] = 0.0
            color_col_plot = "__color"
    else:
        color_col_plot = COLOR_COL

    # Boundaries for colorbar
    df_for_bounds = d_plot[d_plot[color_col_plot].notna()]
    if len(df_for_bounds) == 0:
        custom_bounds = np.linspace(0, 1, 6)
    else:
        cmin, cmax = df_for_bounds[color_col_plot].min(), df_for_bounds[color_col_plot].max()
        custom_bounds = np.linspace(cmin, cmax, 6)

    def _overlay(ax, _df_plot):
        xg = np.linspace(_df_plot[X_COL].min(), _df_plot[X_COL].max(), 200)
        hg = np.clip(xg - b_hat, 0.0, None)
        yg = beta1 * xg + beta2 * hg  # centered-space fit

        ax.plot(xg, yg, color="black", linewidth=3.0, zorder=10)
        ax.axvline(b_hat, color="black", linestyle="--", linewidth=2.0, zorder=9)

        ax.text(
            0.02, 0.02,
            f"b̂={b_hat:.3g}\np={pval:.2g}\nCI=[{b_ci[0]:.3g}, {b_ci[1]:.3g}]",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    make_smart_scatter(
        data=d_plot[d_plot["y_centered"].notna() & d_plot[X_COL].notna()],
        x_col=X_COL,
        y_col="y_centered",
        color_col=color_col_plot,
        x_label=X_COL,
        y_label=f"{Y_COL} (centered by series)",
        title=f"Global breakpoint in {Y_COL} vs {X_COL} (pooled FE segmented regression)",
        cmap_name=COLOR_CMAP,
        custom_boundaries=custom_bounds,
        color_label=color_col_plot,
        alpha_scatter=0.9,
        s_scatter=80,
        connect_stenosis_groups=True,
        show_singletons=True,
        location_filter=location_filter,
        figsize=(10.5, 6),
        overlay_fn=_overlay,
        savefig=f"global_breakpoint_{Y_COL}_vs_{X_COL}"
    )


if __name__ == "__main__":
    main()