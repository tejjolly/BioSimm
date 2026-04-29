#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe

# =============================
# CONFIG YOU'LL EDIT
# =============================
# CSV_PATH = "../data/data_manuscript.csv"  # change me
CSV_PATH = "../data/break_test.csv"  # change me
Y_COL = "P_d/P_a"                         # FFR
X_COL = "HMR"                             # domain
SERIES_COLS = ["Condition", "Location", "Stenosis Percentage", "Length"]

MIN_POINTS_PER_SERIES = 3      # drop tiny series
H = 6                      # SaRa bandwidth in "global x-sorted index" units
EDGE_GUARD = 1                # if None, uses H; else integer
N_PERM = 100                 # for p-value
SEED = 0

MIN_SEFF = 3

# =============================
# STENOSIS FILTER OPTIONS
# =============================
DROP_NO_STENOSIS = False          # if True, drop stenosis < STENOSIS_MIN from analysis
STENOSIS_MIN = 0.05              # e.g. 0.05 = 5%

# Plotting can include/exclude even if analysis excludes
INCLUDE_NO_STENOSIS_PLOT = True  # if False, plot also drops stenosis < STENOSIS_MIN

# =============================
# HELPERS
# =============================

def _to_numeric_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def fe_slope(window_df, x_col, y_col, id_col):
    """
    Fixed-effects slope in a single window:
    demean within id_col, then OLS slope = sum(x*y)/sum(x^2)
    Returns (beta, contributing_ids_set)
    """
    if window_df.empty:
        return np.nan, set()

    g = window_df.groupby(id_col, sort=False)

    # series with at least 2 points in this window contribute to slope robustly
    counts = g.size()
    valid_ids = set(counts.index[counts >= 2])
    if not valid_ids:
        return np.nan, set()

    w = window_df[window_df[id_col].isin(valid_ids)].copy()

    # within-series demean (in this window)
    w["_x_dm"] = w[x_col] - w.groupby(id_col)[x_col].transform("mean")
    w["_y_dm"] = w[y_col] - w.groupby(id_col)[y_col].transform("mean")

    denom = np.sum(w["_x_dm"].to_numpy() ** 2)
    if denom <= 0:
        return np.nan, set()

    beta = np.sum(w["_x_dm"].to_numpy() * w["_y_dm"].to_numpy()) / denom
    return float(beta), valid_ids

def scan_common_break_by_threshold(df, x_col, y_col, id_col, min_side_points=2):
    """
    Common-break scan using candidate thresholds in x-units (not pooled index windows).

    For each candidate split between unique x values:
      left  = {x <= b}
      right = {x  > b}

    Compute FE slope on each side using only series that have >=2 points on BOTH sides.
    Statistic: D(b) = sqrt(S_eff) * (beta_R - beta_L)

    Returns best b_hat (x-units), sup|D|, and side slopes.
    """
    d = df[[id_col, x_col, y_col]].dropna().copy()
    d = d.sort_values(x_col, kind="mergesort")

    x_unique = np.unique(d[x_col].to_numpy())
    if len(x_unique) < 4:
        raise ValueError("Need at least ~4 unique x values to scan a breakpoint.")

    # candidate thresholds between adjacent unique x values
    b_cands = 0.5 * (x_unique[:-1] + x_unique[1:])

    best = {
        "b_hat": np.nan,
        "sup_absD": -np.inf,
        "D": np.nan,
        "beta_L": np.nan,
        "beta_R": np.nan,
        "Seff": 0,
    }

    # precompute per-series counts on each side efficiently per candidate is harder;
    # given your n is small, do it straightforwardly.
    for b in b_cands:
        left = d[d[x_col] <= b]
        right = d[d[x_col] > b]

        # series with enough points on each side
        nL = left.groupby(id_col).size()
        nR = right.groupby(id_col).size()
        ids = set(nL.index[nL >= min_side_points]).intersection(set(nR.index[nR >= min_side_points]))
        s_eff = len(ids)
        if s_eff < 2:
            continue

        bL, _ = fe_slope(left[left[id_col].isin(ids)], x_col, y_col, id_col)
        bR, _ = fe_slope(right[right[id_col].isin(ids)], x_col, y_col, id_col)
        if not np.isfinite(bL) or not np.isfinite(bR):
            continue

        D = np.sqrt(s_eff) * (bR - bL)
        if np.abs(D) > best["sup_absD"]:
            best.update({
                "b_hat": float(b),
                "sup_absD": float(np.abs(D)),
                "D": float(D),
                "beta_L": float(bL),
                "beta_R": float(bR),
                "Seff": int(s_eff),
            })

    if not np.isfinite(best["sup_absD"]) or best["sup_absD"] < 0:
        raise RuntimeError(
            "All D(b) invalid. Likely no series has >=2 points on BOTH sides for any split. "
            "Try min_side_points=1, or you need more x-support per series."
        )

    return best
def fit_global_fe_linear(df, x_col, y_col, id_col):
    """
    Null model: y_it = mu_i + beta * x_it + e_it
    Fit beta via within transform on full data.
    Returns beta, residuals (aligned to df rows).
    """
    d = df[[id_col, x_col, y_col]].dropna().copy()
    d["_x_dm"] = d[x_col] - d.groupby(id_col)[x_col].transform("mean")
    d["_y_dm"] = d[y_col] - d.groupby(id_col)[y_col].transform("mean")

    denom = np.sum(d["_x_dm"].to_numpy() ** 2)
    if denom <= 0:
        raise ValueError("Degenerate x after demeaning; cannot fit global FE slope.")

    beta = np.sum(d["_x_dm"].to_numpy() * d["_y_dm"].to_numpy()) / denom

    # residuals in original y space:
    # y = mu_i + beta x + e => e = (y - ybar_i) - beta (x - xbar_i)
    d["_e"] = d["_y_dm"] - beta * d["_x_dm"]
    return float(beta), d

import numpy as np
import pandas as pd

def prep_threshold_scan(df, x_col, y_col, id_col, min_side_points=2):
    d = df[[id_col, x_col, y_col]].dropna().copy()
    d = d.sort_values(x_col, kind="mergesort").reset_index(drop=True)

    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)
    sid_cat = pd.Categorical(d[id_col])
    sid = sid_cat.codes.astype(int)
    n_ids = len(sid_cat.categories)

    x_unique = np.unique(x)
    if len(x_unique) < 4:
        raise ValueError("Need >=4 unique x values.")
    b_cands = 0.5 * (x_unique[:-1] + x_unique[1:])

    # Precompute masks and eligible ids for each b (depends only on x and sid)
    masks_left = []
    elig_ids = []
    for b in b_cands:
        left = (x <= b)
        right = ~left
        cntL = np.bincount(sid[left], minlength=n_ids)
        cntR = np.bincount(sid[right], minlength=n_ids)
        elig = (cntL >= min_side_points) & (cntR >= min_side_points)
        masks_left.append(left)
        elig_ids.append(elig)

    return {"d": d, "x": x, "y": y, "sid": sid, "n_ids": n_ids,
            "b_cands": b_cands, "masks_left": masks_left, "elig_ids": elig_ids}

def fe_slope_and_var_on_side(x, y, sid, elig, side_mask):
    """
    FE-within slope and variance on a side, restricted to eligible series.
    Returns (beta, var_beta). Uses classic OLS variance in demeaned space.
    """
    use = side_mask & elig[sid]
    if not np.any(use):
        return np.nan, np.nan

    x_u = x[use]
    y_u = y[use]
    sid_u = sid[use]
    n_ids = elig.size

    cnt = np.bincount(sid_u, minlength=n_ids).astype(float)
    sx  = np.bincount(sid_u, weights=x_u, minlength=n_ids)
    sy  = np.bincount(sid_u, weights=y_u, minlength=n_ids)

    present = cnt > 0
    mx = np.zeros(n_ids); my = np.zeros(n_ids)
    mx[present] = sx[present] / cnt[present]
    my[present] = sy[present] / cnt[present]

    x_dm = x_u - mx[sid_u]
    y_dm = y_u - my[sid_u]

    Sxx = np.sum(x_dm * x_dm)
    if Sxx <= 0:
        return np.nan, np.nan

    beta = np.sum(x_dm * y_dm) / Sxx
    resid = y_dm - beta * x_dm
    n = resid.size

    # df adjustment: crude but stable; FE uses many intercepts, but n is not huge here
    Seff = int(np.sum(elig))  # series contributing at this candidate split
    df = max(n - Seff - 1, 1)
    s2 = np.sum(resid * resid) / df
    var_beta = s2 / Sxx
    return float(beta), float(var_beta)

def supT_common_break(prep, y_override=None):
    x = prep["x"]
    y = prep["y"] if y_override is None else y_override
    sid = prep["sid"]
    b_cands = prep["b_cands"]
    masks_left = prep["masks_left"]
    elig_ids = prep["elig_ids"]

    best = {"b_hat": np.nan, "sup_absT": -np.inf,
            "beta_L": np.nan, "beta_R": np.nan,
            "t": np.nan, "Seff": 0}

    for b, left_mask, elig in zip(b_cands, masks_left, elig_ids):
        Seff = int(np.sum(elig))
        if Seff < MIN_SEFF:
            continue

        bL, vL = fe_slope_and_var_on_side(x, y, sid, elig, left_mask)
        bR, vR = fe_slope_and_var_on_side(x, y, sid, elig, ~left_mask)
        if not (np.isfinite(bL) and np.isfinite(bR) and np.isfinite(vL) and np.isfinite(vR)):
            continue

        denom = np.sqrt(vL + vR)
        if denom <= 0:
            continue

        t = (bR - bL) / denom
        at = abs(t)
        if at > best["sup_absT"]:
            best.update({
                "b_hat": float(b),
                "sup_absT": float(at),
                "beta_L": float(bL),
                "beta_R": float(bR),
                "t": float(t),
                "Seff": Seff,
            })

    if best["sup_absT"] < 0 or not np.isfinite(best["sup_absT"]):
        raise RuntimeError("No valid split produced a finite statistic.")
    return best

def fit_fe_null(prep):
    """
    Fit FE null y = mu_i + beta x + e using within transform on full data.
    Returns beta0 and within residuals e0 aligned to prep order.
    """
    x = prep["x"]; y = prep["y"]; sid = prep["sid"]; n_ids = prep["n_ids"]

    cnt = np.bincount(sid, minlength=n_ids).astype(float)
    sx  = np.bincount(sid, weights=x, minlength=n_ids)
    sy  = np.bincount(sid, weights=y, minlength=n_ids)
    xbar = sx / cnt
    ybar = sy / cnt

    x_dm = x - xbar[sid]
    y_dm = y - ybar[sid]

    Sxx = np.sum(x_dm * x_dm)
    beta0 = np.sum(x_dm * y_dm) / Sxx
    e0 = y_dm - beta0 * x_dm
    return float(beta0), e0, ybar, xbar

def wild_bootstrap_pvalue(prep, n_boot=2000, seed=0, progress_every=200):
    """
    Wild bootstrap p-value for common-break existence using sup|t|.
    """
    rng = np.random.default_rng(seed)

    # observed
    obs = supT_common_break(prep)["sup_absT"]

    beta0, e0, ybar, xbar = fit_fe_null(prep)
    x = prep["x"]; sid = prep["sid"]; n_ids = prep["n_ids"]

    # precompute within x
    x_dm = x - xbar[sid]

    boot_stats = np.empty(n_boot, dtype=float)

    # Rademacher weights
    for b in range(n_boot):
        v = rng.choice([-1.0, 1.0], size=e0.size)
        e_star = e0 * v

        # y_dm* = beta0 x_dm + e_star ; y* = ybar + y_dm*
        y_dm_star = beta0 * x_dm + e_star
        y_star = ybar[sid] + y_dm_star

        boot_stats[b] = supT_common_break(prep, y_override=y_star)["sup_absT"]

        if progress_every and (b + 1) % progress_every == 0:
            print(f"  boot {b+1}/{n_boot}", flush=True)

    p = (1.0 + np.sum(boot_stats >= obs)) / (1.0 + n_boot)
    return float(obs), float(p), boot_stats

def _stenosis_filter(df, sten_col="Stenosis Percentage", min_sten=0.05):
    if sten_col not in df.columns:
        raise KeyError(f"CSV missing '{sten_col}' needed for stenosis filtering.")
    return df[df[sten_col].notna() & (df[sten_col] >= float(min_sten))].copy()

def snap_length_style(val, tol=0.15):
    if pd.isna(val):
        return "solid"
    if np.isclose(val, 1.2, atol=tol):
        return ":"
    if np.isclose(val, 2.5, atol=tol):
        return "--"
    return "solid"

def vessel_line_color(loc):
    if loc in ("LAD", "LCX"):
        return "grey"
    return "gray"

def stenosis_marker(sten_val):
    if pd.isna(sten_val): return "x"
    if sten_val < 0.10: return "o"
    if 0.40 <= sten_val <= 0.50: return "s"
    if 0.55 <= sten_val <= 0.65: return "^"
    return "x"

def build_midpoints_df(df, *, x_col, y_col, id_col):
    """
    Midpoints between adjacent points *within each series* (sorted by x).
    Returns columns: hmr_mid, y_mid
    """
    d = df[[id_col, x_col, y_col]].dropna().copy()
    rows = []
    for sid, g in d.groupby(id_col, sort=False):
        g = g.sort_values(x_col).reset_index(drop=True)
        if len(g) < 2:
            continue
        x = g[x_col].to_numpy(float)
        y = g[y_col].to_numpy(float)
        for i in range(len(g) - 1):
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            x_mid = float(np.sqrt(x0 * x1)) if (x0 > 0 and x1 > 0) else float(0.5 * (x0 + x1))
            y_mid = float(0.5 * (y0 + y1))
            rows.append({"hmr_mid": x_mid, "y_mid": y_mid})
    return pd.DataFrame(rows)

def plot_common_break(
    df_plot,
    *,
    x_col,
    y_col,
    cutoff,
    include_midpoints=True,
    figsize=(8.5, 5),
    savefig=False,
    outdir="images",
    fname="common_break_plot",
    labels=False,
    color_col=None,
    cmap_name="BuPu",
    custom_boundaries=None,
):
    """
    Scatter points + connect per-series + red hollow midpoints + vertical dashed cutoff.
    If color_col is provided, points are colored by that column (like your ruptures script).
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=figsize)

    d = df_plot.copy()
    d = d[d[x_col].notna() & d[y_col].notna()].copy()

    # Optional color mapping (if you want HSR coloring etc.)
    use_color = (color_col is not None) and (color_col in d.columns) and d[color_col].notna().any()
    if use_color:
        d = d[d[color_col].notna()].copy()
        if custom_boundaries is None:
            cmin, cmax = float(d[color_col].min()), float(d[color_col].max())
            custom_boundaries = np.linspace(cmin, cmax, 6)
        norm = colors.BoundaryNorm(custom_boundaries, ncolors=256, clip=True)
        cmap = plt.get_cmap(cmap_name)
    else:
        norm = None
        cmap = None

    first_scatter = None

    # If these exist, mimic your series grouping style; else fall back to series_id only.
    group_cols = []
    for c in ["Stenosis Percentage", "Length", "Location", "series_id"]:
        if c in d.columns:
            group_cols.append(c)
    if "series_id" not in group_cols and "series_id" in d.columns:
        group_cols.append("series_id")
    if not group_cols:
        group_cols = ["series_id"]

    # We want to connect lines per series_id specifically
    if "series_id" not in d.columns:
        raise KeyError("plot_common_break expects a 'series_id' column.")

    # Scatter + connect per series_id; marker/linestyle from stenosis/length/location if available
    for sid, gdf in d.groupby("series_id", sort=False):
        gdf = gdf.sort_values(x_col)

        sten_val = gdf["Stenosis Percentage"].iloc[0] if "Stenosis Percentage" in gdf.columns else np.nan
        length_val = gdf["Length"].iloc[0] if "Length" in gdf.columns else np.nan
        loc = gdf["Location"].iloc[0] if "Location" in gdf.columns else None

        marker_style = stenosis_marker(sten_val)
        linestyle = snap_length_style(length_val, tol=0.15)
        line_color = vessel_line_color(loc) if loc is not None else "grey"

        if use_color:
            sc = plt.scatter(
                gdf[x_col], gdf[y_col],
                c=gdf[color_col], cmap=cmap, norm=norm,
                edgecolor="k", alpha=0.8, s=60,
                marker=marker_style, zorder=3
            )
        else:
            sc = plt.scatter(
                gdf[x_col], gdf[y_col],
                edgecolor="k", alpha=0.8, s=60,
                marker=marker_style, zorder=3
            )

        if first_scatter is None:
            first_scatter = sc

        if len(gdf) > 1:
            plt.plot(
                gdf[x_col], gdf[y_col],
                linestyle=linestyle, color=line_color,
                alpha=0.8, linewidth=2.0, zorder=1
            )

        if labels and ("Geometry Number" in gdf.columns):
            for xi, yi, lab in zip(gdf[x_col], gdf[y_col], gdf["Geometry Number"]):
                plt.text(
                    xi, yi, str(lab), fontsize=8, ha="right", va="top",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")]
                )

    # Colorbar (only if using color)
    if use_color and first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        cbar.set_label(color_col)

    # Midpoints
    if include_midpoints:
        mids = build_midpoints_df(d, x_col=x_col, y_col=y_col, id_col="series_id")
        if len(mids) > 0:
            plt.scatter(
                mids["hmr_mid"].to_numpy(float),
                mids["y_mid"].to_numpy(float),
                s=80,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=5
            )

    # Breakpoint
    if np.isfinite(cutoff):
        plt.axvline(x=float(cutoff), color="red", linestyle="--", linewidth=2.5, zorder=2)

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

# =============================
# MAIN
# =============================
def main():
    df = pd.read_csv(CSV_PATH)

    # coerce numerics used
    _to_numeric_inplace(df, [X_COL, Y_COL, "Stenosis Percentage", "Length", "Geometry Number"])

    # build series id
    missing = [c for c in SERIES_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing SERIES_COLS in CSV: {missing}")

    df = df.dropna(subset=SERIES_COLS + [X_COL, Y_COL]).copy()
    df["series_id"] = df[SERIES_COLS].astype(str).agg("|".join, axis=1)

    # ---- Optional: drop low/no stenosis (ANALYSIS) ----
    df_plot_base = df.copy()  # keep a plotting base before analysis filtering

    if DROP_NO_STENOSIS:
        df = _stenosis_filter(df, sten_col="Stenosis Percentage", min_sten=STENOSIS_MIN)

    # drop tiny series (AFTER stenosis filtering, because filtering changes counts)
    counts = df.groupby("series_id").size()
    keep_ids = set(counts.index[counts >= MIN_POINTS_PER_SERIES])
    df = df[df["series_id"].isin(keep_ids)].copy()

    # ---- Build plotting dataframe (can include no-stenosis if you want) ----
    df_plot = df_plot_base.copy()
    if not INCLUDE_NO_STENOSIS_PLOT:
        df_plot = _stenosis_filter(df_plot, sten_col="Stenosis Percentage", min_sten=STENOSIS_MIN)

    # keep only series that survived the MIN_POINTS filter (so lines aren’t misleading)
    df_plot = df_plot[df_plot["series_id"].isin(keep_ids)].copy()

    print("==============================")
    print("DATA SUMMARY")
    print("==============================")
    print("n rows:", len(df))
    print("n series:", df["series_id"].nunique())
    print(df.groupby("series_id").size().describe())

    scan_D = scan_common_break_by_threshold(
        df,
        x_col=X_COL,
        y_col=Y_COL,
        id_col="series_id",
        min_side_points=2,  # start here; if still fails, set to 1
    )
    scan = scan_D

    print("\n==============================")
    print("COMMON BREAK (threshold scan)")
    print("==============================")
    print(f"b_hat ({X_COL} units): {scan['b_hat']:.6g}")
    print(f"sup|D|: {scan['sup_absD']:.6g}")
    print(f"Seff (series contributing): {scan['Seff']}")
    print(f"beta_L: {scan['beta_L']:.6g}")
    print(f"beta_R: {scan['beta_R']:.6g}")
    print(f"delta_beta: {(scan['beta_R'] - scan['beta_L']):.6g}")

    prep = prep_threshold_scan(df, x_col=X_COL, y_col=Y_COL, id_col="series_id", min_side_points=2)

    scan_t = supT_common_break(prep)
    scan = scan_t
    print("\n==============================")
    print("COMMON BREAK (sup|t| threshold scan)")
    print("==============================")
    print(f"b_hat ({X_COL} units): {scan['b_hat']:.6g}")
    print(f"sup|t|: {scan['sup_absT']:.6g}")
    print(f"Seff: {scan['Seff']}")
    print(f"beta_L: {scan['beta_L']:.6g}")
    print(f"beta_R: {scan['beta_R']:.6g}")
    print(f"delta_beta: {(scan['beta_R'] - scan['beta_L']):.6g}")

    # ---- Plot data + common breakpoint (match your ruptures-style overlays) ----
    plot_common_break(
        df_plot,
        x_col=X_COL,
        y_col=Y_COL,
        cutoff=scan_D["b_hat"],  # vertical red dashed
        include_midpoints=True,  # red hollow midpoints
        figsize=(8.5, 5),
        savefig=False,  # set True if desired
        outdir="images",
        fname=f"{Y_COL}_vs_{X_COL}_common_break",
        labels=False,
        color_col=None,  # set e.g. "HSR" if you want colored points
        cmap_name="BuPu",
        custom_boundaries=None,
    )

    plt.axvline(
        x=scan_t["b_hat"],
        color="darkred",
        linestyle=":",
        linewidth=2.5,
        label="sup|t| breakpoint",
        zorder=2,
    )

    obsT, p_boot, _ = wild_bootstrap_pvalue(prep, n_boot=N_PERM, seed=SEED, progress_every=200)

    print("\n==============================")
    print("SIGNIFICANCE (wild bootstrap under FE no-break null)")
    print("==============================")
    print(f"Observed sup|t|: {obsT:.6g}")
    print(f"Bootstrap p-value: {p_boot:.6g}")
if __name__ == "__main__":
    main()