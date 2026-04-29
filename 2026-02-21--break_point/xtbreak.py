#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
xtbreak-like common-break panel test over an ordered index (HMR as "time").

Core differences vs your current code:
- Breakpoints are estimated by argmin SSR (Bai–Perron) using dynamic programming.
- No continuity (no hinge). Coefficients are regime-specific as in xtbreak.
- Trimming is defined over the index grid length T (unique HMR values), not per-series side counts.
- Supports s = 0..S_MAX breaks and sequential F(s+1|s) tests.

Model (breaking regressors only, X empty for additivity / DP):
  y_{i,t} = alpha_i + w_{i,t}' delta_j + e_{i,t},  t in regime j
where:
  i = series_id
  t = ordered HMR index (common grid)
  w = BREAKING_COLS (default: [HMR])  -> "effect of HMR changes across regimes"
Fixed effects alpha_i are removed by within-demeaning.

Inference:
  cluster sign-flip ("wild bootstrap") at the series level.

Note:
- If you want non-breaking regressors X (like election dummy in the slides),
  DP additivity breaks; you need a more complex partialling-out approach.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe

# =============================
# CONFIG YOU'LL EDIT
# =============================
CSV_PATH = "../data/data_manuscript.csv"  # change me

Y_COL = "FFR"
INDEX_COL = "HMR"  # "time" analogue (ordered index)

if Y_COL == "FFR":
    Y_COL = "P_d/P_a"
if INDEX_COL == "FFR":
    INDEX_COL = "P_d/P_a"

SERIES_COLS = ["Condition", "Location", "Stenosis Percentage", "Length"]

# breaking regressors w_it (their coefficients change by regime)
# Default: allow slope in HMR to change across regimes (xtbreak example: CCI slope changes).
BREAKING_COLS = [INDEX_COL]

# ===== Break settings (xtbreak-like) =====
S_MAX = 4          # max breaks to consider
TRIM_EPS = 0.3   # trimming epsilon; segment length h = max(2, ceil(eps*T))
ALPHA = 0.05       # sequential test level

# bootstrap (series-level sign flips)
USE_EXACT_SIGNFLIP_IF_POSSIBLE = True
N_BOOT = 999  # used if exact not feasible

# =============================
# STENOSIS FILTER OPTIONS (keep your logic)
# =============================
DROP_NO_STENOSIS = True
STENOSIS_MIN = 0.05
INCLUDE_NO_STENOSIS_PLOT = True

MIN_POINTS_PER_SERIES = 2  # drop tiny series

# =============================
# HELPERS
# =============================
def _to_numeric_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

def plot_breaks(
    df_plot,
    *,
    x_col,
    y_col,
    cutoffs,
    include_midpoints=True,
    figsize=(8.5, 5),
    savefig=False,
    outdir="images",
    fname="xtbreak_like_plot",
    labels=False,
    color_col=None,
    cmap_name="BuPu",
    custom_boundaries=None,
):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=figsize)

    d = df_plot.copy()
    d = d[d[x_col].notna() & d[y_col].notna()].copy()

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

    if "series_id" not in d.columns:
        raise KeyError("plot_breaks expects a 'series_id' column.")

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

    if use_color and first_scatter is not None:
        cbar = plt.colorbar(first_scatter, ticks=custom_boundaries)
        cbar.set_label(color_col)

    if include_midpoints:
        mids = build_midpoints_df(d, x_col=x_col, y_col=y_col, id_col="series_id")
        if len(mids) > 0:
            plt.scatter(
                mids["hmr_mid"].to_numpy(float),
                mids["y_mid"].to_numpy(float),
                s=80, marker="o",
                facecolors="none", edgecolors="red",
                linewidths=2, zorder=5
            )

    for c in cutoffs:
        if np.isfinite(c):
            plt.axvline(x=float(c), color="red", linestyle="--", linewidth=2.5, zorder=2)

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
# CORE: xtbreak-like DP estimation and tests
# =============================
def make_panel_arrays(df, *, id_col, y_col, index_col, w_cols):
    """
    Build panel arrays with INDEX_COL as the ordered "time" index, and W made from w_cols.
    IMPORTANT: avoid duplicating index_col inside W when w_cols includes it.
    """
    # de-dupe w_cols while preserving order
    w_cols = list(w_cols)
    seen = set()
    w_cols_u = []
    for c in w_cols:
        if c not in seen:
            w_cols_u.append(c)
            seen.add(c)

    # now build dataframe without duplicating columns
    cols = [id_col, y_col, index_col] + w_cols_u
    # de-dupe full cols list too (extra safety)
    seen = set()
    cols_u = []
    for c in cols:
        if c not in seen:
            cols_u.append(c)
            seen.add(c)

    d = df[cols_u].dropna().copy()

    # series codes
    sid_cat = pd.Categorical(d[id_col])
    sid = sid_cat.codes.astype(int)
    n_ids = len(sid_cat.categories)

    # ordered "time" grid (unique index values)
    grid = np.unique(np.sort(d[index_col].to_numpy(float)))
    T = len(grid)
    if T < 3:
        raise ValueError("Need at least 3 unique INDEX_COL values to test breaks.")

    # map each obs to time index
    x = d[index_col].to_numpy(float)
    t_idx = np.searchsorted(grid, x)

    # arrays
    y = d[y_col].to_numpy(float)

    # W is only from w_cols_u (q = len(w_cols_u))
    W = np.column_stack([d[c].to_numpy(float) for c in w_cols_u])
    q = W.shape[1]

    # within-demean (FE)
    cnt = np.bincount(sid, minlength=n_ids).astype(float)
    if np.any(cnt <= 0):
        raise RuntimeError("Empty series encountered after filtering.")

    sy = np.bincount(sid, weights=y, minlength=n_ids)
    ybar = sy / cnt
    y_dm = y - ybar[sid]

    W_dm = np.empty_like(W)
    for j in range(q):
        sw = np.bincount(sid, weights=W[:, j], minlength=n_ids)
        wbar = sw / cnt
        W_dm[:, j] = W[:, j] - wbar[sid]

    return {
        "d": d,
        "sid": sid,
        "n_ids": n_ids,
        "grid": grid,
        "T": T,
        "t_idx": t_idx,
        "y_dm": y_dm,
        "W_dm": W_dm,
        "q": q,
        "cnt": cnt,
    }

def compute_h(trim_eps, T):
    # xtbreak: h = eps * T (as periods). Enforce >=2 periods for identifiability in tiny T.
    return max(2, int(np.ceil(trim_eps * T)))

def precompute_segments(t_idx, W_dm, *, T, h, min_obs, ridge=0.0):
    """
    Precompute per-segment objects for fast SSR(y) evaluation across bootstraps.
    Segment is defined on time periods [a..b] inclusive (0-index).
    """
    q = W_dm.shape[1]
    seg = {}  # (a,b) -> dict(idx, P, valid)
    for a in range(T):
        for b in range(a, T):
            if (b - a + 1) < h:
                continue
            idx = np.where((t_idx >= a) & (t_idx <= b))[0]
            if idx.size < min_obs:
                continue
            X = W_dm[idx, :]  # n x q
            XtX = X.T @ X
            if ridge > 0:
                XtX = XtX + ridge * np.eye(q)
            # check invertibility / rank
            if np.linalg.matrix_rank(XtX) < q:
                continue
            invXtX = np.linalg.inv(XtX)
            P = invXtX @ X.T  # q x n  (beta = P @ y_seg)
            seg[(a, b)] = {"idx": idx, "P": P, "X": X, "q": q}
    return seg

def ssr_for_segment(seg_obj, y_dm):
    idx = seg_obj["idx"]
    X = seg_obj["X"]
    P = seg_obj["P"]
    y = y_dm[idx]
    beta = P @ y
    r = y - X @ beta
    return float(r @ r)

def dp_min_ssr(seg_dict, *, y_dm, T, h, s):
    """
    Dynamic programming to find breakpoints (s breaks => s+1 segments) that minimize SSR.
    Returns (SSR_min, breaks), where breaks are time indices k in 1..T-1
    meaning a break between k-1 and k (0-index: segment ends at k-1, next starts at k).
    """
    r = s + 1  # number of segments
    INF = np.inf
    # feasibility cap for s
    max_breaks_possible = (T // h) - 1
    if max_breaks_possible < 0:
        max_breaks_possible = 0
    if s > max_breaks_possible:
        raise ValueError(f"s={s} breaks infeasible for T={T}, h={h}. Max is {max_breaks_possible}.")

    # dp[segcount][t_end] where t_end is inclusive index in 0..T-1 for end of last segment
    dp = np.full((r + 1, T), INF, dtype=float)
    prev = np.full((r + 1, T), -1, dtype=int)

    # base: 1 segment from 0..t
    for t in range(T):
        obj = seg_dict.get((0, t), None)
        if obj is None:
            continue
        dp[1, t] = ssr_for_segment(obj, y_dm)
        prev[1, t] = -1

    # transitions
    for segcount in range(2, r + 1):
        # last segment must have >= h periods, so its start <= t-h+1
        for t in range(T):
            # need room for segcount segments: minimal end index is segcount*h - 1
            if t < segcount * h - 1:
                continue
            best_val = INF
            best_k = -1
            # previous end k must be at least (segcount-1)*h - 1
            k_min = (segcount - 1) * h - 1
            k_max = t - h
            for k in range(k_min, k_max + 1):
                if not np.isfinite(dp[segcount - 1, k]):
                    continue
                obj = seg_dict.get((k + 1, t), None)
                if obj is None:
                    continue
                val = dp[segcount - 1, k] + ssr_for_segment(obj, y_dm)
                if val < best_val:
                    best_val = val
                    best_k = k
            if best_k >= 0:
                dp[segcount, t] = best_val
                prev[segcount, t] = best_k

    if not np.isfinite(dp[r, T - 1]):
        raise RuntimeError("DP failed to find a feasible segmentation. Relax trimming or min_obs constraints.")

    # backtrack breaks
    breaks = []
    t = T - 1
    for segcount in range(r, 1, -1):
        k = prev[segcount, t]
        if k < 0:
            raise RuntimeError("Backtrack failed unexpectedly.")
        breaks.append(k + 1)  # break starts at k+1
        t = k
    breaks = sorted(breaks)
    return float(dp[r, T - 1]), breaks

def fit_piecewise(seg_dict, *, y_dm, W_dm, t_idx, breaks, T):
    """
    Fit piecewise coefficients delta_j per segment given breaks (time indices).
    Returns y_hat (demeaned space) and list of betas per segment.
    """
    cuts = [0] + breaks + [T]
    betas = []
    y_hat = np.zeros_like(y_dm)

    for j in range(len(cuts) - 1):
        a = cuts[j]
        b = cuts[j + 1] - 1
        obj = seg_dict.get((a, b), None)
        if obj is None:
            raise RuntimeError(f"Missing segment object for [{a},{b}].")
        idx = obj["idx"]
        X = obj["X"]
        P = obj["P"]
        y = y_dm[idx]
        beta = P @ y
        betas.append(beta)
        y_hat[idx] = X @ beta

    return y_hat, betas

def ssr_for_breaks(seg_dict, *, y_dm, breaks, T):
    cuts = [0] + breaks + [T]
    ssr = 0.0
    for j in range(len(cuts) - 1):
        a = cuts[j]
        b = cuts[j + 1] - 1
        obj = seg_dict.get((a, b), None)
        if obj is None:
            raise RuntimeError(f"Missing segment object for [{a},{b}].")
        ssr += ssr_for_segment(obj, y_dm)
    return float(ssr)

def supF_from_ssr(SSR0, SSRs, *, s, q, n_obs, n_ids):
    # restrictions = s * q (each break adds a new delta relative to previous)
    num_df = s * q
    den_df = n_obs - n_ids - (s + 1) * q
    if den_df <= 0:
        return np.nan
    num = (SSR0 - SSRs) / float(num_df)
    den = SSRs / float(den_df)
    if den <= 0:
        return np.nan
    return float(num / den)

def sequential_F_splus1_given_s(seg_dict, *, y_dm, breaks_s, SSR_s, T, h, q, n_obs, n_ids):
    """
    Implements slide-style:
      F(s+1|s) = max over segments j of (max over tau in that segment of F(tau | T_hat_s))
    Here we:
      - keep the s breaks fixed
      - try inserting one additional break tau within each segment
      - compute the best F improvement
    """
    den_df = n_obs - n_ids - (len(breaks_s) + 2) * q  # s+1 breaks => (s+2) segments
    if den_df <= 0:
        return np.nan, None

    cuts = [0] + breaks_s + [T]
    best_F = -np.inf
    best_tau = None

    for j in range(len(cuts) - 1):
        a = cuts[j]
        b = cuts[j + 1]
        # candidate tau must leave >=h periods on both sides inside [a, b)
        for tau in range(a + h, b - h + 1):
            # new breaks = breaks_s plus tau
            new_breaks = sorted(breaks_s + [tau])
            SSR_new = ssr_for_breaks(seg_dict, y_dm=y_dm, breaks=new_breaks, T=T)
            num = (SSR_s - SSR_new) / float(q)
            den = SSR_new / float(den_df)
            if den <= 0:
                continue
            F = float(num / den)
            if F > best_F:
                best_F = F
                best_tau = tau

    if not np.isfinite(best_F):
        return np.nan, None
    return best_F, best_tau

def make_cutoffs_from_breaks(grid, breaks):
    # break at time index k means split between grid[k-1] and grid[k]
    cutoffs = []
    for k in breaks:
        xL = float(grid[k - 1])
        xR = float(grid[k])
        cutoffs.append(0.5 * (xL + xR))
    return cutoffs

def bootstrap_sequential_pvalue(panel, seg_dict, *, s, breaks_obs, SSR_s_obs, F_obs, h, exact_ok=True):
    """
    Cluster sign-flip bootstrap p-value for F(s+1|s).
    Null: s breaks (unknown), with breakpoints re-estimated each bootstrap draw.

    Steps per draw:
      1) Fit s-break model on y*? For bootstrap under null, we:
         - fit s-break on original y_dm, obtain y_hat + residual e
         - sign flip residual by series to get y_star = y_hat + e_star
      2) Re-estimate s breaks on y_star via DP (unknown breaks).
      3) Compute F(s+1|s) on y_star using its own estimated breaks.
    """
    sid = panel["sid"]
    n_ids = panel["n_ids"]
    T = panel["T"]
    q = panel["q"]
    n_obs = panel["y_dm"].size
    y_dm = panel["y_dm"]

    # fit null (s breaks) on original
    SSR_s, breaks_s = dp_min_ssr(seg_dict, y_dm=y_dm, T=T, h=h, s=s)
    y_hat, _ = fit_piecewise(seg_dict, y_dm=y_dm, W_dm=panel["W_dm"], t_idx=panel["t_idx"], breaks=breaks_s, T=T)
    e = y_dm - y_hat

    # choose sign draws
    if exact_ok and USE_EXACT_SIGNFLIP_IF_POSSIBLE and (2 ** n_ids <= N_BOOT):
        all_signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n_ids)), dtype=float)
    elif exact_ok and USE_EXACT_SIGNFLIP_IF_POSSIBLE and (2 ** n_ids <= 1024):
        all_signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n_ids)), dtype=float)
    else:
        rng = np.random.default_rng(0)
        all_signs = rng.choice([-1.0, 1.0], size=(N_BOOT, n_ids), replace=True).astype(float)

    stats = []
    for g in all_signs:
        y_star = y_hat + e * g[sid]

        # re-estimate s breaks under null on y_star
        SSR_star_s, breaks_star_s = dp_min_ssr(seg_dict, y_dm=y_star, T=T, h=h, s=s)
        # compute sequential stat on y_star
        F_star, _ = sequential_F_splus1_given_s(
            seg_dict,
            y_dm=y_star,
            breaks_s=breaks_star_s,
            SSR_s=SSR_star_s,
            T=T, h=h, q=q,
            n_obs=n_obs, n_ids=n_ids
        )
        if np.isfinite(F_star):
            stats.append(F_star)

    stats = np.asarray(stats, dtype=float)
    p = (np.sum(stats >= F_obs) + 1.0) / (stats.size + 1.0)
    return float(p), stats

# =============================
# MAIN
# =============================
def main():
    df = pd.read_csv(CSV_PATH)

    # numeric columns
    _to_numeric_inplace(df, [INDEX_COL, Y_COL, "Stenosis Percentage", "Length", "Geometry Number"] + BREAKING_COLS)

    # series id
    missing = [c for c in SERIES_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing SERIES_COLS in CSV: {missing}")

    df = df.dropna(subset=SERIES_COLS + [INDEX_COL, Y_COL]).copy()
    df["series_id"] = df[SERIES_COLS].astype(str).agg("|".join, axis=1)

    # plot base before analysis filtering
    df_plot_base = df.copy()

    # analysis filter: stenosis
    if DROP_NO_STENOSIS:
        df = _stenosis_filter(df, sten_col="Stenosis Percentage", min_sten=STENOSIS_MIN)

    # drop tiny series
    counts = df.groupby("series_id").size()
    keep_ids = set(counts.index[counts >= MIN_POINTS_PER_SERIES])
    df = df[df["series_id"].isin(keep_ids)].copy()

    # plotting df (optionally include no stenosis)
    df_plot = df_plot_base.copy()
    if not INCLUDE_NO_STENOSIS_PLOT:
        df_plot = _stenosis_filter(df_plot, sten_col="Stenosis Percentage", min_sten=STENOSIS_MIN)
    df_plot = df_plot[df_plot["series_id"].isin(keep_ids)].copy()

    # build panel arrays
    panel = make_panel_arrays(
        df,
        id_col="series_id",
        y_col=Y_COL,
        index_col=INDEX_COL,
        w_cols=BREAKING_COLS
    )

    T = panel["T"]
    n_obs = panel["y_dm"].size
    n_ids = panel["n_ids"]
    q = panel["q"]

    h = compute_h(TRIM_EPS, T)
    # need enough obs per segment to estimate q params
    min_obs = max(q + 1, 2 * q)

    seg_dict = precompute_segments(
        panel["t_idx"], panel["W_dm"],
        T=T, h=h, min_obs=min_obs
    )

    # effective feasible max breaks
    max_breaks_possible = max(0, (T // h) - 1)
    smax_eff = min(S_MAX, max_breaks_possible)

    print("\n==============================")
    print("xtbreak-like setup (HMR as index)")
    print("==============================")
    print(f"INDEX_COL (time analog): {INDEX_COL}")
    print(f"BREAKING_COLS (w): {BREAKING_COLS}  (q={q})")
    print(f"T (unique {INDEX_COL} grid): {T}")
    print(f"Trimming eps={TRIM_EPS} -> h={h} periods per segment")
    print(f"n_obs={n_obs}, n_series={n_ids}")
    print(f"S_MAX requested={S_MAX}, feasible smax={smax_eff}")

    # estimate SSR for s=0..smax
    SSR = {}
    breaks = {}
    for s in range(0, smax_eff + 1):
        SSR[s], breaks[s] = dp_min_ssr(seg_dict, y_dm=panel["y_dm"], T=T, h=h, s=s)

    SSR0 = SSR[0]

    print("\n==============================")
    print("supF(s) (unknown breaks; argmin SSR partitions)")
    print("==============================")
    for s in range(1, smax_eff + 1):
        supF_s = supF_from_ssr(SSR0, SSR[s], s=s, q=q, n_obs=n_obs, n_ids=n_ids)
        cutoffs = make_cutoffs_from_breaks(panel["grid"], breaks[s])
        print(f"s={s}: supF={supF_s:.6g}  SSR={SSR[s]:.6g}  breaks(t)={breaks[s]}  cutoffs({INDEX_COL})={[round(c,6) for c in cutoffs]}")

    # sequential determination of number of breaks
    print("\n==============================")
    print("Sequential test F(s+1|s) with cluster sign-flip p-values")
    print("==============================")
    s_hat = 0
    while s_hat < smax_eff:
        # observed stat using observed y with estimated breaks_s_hat
        F_obs, tau_obs = sequential_F_splus1_given_s(
            seg_dict,
            y_dm=panel["y_dm"],
            breaks_s=breaks[s_hat],
            SSR_s=SSR[s_hat],
            T=T, h=h, q=q,
            n_obs=n_obs, n_ids=n_ids
        )
        if not np.isfinite(F_obs):
            print(f"F({s_hat+1}|{s_hat}) not computable (df issues). Stop.")
            break

        p_boot, _ = bootstrap_sequential_pvalue(
            panel, seg_dict,
            s=s_hat,
            breaks_obs=breaks[s_hat],
            SSR_s_obs=SSR[s_hat],
            F_obs=F_obs,
            h=h,
            exact_ok=True
        )

        tau_val = None
        if tau_obs is not None:
            tau_val = 0.5 * (panel["grid"][tau_obs - 1] + panel["grid"][tau_obs])
        print(f"F({s_hat+1}|{s_hat}) = {F_obs:.6g}  p_boot={p_boot:.6g}  best_new_break_t={tau_obs}  cutoff({INDEX_COL})={tau_val}")

        if p_boot < ALPHA:
            s_hat += 1
        else:
            break

    print("\n==============================")
    print(f"Detected number of breaks (sequential @ alpha={ALPHA}): {s_hat}")
    print("==============================")
    cutoffs_hat = make_cutoffs_from_breaks(panel["grid"], breaks[s_hat])
    print(f"Breaks (time index): {breaks[s_hat]}")
    print(f"Break cutoffs in {INDEX_COL} units: {[float(c) for c in cutoffs_hat]}")
    print(f"SSR(s_hat)={SSR[s_hat]:.6g}")

    # ---- Plot ----
    plot_breaks(
        df_plot,
        x_col=INDEX_COL,
        y_col=Y_COL,
        cutoffs=cutoffs_hat,
        include_midpoints=True,
        figsize=(8.5, 5),
        savefig=False,
        outdir="images",
        fname=f"{Y_COL}_vs_{INDEX_COL}_xtbreak_like",
        labels=False,
        color_col=None,
        cmap_name="BuPu",
        custom_boundaries=None,
    )

if __name__ == "__main__":
    main()