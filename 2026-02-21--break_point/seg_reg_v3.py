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
# CSV_PATH = "../data/break_test.csv"        # change me
CSV_PATH = "../data/data_manuscript.csv"        # change me
Y_COL = "CFR"                         # FFR
X_COL = "HMR"                             # domain
SERIES_COLS = ["Condition", "Location", "Stenosis Percentage", "Length"]

MIN_POINTS_PER_SERIES = 2     # drop tiny series
MIN_SIDE_POINTS = 0            # require >= this many points on BOTH sides (per series) to "count"
MIN_SEFF = 4                   # require >= this many eligible series at a candidate split

# =============================
# STENOSIS FILTER OPTIONS
# =============================
if Y_COL == "FFR":
    DROP_NO_STENOSIS = True
    Y_COL = 'P_d/P_a'
elif Y_COL == "HSR":
    DROP_NO_STENOSIS = True          # if True, drop stenosis < STENOSIS_MIN from analysis
else:
    DROP_NO_STENOSIS = False          # if True, drop stenosis < STENOSIS_MIN from analysis

STENOSIS_MIN = 0.05               # e.g. 0.05 = 5%

# Plotting can include/exclude even if analysis excludes
INCLUDE_NO_STENOSIS_PLOT = True   # if False, plot also drops stenosis < STENOSIS_MIN

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
    import os
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
        raise KeyError("plot_common_break expects a 'series_id' column.")

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
# CORE: FE segmented regression supF
# =============================
def prep_supF(df, x_col, y_col, id_col, min_side_points=2):
    d = df[[id_col, x_col, y_col]].dropna().copy()
    d = d.sort_values(x_col, kind="mergesort").reset_index(drop=True)

    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)

    sid_cat = pd.Categorical(d[id_col])
    sid = sid_cat.codes.astype(int)
    n_ids = len(sid_cat.categories)

    # counts + within transform (series FE)
    cnt = np.bincount(sid, minlength=n_ids).astype(float)
    sx  = np.bincount(sid, weights=x, minlength=n_ids)
    sy  = np.bincount(sid, weights=y, minlength=n_ids)
    xbar = sx / cnt
    ybar = sy / cnt
    x_dm = x - xbar[sid]
    y_dm = y - ybar[sid]

    # candidate thresholds between adjacent unique x
    x_unique = np.unique(x)
    if len(x_unique) < 4:
        raise ValueError("Need >=4 unique x values to scan a breakpoint.")
    b_cands = 0.5 * (x_unique[:-1] + x_unique[1:])

    # eligibility masks depend only on x + sid
    elig_list = []
    use_list = []
    for b in b_cands:
        left = (x <= b)
        right = ~left
        cntL = np.bincount(sid[left], minlength=n_ids)
        cntR = np.bincount(sid[right], minlength=n_ids)
        elig = (cntL >= min_side_points) & (cntR >= min_side_points)
        elig_list.append(elig)
        use_list.append(elig[sid])  # keep all obs from eligible series

    return {
        "d": d, "x": x, "y": y,
        "sid": sid, "n_ids": n_ids,
        "cnt": cnt,
        "x_dm": x_dm, "y_dm": y_dm,
        "b_cands": b_cands,
        "elig_list": elig_list,
        "use_list": use_list,
    }

def supF_common_break(prep, y_dm_override=None):
    x = prep["x"]
    sid = prep["sid"]
    cnt = prep["cnt"]
    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"] if y_dm_override is None else y_dm_override
    b_cands = prep["b_cands"]
    elig_list = prep["elig_list"]
    use_list = prep["use_list"]
    n_ids = prep["n_ids"]

    best = {
        "b_hat": np.nan,
        "supF": -np.inf,
        "Seff": 0,
        "beta_pre": np.nan,
        "beta_post": np.nan,
        "delta_slope": np.nan,
    }

    for b, elig, use in zip(b_cands, elig_list, use_list):
        Seff = int(np.sum(elig))
        if Seff < MIN_SEFF:
            continue

        xu = x_dm[use]
        yu = y_dm[use]
        sid_u = sid[use]
        n_use = yu.size

        # df for FE model: n - (#series intercepts) - (#slopes)
        df2 = n_use - Seff - 2
        if df2 <= 1:
            continue

        # Null: yu ~ beta * xu
        Sxx = float(np.sum(xu * xu))
        if Sxx <= 0:
            continue
        beta0 = float(np.sum(xu * yu) / Sxx)
        r0 = yu - beta0 * xu
        SSR0 = float(np.sum(r0 * r0))

        # Hinge term h = (x - b)_+, demean within series
        h = np.maximum(0.0, x[use] - float(b))
        sh = np.bincount(sid_u, weights=h, minlength=n_ids)
        hbar = sh / cnt  # means for all series (only elig ones matter)
        h_dm = h - hbar[sid_u]

        # Alt: yu ~ b1*xu + b2*h_dm  (2 regressors, no intercept)
        S11 = float(np.sum(xu * xu))
        S22 = float(np.sum(h_dm * h_dm))
        S12 = float(np.sum(xu * h_dm))
        det = S11 * S22 - S12 * S12
        if det <= 1e-14:
            continue

        Sy1 = float(np.sum(xu * yu))
        Sy2 = float(np.sum(h_dm * yu))
        b1 = (Sy1 * S22 - Sy2 * S12) / det
        b2 = (Sy2 * S11 - Sy1 * S12) / det

        r1 = yu - b1 * xu - b2 * h_dm
        SSR1 = float(np.sum(r1 * r1))

        # F-test for gamma=b2=0 at THIS b
        num = (SSR0 - SSR1) / 1.0
        den = SSR1 / float(df2)
        if den <= 0:
            continue
        F = num / den

        if F > best["supF"]:
            best.update({
                "b_hat": float(b),
                "supF": float(F),
                "Seff": Seff,
                "beta_pre": float(b1),
                "beta_post": float(b1 + b2),
                "delta_slope": float(b2),
            })

    if not np.isfinite(best["supF"]) or best["supF"] < 0:
        raise RuntimeError("No valid candidate split produced a finite supF.")
    return best

import itertools
import numpy as np

def exact_cluster_sign_supF_pvalue(prep):
    # observed
    obs_res = supF_common_break(prep)
    obs = obs_res["supF"]

    # null fit on full demeaned data
    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"]
    sid  = prep["sid"]
    n_ids = prep["n_ids"]

    beta0 = float(np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm))
    e0 = y_dm - beta0 * x_dm

    stats = []
    for signs in itertools.product([-1.0, 1.0], repeat=n_ids):
        g = np.asarray(signs, dtype=float)
        y_star = beta0 * x_dm + e0 * g[sid]
        stats.append(supF_common_break(prep, y_dm_override=y_star)["supF"])

    stats = np.asarray(stats)
    # p = np.mean(stats >= obs)  # exact (discrete) p-value
    p = (np.sum(stats >= obs) + 1.0) / (stats.size + 1.0)
    return obs_res, float(p), stats
# =============================
# MAIN
# =============================
def main():
    df = pd.read_csv(CSV_PATH)

    _to_numeric_inplace(df, [X_COL, Y_COL, "Stenosis Percentage", "Length", "Geometry Number"])

    missing = [c for c in SERIES_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing SERIES_COLS in CSV: {missing}")

    df = df.dropna(subset=SERIES_COLS + [X_COL, Y_COL]).copy()
    df["series_id"] = df[SERIES_COLS].astype(str).agg("|".join, axis=1)

    # ---- keep a plotting base before analysis filtering ----
    df_plot_base = df.copy()

    # ---- Optional: drop low/no stenosis (ANALYSIS) ----
    if DROP_NO_STENOSIS:
        df = _stenosis_filter(df, sten_col="Stenosis Percentage", min_sten=STENOSIS_MIN)

    # ---- drop tiny series (after any analysis filtering) ----
    counts = df.groupby("series_id").size()
    keep_ids = set(counts.index[counts >= MIN_POINTS_PER_SERIES])
    df = df[df["series_id"].isin(keep_ids)].copy()

    # ---- plotting df (can include no-stenosis if desired) ----
    df_plot = df_plot_base.copy()
    if not INCLUDE_NO_STENOSIS_PLOT:
        df_plot = _stenosis_filter(df_plot, sten_col="Stenosis Percentage", min_sten=STENOSIS_MIN)
    df_plot = df_plot[df_plot["series_id"].isin(keep_ids)].copy()

    prep = prep_supF(df, x_col=X_COL, y_col=Y_COL, id_col="series_id", min_side_points=MIN_SIDE_POINTS)

    res, p_val, _ = exact_cluster_sign_supF_pvalue(prep)

    print("\n==============================")
    print("COMMON BREAK (FE segmented regression; supF)")
    print("==============================")
    print(f"b_hat ({X_COL} units): {res['b_hat']:.6g}")
    print(f"p-value: {p_val:.6g}\n")

    print(f"supF: {res['supF']:.6g}")
    print(f"Seff (eligible series): {res['Seff']}")
    print(f"pre-slope (beta_pre): {res['beta_pre']:.6g}")
    print(f"post-slope (beta_post): {res['beta_post']:.6g}")
    print(f"delta_slope (post-pre): {res['delta_slope']:.6g}")

    print("==============================")
    print("DATA SUMMARY")
    print("==============================")
    print("n rows:", len(df))
    print("n series:", df["series_id"].nunique())
    print(df.groupby("series_id").size().describe())

    # ---- Plot ----
    plot_common_break(
        df_plot,
        x_col=X_COL,
        y_col=Y_COL,
        cutoff=res["b_hat"],
        include_midpoints=True,
        figsize=(8.5, 5),
        savefig=False,
        outdir="images",
        fname=f"{Y_COL}_vs_{X_COL}_supF_break",
        labels=False,
        color_col='HSR',
        cmap_name="BuPu",
        custom_boundaries=None,
    )

if __name__ == "__main__":
    main()