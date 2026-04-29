#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import time

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

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
# ROBUSTNESS-CHECK CONFIG (A-E)
# =============================
ALPHA = 0.05
RUN_ROBUSTNESS_CHECKS = True

# A) Per-series support at b_hat
STRICT_MIN_SIDE_POINTS = 2

# B) Leave-one-series-out stability
LOSO_MAX_B_REL_SHIFT = 0.20
LOSO_MIN_SIG_FRAC_IF_BASE_SIG = 0.75

# C) supF profile shape
PROFILE_NEAR_MAX_FRAC = 0.95
PROFILE_SPIKE_GAP_RATIO_FAIL = 0.50

# D) Trimming / admissible-break sensitivity
TRIM_SHARES = (0.00, 0.20, 0.25)
TRIM_TARGET_MIN_SEFF = 6
TRIM_MAX_B_REL_SHIFT = 0.20
TRIM_MIN_SIG_FRAC_IF_BASE_SIG = 0.60

# E) Design-based simulation under null/alternative
RUN_SIMULATION_CHECK = True
SIM_N_REPS = 200
SIM_SEED = 20260227
SIM_PROGRESS_EVERY = 50
SIM_SIZE_TOL_PASS = 0.03
SIM_SIZE_TOL_FAIL = 0.10
SIM_POWER_MIN_PASS = 0.50
SIM_POWER_MIN_FAIL = 0.30

# Exact enumeration uses 2^G sign patterns.
MAX_EXACT_CLUSTERS = 12

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
def _status(level, label, detail):
    print(f"[{level.upper()}] {label}: {detail}")


def _short_sid(text, max_len=64):
    s = str(text)
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


def enumerate_cluster_signs(n_ids):
    if n_ids > MAX_EXACT_CLUSTERS:
        raise ValueError(
            f"Exact sign enumeration requires 2^G draws; G={n_ids} exceeds "
            f"MAX_EXACT_CLUSTERS={MAX_EXACT_CLUSTERS}."
        )
    return np.asarray(list(itertools.product([-1.0, 1.0], repeat=n_ids)), dtype=float)


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
        "sid_levels": np.asarray(sid_cat.categories).astype(str),
        "cnt": cnt,
        "x_dm": x_dm, "y_dm": y_dm,
        "b_cands": b_cands,
        "elig_list": elig_list,
        "use_list": use_list,
    }

def supF_common_break(
    prep,
    y_dm_override=None,
    *,
    min_seff_override=None,
    trim_share=0.0,
    return_profile=False,
):
    x = prep["x"]
    sid = prep["sid"]
    cnt = prep["cnt"]
    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"] if y_dm_override is None else y_dm_override
    b_cands = prep["b_cands"]
    elig_list = prep["elig_list"]
    use_list = prep["use_list"]
    n_ids = prep["n_ids"]

    min_seff = MIN_SEFF if min_seff_override is None else int(min_seff_override)
    trim_share = float(trim_share)
    n_obs = x.size

    best = {
        "b_hat": np.nan,
        "supF": -np.inf,
        "Seff": 0,
        "beta_pre": np.nan,
        "beta_post": np.nan,
        "delta_slope": np.nan,
    }
    profile_rows = []

    for b, elig, use in zip(b_cands, elig_list, use_list):
        n_left = int(np.sum(x <= b))
        n_right = int(n_obs - n_left)
        left_share = n_left / float(n_obs)
        right_share = n_right / float(n_obs)
        if min(left_share, right_share) < trim_share:
            continue

        Seff = int(np.sum(elig))
        if Seff < min_seff:
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

        profile_rows.append({
            "b": float(b),
            "F": float(F),
            "Seff": Seff,
            "n_use": int(n_use),
            "df2": int(df2),
            "left_share": float(left_share),
            "right_share": float(right_share),
            "beta_pre": float(b1),
            "beta_post": float(b1 + b2),
            "delta_slope": float(b2),
        })

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
        raise RuntimeError(
            "No valid candidate split produced a finite supF "
            f"(trim_share={trim_share}, min_seff={min_seff})."
        )
    if return_profile:
        profile_df = pd.DataFrame(profile_rows)
        if len(profile_df) > 0:
            profile_df = profile_df.sort_values("F", ascending=False).reset_index(drop=True)
        return best, profile_df
    return best


def demean_hinge(prep, b):
    x = prep["x"]
    sid = prep["sid"]
    cnt = prep["cnt"]
    n_ids = prep["n_ids"]

    h = np.maximum(0.0, x - float(b))
    sh = np.bincount(sid, weights=h, minlength=n_ids)
    hbar = sh / cnt
    return h - hbar[sid]


def exact_cluster_sign_supF_pvalue(
    prep,
    *,
    y_dm_data=None,
    min_seff_override=None,
    trim_share=0.0,
    sign_matrix=None,
):
    # observed
    y_dm = prep["y_dm"] if y_dm_data is None else np.asarray(y_dm_data, dtype=float)
    if y_dm.shape != prep["y_dm"].shape:
        raise ValueError("y_dm_data shape mismatch with prep['y_dm'].")

    obs_res = supF_common_break(
        prep,
        y_dm_override=y_dm,
        min_seff_override=min_seff_override,
        trim_share=trim_share,
    )
    obs = obs_res["supF"]

    # null fit on full demeaned data
    x_dm = prep["x_dm"]
    sid  = prep["sid"]
    n_ids = prep["n_ids"]
    if sign_matrix is None:
        sign_matrix = enumerate_cluster_signs(n_ids)

    beta0 = float(np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm))
    e0 = y_dm - beta0 * x_dm

    stats = np.empty(sign_matrix.shape[0], dtype=float)
    for i, g in enumerate(sign_matrix):
        y_star = beta0 * x_dm + e0 * g[sid]
        stats[i] = supF_common_break(
            prep,
            y_dm_override=y_star,
            min_seff_override=min_seff_override,
            trim_share=trim_share,
        )["supF"]

    p = (np.sum(stats >= obs) + 1.0) / (stats.size + 1.0)
    return obs_res, float(p), stats


def check_A_regime_support(prep, b_hat):
    print("\n------------------------------")
    print("CHECK A: REGIME SUPPORT AT b_hat")
    print("------------------------------")

    x = prep["x"]
    sid = prep["sid"]
    sid_levels = prep["sid_levels"]
    n_ids = prep["n_ids"]

    left = x <= b_hat
    cntL = np.bincount(sid[left], minlength=n_ids)
    cntR = np.bincount(sid[~left], minlength=n_ids)
    strict_ok = (cntL >= STRICT_MIN_SIDE_POINTS) & (cntR >= STRICT_MIN_SIDE_POINTS)

    out = pd.DataFrame({
        "series_idx": np.arange(n_ids, dtype=int),
        "series_id": [_short_sid(s) for s in sid_levels],
        "n_left": cntL.astype(int),
        "n_right": cntR.astype(int),
        "n_total": (cntL + cntR).astype(int),
        f"ge_{STRICT_MIN_SIDE_POINTS}_each_side": strict_ok,
    })
    print(out.to_string(index=False))

    strict_count = int(strict_ok.sum())
    if strict_count == n_ids:
        level = "pass"
        detail = f"All {n_ids}/{n_ids} series have >= {STRICT_MIN_SIDE_POINTS} points on each side."
    elif strict_count >= MIN_SEFF:
        level = "warn"
        detail = (
            f"{strict_count}/{n_ids} series have >= {STRICT_MIN_SIDE_POINTS} points on each side. "
            f"This still clears MIN_SEFF={MIN_SEFF}, but support is thin."
        )
    else:
        level = "fail"
        detail = (
            f"Only {strict_count}/{n_ids} series have >= {STRICT_MIN_SIDE_POINTS} points on each side; "
            f"below MIN_SEFF={MIN_SEFF}."
        )
    _status(level, "A", detail)
    return {"level": level, "table": out}


def check_B_leave_one_series_out(df, x_col, y_col, id_col, base_res, base_p):
    print("\n------------------------------")
    print("CHECK B: LEAVE-ONE-SERIES-OUT STABILITY")
    print("------------------------------")

    series_values = sorted(df[id_col].astype(str).unique())
    rows = []
    for drop_idx, sid_val in enumerate(series_values):
        dsub = df[df[id_col].astype(str) != sid_val].copy()
        row = {
            "drop_idx": drop_idx,
            "drop_series": _short_sid(sid_val),
            "success": False,
            "b_hat": np.nan,
            "p_value": np.nan,
            "delta_slope": np.nan,
            "error": "",
        }
        try:
            prep_sub = prep_supF(
                dsub,
                x_col=x_col,
                y_col=y_col,
                id_col=id_col,
                min_side_points=MIN_SIDE_POINTS,
            )
            sign_matrix = enumerate_cluster_signs(prep_sub["n_ids"])
            res_sub, p_sub, _ = exact_cluster_sign_supF_pvalue(
                prep_sub,
                sign_matrix=sign_matrix,
            )
            row.update({
                "success": True,
                "b_hat": float(res_sub["b_hat"]),
                "p_value": float(p_sub),
                "delta_slope": float(res_sub["delta_slope"]),
            })
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    loso = pd.DataFrame(rows)
    print(loso[["drop_idx", "success", "b_hat", "p_value", "delta_slope", "drop_series"]].to_string(index=False))

    success = loso["success"].to_numpy(bool)
    n_total = int(len(loso))
    n_success = int(np.sum(success))
    n_fail = n_total - n_success
    if n_success == 0:
        _status("fail", "B", "No successful leave-one-series-out fits.")
        return {"level": "fail", "table": loso}

    x_vals = df[x_col].to_numpy(float)
    x_range = float(np.max(x_vals) - np.min(x_vals))
    if x_range <= 0:
        x_range = 1.0

    sign_ref = np.sign(float(base_res["delta_slope"]))
    sign_obs = np.sign(loso.loc[success, "delta_slope"].to_numpy(float))
    if sign_ref == 0:
        sign_flip = int(np.sum(sign_obs != 0))
    else:
        sign_flip = int(np.sum((sign_obs != 0) & (sign_obs != sign_ref)))

    rel_shift = np.abs(loso.loc[success, "b_hat"].to_numpy(float) - float(base_res["b_hat"])) / x_range
    max_rel_shift = float(np.max(rel_shift)) if rel_shift.size else np.nan

    if base_p < ALPHA:
        sig_frac = float(np.mean(loso.loc[success, "p_value"].to_numpy(float) < ALPHA))
    else:
        sig_frac = np.nan

    min_success = int(np.ceil(0.75 * n_total))
    if n_success < min_success or sign_flip > 0:
        level = "fail"
    elif (
        n_fail > 0
        or max_rel_shift > LOSO_MAX_B_REL_SHIFT
        or (base_p < ALPHA and sig_frac < LOSO_MIN_SIG_FRAC_IF_BASE_SIG)
    ):
        level = "warn"
    else:
        level = "pass"

    detail = (
        f"success={n_success}/{n_total}, sign_flips={sign_flip}, "
        f"max_rel_b_shift={max_rel_shift:.3f}"
    )
    if base_p < ALPHA:
        detail += f", sig_frac={sig_frac:.3f} (alpha={ALPHA})"
    _status(level, "B", detail)
    return {"level": level, "table": loso}


def check_C_supF_profile(profile):
    print("\n------------------------------")
    print("CHECK C: supF PROFILE SHARPNESS")
    print("------------------------------")

    if len(profile) == 0:
        _status("fail", "C", "No valid profile entries.")
        return {"level": "fail", "table": profile}

    top_k = min(5, len(profile))
    print(profile.head(top_k)[["b", "F", "Seff", "left_share", "right_share"]].to_string(index=False))

    topF = float(profile.iloc[0]["F"])
    secondF = float(profile.iloc[1]["F"]) if len(profile) > 1 else np.nan
    near_mask = profile["F"].to_numpy(float) >= (PROFILE_NEAR_MAX_FRAC * topF)
    near_count = int(np.sum(near_mask))
    near_b = profile.loc[near_mask, "b"].to_numpy(float)
    near_span = float(np.max(near_b) - np.min(near_b)) if near_b.size > 1 else 0.0
    gap_ratio = ((topF - secondF) / topF) if (len(profile) > 1 and topF > 0) else np.nan

    if near_count >= 2:
        level = "pass"
    elif np.isfinite(gap_ratio) and gap_ratio > PROFILE_SPIKE_GAP_RATIO_FAIL:
        level = "fail"
    else:
        level = "warn"

    detail = (
        f"near_max_count={near_count}, near_max_span={near_span:.4g}, "
        f"topF={topF:.4g}, secondF={secondF:.4g}, gap_ratio={gap_ratio:.3f}"
    )
    _status(level, "C", detail)
    return {"level": level, "table": profile}


def check_D_trimming_sensitivity(prep, base_res, base_p):
    print("\n------------------------------")
    print("CHECK D: TRIMMING / ADMISSIBLE-REGION SENSITIVITY")
    print("------------------------------")

    n_ids = prep["n_ids"]
    strict_min_seff = min(n_ids, max(MIN_SEFF, TRIM_TARGET_MIN_SEFF))
    min_seff_grid = sorted(set([MIN_SEFF, strict_min_seff]))
    trim_grid = sorted(set(float(v) for v in TRIM_SHARES))
    sign_matrix = enumerate_cluster_signs(n_ids)

    rows = []
    for trim_share in trim_grid:
        for min_seff in min_seff_grid:
            row = {
                "trim_share": float(trim_share),
                "min_seff": int(min_seff),
                "success": False,
                "b_hat": np.nan,
                "p_value": np.nan,
                "delta_slope": np.nan,
                "error": "",
            }
            try:
                res_i, p_i, _ = exact_cluster_sign_supF_pvalue(
                    prep,
                    min_seff_override=min_seff,
                    trim_share=trim_share,
                    sign_matrix=sign_matrix,
                )
                row.update({
                    "success": True,
                    "b_hat": float(res_i["b_hat"]),
                    "p_value": float(p_i),
                    "delta_slope": float(res_i["delta_slope"]),
                })
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    grid = pd.DataFrame(rows)
    print(grid[["trim_share", "min_seff", "success", "b_hat", "p_value", "delta_slope"]].to_string(index=False))

    success = grid["success"].to_numpy(bool)
    n_total = int(len(grid))
    n_success = int(np.sum(success))
    if n_success == 0:
        _status("fail", "D", "No successful runs across trimming/min_seff grid.")
        return {"level": "fail", "table": grid}

    x = prep["x"]
    x_range = float(np.max(x) - np.min(x))
    if x_range <= 0:
        x_range = 1.0

    sign_ref = np.sign(float(base_res["delta_slope"]))
    sign_obs = np.sign(grid.loc[success, "delta_slope"].to_numpy(float))
    if sign_ref == 0:
        sign_flip = int(np.sum(sign_obs != 0))
    else:
        sign_flip = int(np.sum((sign_obs != 0) & (sign_obs != sign_ref)))
    rel_shift = np.abs(grid.loc[success, "b_hat"].to_numpy(float) - float(base_res["b_hat"])) / x_range
    max_rel_shift = float(np.max(rel_shift))
    if base_p < ALPHA:
        sig_frac = float(np.mean(grid.loc[success, "p_value"].to_numpy(float) < ALPHA))
    else:
        sig_frac = np.nan

    strict_mask = (
        (grid["trim_share"].to_numpy(float) >= 0.20)
        & (grid["min_seff"].to_numpy(int) >= strict_min_seff)
    )
    strict_total = int(np.sum(strict_mask))
    strict_success = int(np.sum(grid.loc[strict_mask, "success"].to_numpy(bool)))

    if strict_total > 0 and strict_success == 0:
        level = "fail"
    elif sign_flip > 0:
        level = "fail"
    elif (
        n_success < n_total
        or max_rel_shift > TRIM_MAX_B_REL_SHIFT
        or (base_p < ALPHA and sig_frac < TRIM_MIN_SIG_FRAC_IF_BASE_SIG)
    ):
        level = "warn"
    else:
        level = "pass"

    detail = (
        f"success={n_success}/{n_total}, strict_success={strict_success}/{strict_total}, "
        f"sign_flips={sign_flip}, max_rel_b_shift={max_rel_shift:.3f}"
    )
    if base_p < ALPHA:
        detail += f", sig_frac={sig_frac:.3f} (alpha={ALPHA})"
    _status(level, "D", detail)
    return {"level": level, "table": grid}


def check_E_design_simulation(prep, base_res):
    print("\n------------------------------")
    print("CHECK E: DESIGN-BASED SIZE/POWER SIMULATION")
    print("------------------------------")
    print(
        f"Running {SIM_N_REPS} reps under null and {SIM_N_REPS} reps under alternative "
        f"(seed={SIM_SEED})."
    )

    n_ids = prep["n_ids"]
    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"]
    sid = prep["sid"]
    sign_matrix = enumerate_cluster_signs(n_ids)

    beta0 = float(np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm))
    e0 = y_dm - beta0 * x_dm
    h_dm = demean_hinge(prep, base_res["b_hat"])
    beta_pre_alt = float(base_res["beta_pre"])
    delta_alt = float(base_res["delta_slope"])

    rng = np.random.default_rng(SIM_SEED)
    pvals_null = []
    pvals_alt = []
    fail_null = 0
    fail_alt = 0
    t0 = time.time()

    for rep in range(SIM_N_REPS):
        if SIM_PROGRESS_EVERY > 0 and ((rep + 1) % SIM_PROGRESS_EVERY == 0):
            print(f"  progress: {rep + 1}/{SIM_N_REPS}")

        g = rng.choice(np.asarray([-1.0, 1.0]), size=n_ids, replace=True)
        y_null = beta0 * x_dm + e0 * g[sid]
        y_alt = beta_pre_alt * x_dm + delta_alt * h_dm + e0 * g[sid]

        try:
            _, p_null, _ = exact_cluster_sign_supF_pvalue(
                prep,
                y_dm_data=y_null,
                sign_matrix=sign_matrix,
            )
            pvals_null.append(p_null)
        except Exception:
            fail_null += 1

        try:
            _, p_alt, _ = exact_cluster_sign_supF_pvalue(
                prep,
                y_dm_data=y_alt,
                sign_matrix=sign_matrix,
            )
            pvals_alt.append(p_alt)
        except Exception:
            fail_alt += 1

    pvals_null = np.asarray(pvals_null, dtype=float)
    pvals_alt = np.asarray(pvals_alt, dtype=float)

    size_est = float(np.mean(pvals_null < ALPHA)) if pvals_null.size else np.nan
    power_est = float(np.mean(pvals_alt < ALPHA)) if pvals_alt.size else np.nan

    null_valid = int(pvals_null.size)
    alt_valid = int(pvals_alt.size)
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s")

    if null_valid < int(0.8 * SIM_N_REPS) or alt_valid < int(0.8 * SIM_N_REPS):
        level = "fail"
    elif (
        abs(size_est - ALPHA) <= SIM_SIZE_TOL_PASS
        and power_est >= SIM_POWER_MIN_PASS
    ):
        level = "pass"
    elif (
        abs(size_est - ALPHA) > SIM_SIZE_TOL_FAIL
        or power_est < SIM_POWER_MIN_FAIL
    ):
        level = "fail"
    else:
        level = "warn"

    detail = (
        f"size={size_est:.3f}, power={power_est:.3f}, "
        f"null_valid={null_valid}/{SIM_N_REPS}, alt_valid={alt_valid}/{SIM_N_REPS}, "
        f"null_fail={fail_null}, alt_fail={fail_alt}"
    )
    _status(level, "E", detail)
    return {
        "level": level,
        "size_est": size_est,
        "power_est": power_est,
        "null_valid": null_valid,
        "alt_valid": alt_valid,
    }


def run_all_checks(df, prep, base_res, base_p):
    print("\n==============================")
    print("ROBUSTNESS CHECKS (A-E)")
    print("==============================")

    out = {}
    out["A"] = check_A_regime_support(prep, base_res["b_hat"])
    out["B"] = check_B_leave_one_series_out(df, X_COL, Y_COL, "series_id", base_res, base_p)

    _, profile = supF_common_break(prep, return_profile=True)
    out["C"] = check_C_supF_profile(profile)
    out["D"] = check_D_trimming_sensitivity(prep, base_res, base_p)

    if RUN_SIMULATION_CHECK:
        out["E"] = check_E_design_simulation(prep, base_res)
    else:
        out["E"] = {"level": "warn"}
        _status("warn", "E", "Skipped (RUN_SIMULATION_CHECK=False).")

    levels = [out[k]["level"] for k in ["A", "B", "C", "D", "E"]]
    n_fail = sum(l == "fail" for l in levels)
    n_warn = sum(l == "warn" for l in levels)

    if n_fail > 0:
        overall = "fail"
    elif n_warn > 0:
        overall = "warn"
    else:
        overall = "pass"

    _status(
        overall,
        "SUMMARY",
        f"overall={overall.upper()} (pass={levels.count('pass')}, warn={n_warn}, fail={n_fail})",
    )
    return out


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
    sign_matrix = enumerate_cluster_signs(prep["n_ids"])

    res, p_val, _ = exact_cluster_sign_supF_pvalue(prep, sign_matrix=sign_matrix)

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

    if RUN_ROBUSTNESS_CHECKS:
        run_all_checks(df, prep, res, p_val)

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
