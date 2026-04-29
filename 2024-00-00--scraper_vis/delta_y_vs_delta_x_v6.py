#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common-break (shared breakpoint) detection across multiple series using a
SaRa-style local-statistic scan on FE (within-series demeaned) slope estimates.

This is the *simplest practical* implementation in the spirit of:
  - "common breaks in linear panel data models" via local statistics + screening.

Your data format:
Condition, Geometry Number, Location, Stenosis Percentage, Length, ..., HMR, P_d/P_a, ...

What it does:
1) Build a series_id (your "panel unit") from design columns
2) Choose y = P_d/P_a (FFR) and x = HMR
3) Sort ALL observations by x to form a shared domain ordering
4) For each candidate split index t, estimate FE slope on left/right windows
   (within-series demeaned OLS slope)
5) Compute a SaRa-like statistic D(t,h) = sqrt(S_eff) * (beta_R - beta_L)
6) Return:
   - b_hat in x-units (midpoint between x[t] and x[t+1])
   - sup|D|
   - a permutation p-value under the null "no break" using residual permutation
     within series

Notes:
- This is designed for *unbalanced* panels (not every series has every x).
- If your series are monotone in x (HMR), great. If not, it still works but
  interpretation is weaker.
"""

import numpy as np
import pandas as pd

# =============================
# CONFIG YOU'LL EDIT
# =============================
CSV_PATH = "../data/data_manuscript.csv"  # change me
# CSV_PATH = "../data/break_test.csv"  # change me
Y_COL = "P_d/P_a"                         # FFR
X_COL = "HMR"                             # domain
SERIES_COLS = ["Condition", "Location", "Stenosis Percentage", "Length"]

MIN_POINTS_PER_SERIES = 3      # drop tiny series
H = 6                      # SaRa bandwidth in "global x-sorted index" units
EDGE_GUARD = 1                # if None, uses H; else integer
N_PERM = 100                 # for p-value
SEED = 0

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
    df = max(n - 2, 1)
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
        if Seff < 2:
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

    # drop tiny series
    counts = df.groupby("series_id").size()
    keep_ids = set(counts.index[counts >= MIN_POINTS_PER_SERIES])
    df = df[df["series_id"].isin(keep_ids)].copy()

    print("==============================")
    print("DATA SUMMARY")
    print("==============================")
    print("n rows:", len(df))
    print("n series:", df["series_id"].nunique())
    print(df.groupby("series_id").size().describe())

    scan = scan_common_break_by_threshold(
        df,
        x_col=X_COL,
        y_col=Y_COL,
        id_col="series_id",
        min_side_points=2,  # start here; if still fails, set to 1
    )

    print("\n==============================")
    print("COMMON BREAK (threshold scan)")
    print("==============================")
    print(f"b_hat ({X_COL} units): {scan['b_hat']:.6g}")
    print(f"sup|D|: {scan['sup_absD']:.6g}")
    print(f"Seff (series contributing): {scan['Seff']}")
    print(f"beta_L: {scan['beta_L']:.6g}")
    print(f"beta_R: {scan['beta_R']:.6g}")
    print(f"delta_beta: {(scan['beta_R'] - scan['beta_L']):.6g}")

    # obs, pval, _ = perm_pvalue_supD_thresholdscan(
    #     df_full=df,
    #     x_col=X_COL,
    #     y_col=Y_COL,
    #     id_col="series_id",
    #     min_side_points=2,  # if it errors, try 1
    #     n_perm=N_PERM,
    #     seed=SEED,
    # )

    # print("\n==============================")
    # print("SIGNIFICANCE (Permutation under no-break FE linear null)")
    # print("==============================")
    # print(f"Observed sup|D|: {obs:.6g}")
    # print(f"Permutation p-value: {pval:.6g}")

    prep = prep_threshold_scan(df, x_col=X_COL, y_col=Y_COL, id_col="series_id", min_side_points=2)

    scan = supT_common_break(prep)
    print("\n==============================")
    print("COMMON BREAK (sup|t| threshold scan)")
    print("==============================")
    print(f"b_hat ({X_COL} units): {scan['b_hat']:.6g}")
    print(f"sup|t|: {scan['sup_absT']:.6g}")
    print(f"Seff: {scan['Seff']}")
    print(f"beta_L: {scan['beta_L']:.6g}")
    print(f"beta_R: {scan['beta_R']:.6g}")
    print(f"delta_beta: {(scan['beta_R'] - scan['beta_L']):.6g}")

    obsT, p_boot, _ = wild_bootstrap_pvalue(prep, n_boot=N_PERM, seed=SEED, progress_every=200)

    print("\n==============================")
    print("SIGNIFICANCE (wild bootstrap under FE no-break null)")
    print("==============================")
    print(f"Observed sup|t|: {obsT:.6g}")
    print(f"Bootstrap p-value: {p_boot:.6g}")
if __name__ == "__main__":
    main()