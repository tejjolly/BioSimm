#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Breakpoint diagnostics for your FE segmented-regression supF workflow.

Implements:
(A) Regime support at b_hat (per-series left/right counts)
(B) Leave-one-series-out stability (jackknife over clusters)
(C) supF profile sharpness (F(b) vs b, top-5 candidates, gap metrics)
(D) Sensitivity to trimming / admissible candidates / MIN_SIDE_POINTS / MIN_SEFF
(E) Size/power simulation under your exact x-design (fixed x per series)

Dependencies: numpy, pandas, matplotlib
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# USER CONFIG
# =============================
CSV_PATH = "../data/data_manuscript.csv"

# You can set these like in your script:
Y_COL_IN = "FFR"      # "FFR" or "CFR" or other numeric column name
X_COL    = "HMR"      # "HMR" or "HSR"

SERIES_COLS = ["Condition", "Location", "Stenosis Percentage", "Length"]

# Analysis filtering (match your logic)
STENOSIS_MIN = 0.05
DROP_NO_STENOSIS_FOR_FFR = True  # your original behavior

MIN_POINTS_PER_SERIES = 2

# Baseline breakpoint scan settings (we will vary these in sensitivity checks)
BASE_MIN_SIDE_POINTS = 2     # per-series min points on EACH side at candidate b
BASE_MIN_SEFF        = 4     # min eligible series at candidate b

# Trimming / admissible candidate constraints (optional)
# - trim_frac constrains overall split so that at least trim_frac of ALL observations
#   lie on each side. (e.g., 0.10 means >= 10% on each side)
BASE_TRIM_FRAC_OVERALL = 0.00
# - you can also impose an absolute min total points on each side
BASE_MIN_TOTAL_SIDE = 0       # 0 disables

OUTDIR = "break_diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

RNG_SEED = 123


# =============================
# HELPERS: data prep
# =============================
def _to_numeric_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _stenosis_filter(df, sten_col="Stenosis Percentage", min_sten=0.05):
    if sten_col not in df.columns:
        raise KeyError(f"Missing '{sten_col}' needed for stenosis filtering.")
    return df[df[sten_col].notna() & (df[sten_col] >= float(min_sten))].copy()


def load_analysis_df(
    csv_path,
    *,
    x_col,
    y_col_in,
    series_cols,
    min_points_per_series=2,
    stenosis_min=0.05,
    drop_no_stenosis_for_ffr=True,
):
    df = pd.read_csv(csv_path)

    # Map your "FFR" alias to actual column
    if y_col_in == "FFR":
        y_col = "P_d/P_a"
        drop_no_stenosis = drop_no_stenosis_for_ffr
    else:
        y_col = y_col_in
        drop_no_stenosis = False

    _to_numeric_inplace(df, [x_col, y_col, "Stenosis Percentage", "Length", "Geometry Number"])

    missing = [c for c in series_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing SERIES_COLS in CSV: {missing}")

    df = df.dropna(subset=series_cols + [x_col, y_col]).copy()
    df["series_id"] = df[series_cols].astype(str).agg("|".join, axis=1)

    # Stenosis filter (analysis)
    if drop_no_stenosis:
        df = _stenosis_filter(df, sten_col="Stenosis Percentage", min_sten=stenosis_min)

    # Drop tiny series
    counts = df.groupby("series_id").size()
    keep_ids = set(counts.index[counts >= int(min_points_per_series)])
    df = df[df["series_id"].isin(keep_ids)].copy()

    return df, x_col, y_col


# =============================
# CORE: FE segmented regression supF (parameterized)
# =============================
def prep_supF(
    df,
    *,
    x_col,
    y_col,
    id_col,
    min_side_points_series=1,
    trim_frac_overall=0.0,
    min_total_side=0,
):
    """
    Prepare demeaned arrays and candidate b list.
    Adds overall trimming constraints:
      - overall left_count >= max(min_total_side, ceil(trim_frac_overall*n))
      - overall right_count >= same
    """
    d = df[[id_col, x_col, y_col]].dropna().copy()
    d = d.sort_values(x_col, kind="mergesort").reset_index(drop=True)

    x = d[x_col].to_numpy(float)
    y = d[y_col].to_numpy(float)

    sid_cat = pd.Categorical(d[id_col])
    sid = sid_cat.codes.astype(int)
    n_ids = len(sid_cat.categories)

    # within transform (series FE)
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
    b_cands_full = 0.5 * (x_unique[:-1] + x_unique[1:])

    n = len(x)
    min_side_overall = int(np.ceil(float(trim_frac_overall) * n))
    min_side_overall = max(min_side_overall, int(min_total_side), 1)

    b_cands = []
    elig_list = []
    use_list = []
    overall_counts = []  # (L, R)

    for b in b_cands_full:
        left = (x <= b)
        right = ~left
        nL = int(np.sum(left))
        nR = int(np.sum(right))

        # overall trimming constraint
        if (nL < min_side_overall) or (nR < min_side_overall):
            continue

        cntL = np.bincount(sid[left], minlength=n_ids)
        cntR = np.bincount(sid[right], minlength=n_ids)
        elig = (cntL >= int(min_side_points_series)) & (cntR >= int(min_side_points_series))

        b_cands.append(float(b))
        elig_list.append(elig)
        use_list.append(elig[sid])  # keep all obs from eligible series
        overall_counts.append((nL, nR))

    if len(b_cands) == 0:
        raise RuntimeError(
            "No candidate b survived overall trimming constraints. "
            "Try reducing trim_frac_overall / min_total_side."
        )

    return {
        "d": d, "x": x, "y": y,
        "sid": sid, "n_ids": n_ids,
        "cnt": cnt,
        "x_dm": x_dm, "y_dm": y_dm,
        "b_cands": np.asarray(b_cands, float),
        "elig_list": elig_list,
        "use_list": use_list,
        "overall_counts": overall_counts,
        "id_categories": sid_cat.categories,
    }


def supF_profile_common_break(prep, *, min_seff=4, y_dm_override=None):
    """
    Compute F(b) for each candidate. Returns a DataFrame + best row dict.
    """
    x = prep["x"]
    sid = prep["sid"]
    cnt = prep["cnt"]
    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"] if y_dm_override is None else y_dm_override
    b_cands = prep["b_cands"]
    elig_list = prep["elig_list"]
    use_list = prep["use_list"]
    overall_counts = prep["overall_counts"]
    n_ids = prep["n_ids"]

    rows = []
    best = None

    for (b, elig, use, (nL, nR)) in zip(b_cands, elig_list, use_list, overall_counts):
        Seff = int(np.sum(elig))
        if Seff < int(min_seff):
            continue

        xu = x_dm[use]
        yu = y_dm[use]
        sid_u = sid[use]
        n_use = yu.size

        # df for FE model: n_use - (#series intercepts) - (#slopes)
        df2 = int(n_use - Seff - 2)
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
        hbar = sh / cnt
        h_dm = h - hbar[sid_u]

        # Alt: yu ~ b1*xu + b2*h_dm
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

        num = (SSR0 - SSR1)
        den = SSR1 / float(df2)
        if den <= 0:
            continue
        F = float(num / den)

        row = dict(
            b=float(b),
            F=float(F),
            Seff=int(Seff),
            n_use=int(n_use),
            df2=int(df2),
            nL=int(nL), nR=int(nR),
            beta_pre=float(b1),
            beta_post=float(b1 + b2),
            delta_slope=float(b2),
            SSR0=float(SSR0),
            SSR1=float(SSR1),
        )
        rows.append(row)

        if (best is None) or (F > best["F"]):
            best = row

    prof = pd.DataFrame(rows).sort_values("b").reset_index(drop=True)
    if best is None:
        raise RuntimeError("No valid candidate split produced a finite supF (after min_seff/df constraints).")
    return prof, best


def exact_cluster_sign_supF_pvalue(prep, *, min_seff=4):
    """
    Exact (2^G) Rademacher sign enumeration at cluster level.
    Returns: best_row, p_value, null_stats (array)
    """
    prof_obs, best_obs = supF_profile_common_break(prep, min_seff=min_seff)
    obs = best_obs["F"]

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
        _, best_star = supF_profile_common_break(prep, min_seff=min_seff, y_dm_override=y_star)
        stats.append(best_star["F"])

    stats = np.asarray(stats, float)
    # small-sample "plus-one" adjustment
    p = (np.sum(stats >= obs) + 1.0) / (stats.size + 1.0)
    return best_obs, float(p), stats, prof_obs


def bootstrap_cluster_sign_supF_pvalue(prep, *, min_seff=4, B=999, seed=0):
    """
    Faster Monte Carlo wild cluster bootstrap p-value (Rademacher).
    Returns: best_obs_row, p_value, boot_stats (array)
    """
    rng = np.random.default_rng(seed)
    prof_obs, best_obs = supF_profile_common_break(prep, min_seff=min_seff)
    obs = best_obs["F"]

    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"]
    sid  = prep["sid"]
    n_ids = prep["n_ids"]

    beta0 = float(np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm))
    e0 = y_dm - beta0 * x_dm

    boot = np.empty(B, float)
    for b in range(B):
        g = rng.choice([-1.0, 1.0], size=n_ids)
        y_star = beta0 * x_dm + e0 * g[sid]
        _, best_star = supF_profile_common_break(prep, min_seff=min_seff, y_dm_override=y_star)
        boot[b] = best_star["F"]

    p = (np.sum(boot >= obs) + 1.0) / (B + 1.0)
    return best_obs, float(p), boot, prof_obs


# =============================
# (A) Regime support at b_hat
# =============================
def regime_support_table(df, *, x_col, id_col, b_hat):
    d = df[[id_col, x_col]].dropna().copy()
    d["is_left"] = d[x_col] <= float(b_hat)
    g = d.groupby(id_col)["is_left"].agg(["count", "sum"]).rename(columns={"sum": "n_left"})
    g["n_right"] = g["count"] - g["n_left"]
    g = g.sort_values(["n_left", "n_right"], ascending=[True, True]).reset_index()
    return g


# =============================
# (B) Leave-one-series-out stability
# =============================
def leave_one_out(df, *, x_col, y_col, id_col="series_id",
                  min_side_points_series=1, min_seff=4,
                  trim_frac_overall=0.0, min_total_side=0,
                  p_method="exact", B=999, seed=0):
    rows = []
    series_ids = sorted(df[id_col].unique())
    for sid_drop in series_ids:
        dsub = df[df[id_col] != sid_drop].copy()
        try:
            prep = prep_supF(
                dsub, x_col=x_col, y_col=y_col, id_col=id_col,
                min_side_points_series=min_side_points_series,
                trim_frac_overall=trim_frac_overall,
                min_total_side=min_total_side,
            )
            if p_method == "exact":
                best, p, _, _ = exact_cluster_sign_supF_pvalue(prep, min_seff=min_seff)
            else:
                best, p, _, _ = bootstrap_cluster_sign_supF_pvalue(prep, min_seff=min_seff, B=B, seed=seed)
            rows.append(dict(
                dropped=sid_drop,
                b_hat=best["b"],
                supF=best["F"],
                p_value=p,
                Seff=best["Seff"],
                delta_slope=best["delta_slope"],
            ))
        except Exception as e:
            rows.append(dict(
                dropped=sid_drop,
                b_hat=np.nan,
                supF=np.nan,
                p_value=np.nan,
                Seff=np.nan,
                delta_slope=np.nan,
                error=str(e),
            ))
    return pd.DataFrame(rows)


# =============================
# (C) supF profile sharpness
# =============================
def plot_supF_profile(prof, *, title, outpath_png):
    plt.figure(figsize=(8.5, 4.5))
    plt.plot(prof["b"], prof["F"])
    plt.xlabel("candidate breakpoint b")
    plt.ylabel("F(b)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=250, bbox_inches="tight")
    plt.close()


def profile_sharpness_metrics(prof):
    """
    Quantify “spiky vs plateau”:
      - top1, top2, gap
      - gap / top1
      - number of candidates within 5% of top1
    """
    if len(prof) == 0:
        return {}
    prof2 = prof.sort_values("F", ascending=False).reset_index(drop=True)
    top1 = float(prof2.loc[0, "F"])
    top2 = float(prof2.loc[1, "F"]) if len(prof2) > 1 else np.nan
    gap = top1 - top2 if np.isfinite(top2) else np.nan
    within_5pct = int(np.sum(prof["F"] >= 0.95 * top1))
    return dict(top1=top1, top2=top2, gap=gap, gap_over_top1=(gap / top1 if top1 > 0 and np.isfinite(gap) else np.nan),
                n_within_5pct=within_5pct)


# =============================
# (D) Sensitivity grid (trimming + eligibility)
# =============================
def sensitivity_grid(df, *, x_col, y_col, id_col="series_id",
                     min_side_points_list=(1, 2),
                     min_seff_list=(4, 6, 8),
                     trim_frac_list=(0.00, 0.10, 0.20),
                     min_total_side_list=(0, ),
                     p_method="exact", B=999, seed=0):
    rows = []
    for msp in min_side_points_list:
        for mse in min_seff_list:
            for tr in trim_frac_list:
                for mts in min_total_side_list:
                    try:
                        prep = prep_supF(
                            df, x_col=x_col, y_col=y_col, id_col=id_col,
                            min_side_points_series=msp,
                            trim_frac_overall=tr,
                            min_total_side=mts,
                        )
                        if p_method == "exact":
                            best, p, _, prof = exact_cluster_sign_supF_pvalue(prep, min_seff=mse)
                        else:
                            best, p, _, prof = bootstrap_cluster_sign_supF_pvalue(prep, min_seff=mse, B=B, seed=seed)
                        met = profile_sharpness_metrics(prof)
                        rows.append(dict(
                            min_side_points=msp,
                            min_seff=mse,
                            trim_frac_overall=tr,
                            min_total_side=mts,
                            b_hat=best["b"],
                            supF=best["F"],
                            p_value=p,
                            Seff=best["Seff"],
                            delta_slope=best["delta_slope"],
                            n_candidates=len(prof),
                            **met,
                        ))
                    except Exception as e:
                        rows.append(dict(
                            min_side_points=msp,
                            min_seff=mse,
                            trim_frac_overall=tr,
                            min_total_side=mts,
                            b_hat=np.nan,
                            supF=np.nan,
                            p_value=np.nan,
                            Seff=np.nan,
                            delta_slope=np.nan,
                            n_candidates=0,
                            error=str(e),
                        ))
    return pd.DataFrame(rows)


# =============================
# (E) Operating characteristics (size/power) under fixed x-design
# =============================
def compute_hinge_dm_all(prep, b):
    """
    Compute demeaned hinge term h_dm for ALL observations (not eligibility-filtered),
    using each series mean (like your model structure).
    """
    x = prep["x"]
    sid = prep["sid"]
    cnt = prep["cnt"]
    n_ids = prep["n_ids"]

    h = np.maximum(0.0, x - float(b))
    sh = np.bincount(sid, weights=h, minlength=n_ids)
    hbar = sh / cnt
    h_dm = h - hbar[sid]
    return h_dm


def simulate_size_power(
    df,
    *,
    x_col,
    y_col,
    id_col="series_id",
    min_side_points_series=1,
    min_seff=4,
    trim_frac_overall=0.0,
    min_total_side=0,
    alpha=0.05,
    nrep=500,
    b_alt=None,
    delta_alt=0.0,
    use_bootstrap_p=False,
    B_boot=399,
    seed=0,
):
    """
    Size: set delta_alt=0 (null). Power: set delta_alt != 0 with b_alt.
    DGP in demeaned space:
        y_dm = beta0*x_dm + delta*h_dm + e0*g_cluster

    Notes:
    - We keep the observed x-design (same x grid + clustering) and reuse the observed null residuals e0.
    - If use_bootstrap_p=False, we use a fixed critical value from the exact null sign-enumeration
      computed once from the observed null residuals. This is VERY fast and exact for the sign-flip DGP.
    - If use_bootstrap_p=True, we recompute a bootstrap p-value each rep (slower, closer to your pipeline).
    """
    rng = np.random.default_rng(seed)

    prep = prep_supF(
        df, x_col=x_col, y_col=y_col, id_col=id_col,
        min_side_points_series=min_side_points_series,
        trim_frac_overall=trim_frac_overall,
        min_total_side=min_total_side,
    )

    # Fit null on observed demeaned data
    x_dm = prep["x_dm"]
    y_dm = prep["y_dm"]
    sid  = prep["sid"]
    n_ids = prep["n_ids"]

    beta0 = float(np.sum(x_dm * y_dm) / np.sum(x_dm * x_dm))
    e0 = y_dm - beta0 * x_dm

    # Alternative hinge (if requested)
    if b_alt is None:
        # default to observed b_hat under baseline settings
        best_obs, _, _, _ = exact_cluster_sign_supF_pvalue(prep, min_seff=min_seff)
        b_alt = best_obs["b"]
    h_dm_all = compute_hinge_dm_all(prep, b_alt)

    # Fast critical value (exact sign enumeration once) under null residuals
    crit = None
    null_stats = None
    if not use_bootstrap_p:
        # exact null distribution from sign patterns
        # (matches your test’s reference distribution under the sign-flip DGP)
        best_obs, p_obs, stats, _ = exact_cluster_sign_supF_pvalue(prep, min_seff=min_seff)
        null_stats = stats.copy()
        # rejection using p <= alpha is equivalent to F >= crit with appropriate discrete cutoff
        # We'll compute crit as the smallest value such that P(F >= crit) <= alpha (conservative).
        sorted_stats = np.sort(null_stats)[::-1]  # descending
        # choose k so that (k+1)/(M+1) <= alpha  -> k <= alpha*(M+1) - 1
        M = len(sorted_stats)
        k = int(np.floor(alpha * (M + 1) - 1))
        k = max(k, 0)
        crit = float(sorted_stats[k])

    rejects = 0
    Fs = np.empty(nrep, float)
    pvals = np.empty(nrep, float)

    for r in range(nrep):
        g = rng.choice([-1.0, 1.0], size=n_ids)
        y_sim = beta0 * x_dm + float(delta_alt) * h_dm_all + e0 * g[sid]

        # compute supF statistic on y_sim
        prof_sim, best_sim = supF_profile_common_break(prep, min_seff=min_seff, y_dm_override=y_sim)
        F_sim = float(best_sim["F"])
        Fs[r] = F_sim

        if use_bootstrap_p:
            # recompute bootstrap p-value each time (slower)
            # Make a shallow copy of prep, just override y_dm via function call
            # We implement bootstrap by treating y_sim as "observed" y_dm inside this rep.
            # That requires refitting beta0 within the bootstrap function; easiest is to
            # temporarily patch prep["y_dm"].
            y_dm_saved = prep["y_dm"]
            prep["y_dm"] = y_sim
            try:
                _, p_sim, _, _ = bootstrap_cluster_sign_supF_pvalue(prep, min_seff=min_seff, B=B_boot, seed=int(seed + r + 1))
            finally:
                prep["y_dm"] = y_dm_saved
            pvals[r] = p_sim
            if p_sim <= alpha:
                rejects += 1
        else:
            # fast rejection using fixed crit from exact null distribution
            pvals[r] = np.nan
            if F_sim >= crit:
                rejects += 1

    return dict(
        alpha=float(alpha),
        nrep=int(nrep),
        b_alt=float(b_alt),
        delta_alt=float(delta_alt),
        reject_rate=float(rejects / nrep),
        crit=float(crit) if crit is not None else np.nan,
        Fs=Fs,
        pvals=pvals,
        null_stats=null_stats,
    )


# =============================
# MAIN: run all checks
# =============================
def main():
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(RNG_SEED)

    # ---- Load data ----
    df, x_col, y_col = load_analysis_df(
        CSV_PATH,
        x_col=X_COL,
        y_col_in=Y_COL_IN,
        series_cols=SERIES_COLS,
        min_points_per_series=MIN_POINTS_PER_SERIES,
        stenosis_min=STENOSIS_MIN,
        drop_no_stenosis_for_ffr=DROP_NO_STENOSIS_FOR_FFR,
    )

    print("\n==============================")
    print("DATA SUMMARY")
    print("==============================")
    print("n rows:", len(df))
    print("n series:", df["series_id"].nunique())
    print(df.groupby("series_id").size().describe())

    # ---- Baseline estimate (exact p) ----
    prep = prep_supF(
        df, x_col=x_col, y_col=y_col, id_col="series_id",
        min_side_points_series=BASE_MIN_SIDE_POINTS,
        trim_frac_overall=BASE_TRIM_FRAC_OVERALL,
        min_total_side=BASE_MIN_TOTAL_SIDE,
    )
    best, p_val, stats, prof = exact_cluster_sign_supF_pvalue(prep, min_seff=BASE_MIN_SEFF)

    print("\n==============================")
    print("BASELINE COMMON BREAK (exact sign enumeration)")
    print("==============================")
    print(f"b_hat ({x_col} units): {best['b']:.6g}")
    print(f"p-value: {p_val:.6g}")
    print(f"supF: {best['F']:.6g}")
    print(f"Seff: {best['Seff']}")
    print(f"pre-slope:  {best['beta_pre']:.6g}")
    print(f"post-slope: {best['beta_post']:.6g}")
    print(f"delta:      {best['delta_slope']:.6g}")

    # (A) Regime support
    support = regime_support_table(df, x_col=x_col, id_col="series_id", b_hat=best["b"])
    support_path = os.path.join(OUTDIR, "A_regime_support_at_bhat.csv")
    support.to_csv(support_path, index=False)
    print("\n[A] Regime support at b_hat saved to:", support_path)
    print(support)

    # (C) supF profile plot + sharpness metrics
    prof_path = os.path.join(OUTDIR, "C_supF_profile.csv")
    prof.to_csv(prof_path, index=False)

    met = profile_sharpness_metrics(prof)
    print("\n[C] Profile sharpness metrics:", met)

    plot_path = os.path.join(OUTDIR, "C_supF_profile.png")
    plot_supF_profile(
        prof,
        title=f"supF profile: {y_col} vs {x_col} (min_side={BASE_MIN_SIDE_POINTS}, min_seff={BASE_MIN_SEFF}, trim={BASE_TRIM_FRAC_OVERALL})",
        outpath_png=plot_path,
    )
    print("[C] Profile CSV saved to:", prof_path)
    print("[C] Profile plot saved to:", plot_path)

    # top-5 candidates
    top5 = prof.sort_values("F", ascending=False).head(5)
    top5_path = os.path.join(OUTDIR, "C_top5_candidates.csv")
    top5.to_csv(top5_path, index=False)
    print("[C] Top-5 candidates saved to:", top5_path)
    print(top5[["b", "F", "Seff", "nL", "nR", "delta_slope"]])

    # (B) Leave-one-series-out
    loo = leave_one_out(
        df,
        x_col=x_col, y_col=y_col,
        min_side_points_series=BASE_MIN_SIDE_POINTS,
        min_seff=BASE_MIN_SEFF,
        trim_frac_overall=BASE_TRIM_FRAC_OVERALL,
        min_total_side=BASE_MIN_TOTAL_SIDE,
        p_method="exact",
    )
    loo_path = os.path.join(OUTDIR, "B_leave_one_out.csv")
    loo.to_csv(loo_path, index=False)
    print("\n[B] Leave-one-out saved to:", loo_path)
    print(loo)

    # (D) Sensitivity grid
    sens = sensitivity_grid(
        df,
        x_col=x_col, y_col=y_col,
        min_side_points_list=(1, 2),
        min_seff_list=(4, 6, 8),
        trim_frac_list=(0.00, 0.10, 0.20),
        min_total_side_list=(0, 4, 6),  # absolute overall min on each side (0 disables)
        p_method="exact",
    )
    sens_path = os.path.join(OUTDIR, "D_sensitivity_grid.csv")
    sens.to_csv(sens_path, index=False)
    print("\n[D] Sensitivity grid saved to:", sens_path)
    # show a compact view of rows that succeeded
    print(sens.dropna(subset=["p_value"]).sort_values(["p_value", "supF"]).head(12)[
        ["min_side_points", "min_seff", "trim_frac_overall", "min_total_side",
         "b_hat", "p_value", "Seff", "delta_slope", "n_candidates", "gap_over_top1", "n_within_5pct"]
    ])

    # (E) Size / power simulation (fixed x-design)
    # Size under null (delta=0). Fast exact-critical-value mode.
    sim_null = simulate_size_power(
        df,
        x_col=x_col, y_col=y_col,
        min_side_points_series=BASE_MIN_SIDE_POINTS,
        min_seff=BASE_MIN_SEFF,
        trim_frac_overall=BASE_TRIM_FRAC_OVERALL,
        min_total_side=BASE_MIN_TOTAL_SIDE,
        alpha=0.05,
        nrep=500,
        b_alt=best["b"],
        delta_alt=0.0,
        use_bootstrap_p=False,   # fast + exact for sign-flip DGP
        seed=RNG_SEED,
    )
    print("\n[E] SIZE (null, delta=0) rejection rate:", sim_null["reject_rate"], "  crit:", sim_null["crit"])

    # Power under an effect size similar to your estimate (delta ≈ -0.0897)
    sim_alt = simulate_size_power(
        df,
        x_col=x_col, y_col=y_col,
        min_side_points_series=BASE_MIN_SIDE_POINTS,
        min_seff=BASE_MIN_SEFF,
        trim_frac_overall=BASE_TRIM_FRAC_OVERALL,
        min_total_side=BASE_MIN_TOTAL_SIDE,
        alpha=0.05,
        nrep=300,
        b_alt=best["b"],
        delta_alt=best["delta_slope"],   # use your estimated delta
        use_bootstrap_p=False,
        seed=RNG_SEED + 999,
    )
    print("[E] POWER (delta=estimated) rejection rate:", sim_alt["reject_rate"], "  crit:", sim_alt["crit"])

    # Save sim results summaries
    pd.DataFrame([
        {k: sim_null[k] for k in ["alpha", "nrep", "b_alt", "delta_alt", "reject_rate", "crit"]},
        {k: sim_alt[k]  for k in ["alpha", "nrep", "b_alt", "delta_alt", "reject_rate", "crit"]},
    ]).to_csv(os.path.join(OUTDIR, "E_size_power_summary.csv"), index=False)

    # Plot histogram of simulated supF stats
    plt.figure(figsize=(8.5, 4.5))
    plt.hist(sim_null["Fs"], bins=30, alpha=0.8, label="null (delta=0)")
    plt.hist(sim_alt["Fs"], bins=30, alpha=0.8, label="alt (delta=estimated)")
    if np.isfinite(sim_null["crit"]):
        plt.axvline(sim_null["crit"], linestyle="--", linewidth=2.0, label="crit (alpha)")
    plt.xlabel("supF statistic")
    plt.ylabel("count")
    plt.title("Simulated supF under null vs alt (fixed x-design)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "E_sim_supF_hist.png"), dpi=250, bbox_inches="tight")
    plt.close()

    print("\nAll outputs written to:", OUTDIR)


if __name__ == "__main__":
    main()