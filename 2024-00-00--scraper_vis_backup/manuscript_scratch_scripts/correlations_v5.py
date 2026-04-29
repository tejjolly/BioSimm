#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.api as sms
import pingouin as pg


# =============================================================================
# PLOT DEFAULTS
# =============================================================================
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 600,
})


# =============================================================================
# USER SETTINGS
# =============================================================================
DATA_CSV = "../../data/data_manuscript.csv"
SAVE_DIR = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2024-00-00--scraper_vis/images"
SHOW_PLOTS = True  # set False if you only want to save SVGs

# Triangulation artifact control (lower masks more aggressively)
TRI_MASK_MAXEDGE_QUANTILE = 0.90

# Threshold styles (dashed, black, slightly thicker)
THRESH_STYLE = dict(colors="k", linestyles="--", linewidths=2.4)

# CFR–FFR chosen-point markers/colors
CFR_FFR_NONCHOSEN_STYLE = dict(marker="o", s=70, facecolors="none", edgecolors="k", linewidths=1.6)
FFR_CHOSEN_STYLE = dict(marker="x", s=120, linewidths=2.6)  # color set programmatically (BuPu mid)
CFR_CHOSEN_STYLE = dict(marker="^", s=110)                   # face/edge set programmatically (viridis mid)


# =============================================================================
# STATS HELPERS
# =============================================================================
def linear_vs_quadratic_F(x, y):
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    X_lin = sm.add_constant(x)
    X_quad = sm.add_constant(np.c_[x, x**2])

    m_lin = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()

    a = anova_lm(m_lin, m_quad)
    F, p = a.loc[1, "F"], a.loc[1, "Pr(>F)"]
    return F, float(p), (p >= 0.05)  # True -> linear is adequate


def pearson_spearman(x, y):
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return (r_p, p_p), (r_s, p_s), len(df)


def analyze_pair_simple(x, y, label="X vs Y"):
    F, pF, is_linear = linear_vs_quadratic_F(x, y)
    (r_p, p_p), (r_s, p_s), n = pearson_spearman(x, y)

    print(f"[{label}] Nested F-test (quad vs lin): F={F:.3g}, p={pF:.3g}  -> {'linear' if is_linear else 'nonlinear'}")
    if is_linear:
        print(
            f"Report Pearson (primary): R={r_p:+.3f}, p={p_p:.3g}, n={n}; "
            f"Spearman (robustness): ρ={r_s:+.3f}, p={p_s:.3g}"
        )
        return {"primary": ("pearson", r_p, p_p, n), "secondary": ("spearman", r_s, p_s, n)}

    print(
        f"Report Spearman (primary): ρ={r_s:+.3f}, p={p_s:.3g}, n={n}; "
        f"Pearson (comparison): r={r_p:+.3f}, p={p_p:.3g}"
    )
    return {"primary": ("spearman", r_s, p_s, n), "secondary": ("pearson", r_p, p_p, n)}


def local_sensitivity_band(
    df, x_var="HMR", y_var="HSR", z_var="P_d/P_a", condition="Hyperemic",
        exclude_no_stenosis=True, stenosis_col="Stenosis Percentage", location=None,
):
    """
    Computes local standardized sensitivities β_x, β_y of z_var to x_var,y_var
    inside a narrow band around the clinical threshold:
      * FFR (P_d/P_a): 0.80 ± 0.05
      * CFR:           2.00 ± 0.25

    Uses conventional SEs if residuals pass normality + homoscedasticity,
    otherwise HC3 robust SEs. Prints n, which SEs were used, and p-values.
    Returns a dict with betas, CIs, p-values, etc. plus the slice used.
    """
    if z_var == "CFR":
        thresh, tol = 2.0, 0.25
    else:
        thresh, tol = 0.80, 0.05

    dff = df.loc[
        (df["Condition"] == condition)
        & df[x_var].notna()
        & df[y_var].notna()
        & df[z_var].notna()
    ].copy()

    if location is not None and "Location" in dff.columns:
        dff = dff.loc[dff["Location"] == location]

    near = dff.loc[(dff[z_var] - thresh).abs() <= tol].copy()

    if exclude_no_stenosis and (stenosis_col in near.columns):
        near = near.loc[near[stenosis_col] >= 0.05]

    n = len(near)
    if n < 3:
        print(f"[Local sensitivity] Not enough points near {z_var}={thresh:.2f}±{tol} (n={n}).")
        return {"n": n, "slice": near, "thresh": thresh, "tol": tol}

    sx = near[x_var].std(ddof=1)
    sy = near[y_var].std(ddof=1)
    if (sx == 0) or (sy == 0) or np.isnan(sx) or np.isnan(sy):
        print(f"[Local sensitivity] Degenerate std in band (sx={sx}, sy={sy}); cannot standardize.")
        return {"n": n, "slice": near, "thresh": thresh, "tol": tol}

    # standardize predictors
    X1 = (near[x_var] - near[x_var].mean()) / sx
    X2 = (near[y_var] - near[y_var].mean()) / sy
    Y = near[z_var].to_numpy(float)

    X_sm = sm.add_constant(np.c_[X1.to_numpy(float), X2.to_numpy(float)])

    # plain OLS for residuals / R², etc.
    m_ols = sm.OLS(Y, X_sm).fit()

    # residual checks
    _, p_norm = stats.shapiro(m_ols.resid) if len(m_ols.resid) <= 5000 else (np.nan, np.nan)
    _, p_bp, _, _ = sms.het_breuschpagan(m_ols.resid, m_ols.model.exog)
    _, p_white, _, _ = sms.het_white(m_ols.resid, m_ols.model.exog)

    use_hc3 = (not np.isnan(p_norm) and p_norm < 0.05) or (p_bp < 0.05) or (p_white < 0.05)
    m = sm.OLS(Y, X_sm).fit(cov_type="HC3") if use_hc3 else m_ols
    cov_label = "HC3 robust SEs" if use_hc3 else "conventional SEs"

    b0, b1_std, b2_std = m.params
    ci = m.conf_int(alpha=0.05)
    ci_b1_std = ci[1]
    ci_b2_std = ci[2]

    p_b1 = float(m.pvalues[1])
    p_b2 = float(m.pvalues[2])
    t_b1 = float(m.tvalues[1])
    t_b2 = float(m.tvalues[2])

    # fit stats from plain OLS
    resid = Y - m_ols.fittedvalues
    rss = float(np.sum(resid**2))
    sst = float(np.sum((Y - Y.mean())**2))
    r2 = 1 - rss / sst if sst > 0 else np.nan
    p_ = X_sm.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p_) if n > p_ else np.nan
    rmse = float(np.sqrt(rss / n))

    print(
        f"Local standardized sensitivities near {z_var} = {thresh:.2f} ± {tol} "
        f"(n={n}; inference via {cov_label}; "
        f"Shapiro p={p_norm:.3g}, BP p={p_bp:.3g}, White p={p_white:.3g}):\n"
        f"  {x_var}: β={b1_std:+.3f}  (p={p_b1:.3g}, t={t_b1:.2f})  [ {ci_b1_std[0]:+.3f}, {ci_b1_std[1]:+.3f} ]\n"
        f"  {y_var}: β={b2_std:+.3f}  (p={p_b2:.3g}, t={t_b2:.2f})  [ {ci_b2_std[0]:+.3f}, {ci_b2_std[1]:+.3f} ]\n"
        f"  R^2={r2:.3f}, Adj R^2={adj_r2:.3f}, RMSE={rmse:.4f}"
    )

    return {
        "n": n,
        "thresh": thresh,
        "tol": tol,
        "betas_std": {"const": b0, x_var: b1_std, y_var: b2_std},
        "ci_std": {x_var: tuple(ci_b1_std), y_var: tuple(ci_b2_std)},
        "pvals": {x_var: p_b1, y_var: p_b2},
        "tvals": {x_var: t_b1, y_var: t_b2},
        "tests": {"shapiro_p": p_norm, "bp_p": p_bp, "white_p": p_white},
        "used": cov_label,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "slice": near,
    }


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _save_svg(fig, filename):
    _ensure_dir(SAVE_DIR)
    outpath = os.path.join(SAVE_DIR, filename)
    fig.savefig(outpath, transparent=True, bbox_inches="tight")
    print(f"Saved → {outpath}")

def triangulated_band_map(
    df,
    *,
    x="HMR",
    y="HSR",
    z="P_d/P_a",
    levels=None,
    threshold=None,
    cmap_name="BuPu",
    colorbar_label=None,
    out_svg="triangulated.svg",
    mask_maxedge_quantile=0.90,
):
    """
    Triangulated (Delaunay) response-surface visualization using tricontourf,
    with a dashed black threshold contour. Saves transparent SVG.
    """
    d = df[[x, y, z]].dropna().copy()
    if len(d) < 5:
        print(f"[{out_svg}] Not enough points to triangulate (n={len(d)}). Skipping.")
        return

    # De-duplicate identical (x,y) which can break QHull; average z.
    d = d.groupby([x, y], as_index=False)[z].mean()

    X = d[x].to_numpy(float)
    Y = d[y].to_numpy(float)
    Z = d[z].to_numpy(float)

    tri = mtri.Triangulation(X, Y)

    # Mask long-edge triangles (reduces artifacts in sparse regions)
    tris = tri.triangles
    xtri = X[tris]
    ytri = Y[tris]
    e0 = np.sqrt((xtri[:, 1] - xtri[:, 0])**2 + (ytri[:, 1] - ytri[:, 0])**2)
    e1 = np.sqrt((xtri[:, 2] - xtri[:, 1])**2 + (ytri[:, 2] - ytri[:, 1])**2)
    e2 = np.sqrt((xtri[:, 0] - xtri[:, 2])**2 + (ytri[:, 0] - ytri[:, 2])**2)
    max_edge = np.maximum(np.maximum(e0, e1), e2)
    cutoff = np.quantile(max_edge, mask_maxedge_quantile)
    tri.set_mask(max_edge > cutoff)

    if levels is None:
        zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        levels = np.linspace(zmin, zmax, 12)

    if colorbar_label is None:
        colorbar_label = "FFR" if z == "P_d/P_a" else z

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    cf = ax.tricontourf(tri, Z, levels=levels, cmap=plt.get_cmap(cmap_name))

    # Threshold contour: dashed black, slightly thicker
    if threshold is not None:
        ax.tricontour(tri, Z, levels=[threshold], **THRESH_STYLE)

    # Overlay points (open circles)
    ax.scatter(X, Y, s=18, facecolors="none", edgecolors="k", linewidths=0.6, alpha=0.9)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(colorbar_label)

    _save_svg(fig, out_svg)

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_cfr_ffr_with_chosen_points(df_full, res_ffr, res_cfr, out_svg="CFR_vs_FFR_chosen.svg"):
    """
    CFR vs "FFR" plot where x-data is P_d/P_a (Hyperemic) but the axis label is FFR.
      - all non-chosen points: open circles black outline, no fill
      - FFR-band chosen: marker x, BuPu midpoint color
      - CFR-band chosen: marker triangle, viridis midpoint color
      - dashed black threshold lines at x=0.8 and y=2.0
    """
    d = df_full[["P_d/P_a", "CFR"]].dropna().copy()
    if len(d) == 0:
        print(f"[{out_svg}] No rows with both P_d/P_a and CFR. Skipping.")
        return

    idx_ffr = set(res_ffr.get("slice", pd.DataFrame()).index) if isinstance(res_ffr, dict) else set()
    idx_cfr = set(res_cfr.get("slice", pd.DataFrame()).index) if isinstance(res_cfr, dict) else set()

    mask_ffr = d.index.isin(idx_ffr)
    mask_cfr = d.index.isin(idx_cfr)
    mask_other = ~(mask_ffr | mask_cfr)

    x = d["P_d/P_a"].to_numpy(float)
    y = d["CFR"].to_numpy(float)

    # Colors at midpoint of requested colormaps
    c_ffr = plt.get_cmap("BuPu")(0.8)
    c_cfr = plt.get_cmap("viridis")(0.50)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))

    ax.scatter(x[mask_other], y[mask_other], **CFR_FFR_NONCHOSEN_STYLE, zorder=2, label="All other points")

    ax.scatter(x[mask_ffr], y[mask_ffr], color=c_ffr, **FFR_CHOSEN_STYLE, zorder=4, label="FFR band points")

    ax.scatter(
        x[mask_cfr], y[mask_cfr],
        facecolors=c_cfr, edgecolors=c_cfr, linewidths=1.0, **CFR_CHOSEN_STYLE,
        zorder=5, label="CFR band points"
    )

    ax.axvline(0.80, color="k", linestyle="--", linewidth=2.4, zorder=1)
    ax.axhline(2.00, color="k", linestyle="--", linewidth=2.4, zorder=1)

    ax.set_xlabel("FFR")
    ax.set_ylabel("CFR")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12, loc="best")

    _save_svg(fig, out_svg)

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_cfr_ffr_two_band_plots(df_full, res_ffr, res_cfr, save_dir, show_plots=True):
    """
    Creates two CFR–FFR plots:
      1) CFR-band emphasized: CFR threshold strong; FFR threshold faint (alpha=0.25)
      2) FFR-band emphasized: FFR threshold strong; CFR threshold faint (alpha=0.25)
    """
    d = df_full[["P_d/P_a", "CFR"]].dropna().copy()
    if len(d) == 0:
        print("[CFR–FFR band plots] No rows with both P_d/P_a and CFR. Skipping.")
        return

    idx_ffr = set(res_ffr.get("slice", pd.DataFrame()).index) if isinstance(res_ffr, dict) else set()
    idx_cfr = set(res_cfr.get("slice", pd.DataFrame()).index) if isinstance(res_cfr, dict) else set()

    ffr_thresh = float(res_ffr.get("thresh", 0.80))
    cfr_thresh = float(res_cfr.get("thresh", 2.00))

    mask_ffr = d.index.isin(idx_ffr)
    mask_cfr = d.index.isin(idx_cfr)

    x = d["P_d/P_a"].to_numpy(float)  # display as FFR
    y = d["CFR"].to_numpy(float)

    nonchosen = dict(marker="o", s=70, facecolors="none", edgecolors="k", linewidths=1.6)

    # -------------------------
    # (A) CFR band emphasized
    # -------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cfr_tol = float(res_cfr.get("tol", 0.25))
    ax.axhspan(cfr_thresh - cfr_tol, cfr_thresh + cfr_tol, color="gray", alpha=0.33, zorder=0)

    mask_other = ~mask_cfr
    ax.scatter(x[mask_other], y[mask_other], **nonchosen, zorder=2)

    ax.scatter(
        x[mask_cfr], y[mask_cfr],
        marker="o", s=70, facecolors="k", edgecolors="k", linewidths=1.2, zorder=5
    )

    ax.axhline(cfr_thresh, color="k", linestyle="--", linewidth=2.4, alpha=1.0, zorder=1)
    ax.axvline(ffr_thresh, color="k", linestyle="--", linewidth=2.4, alpha=0.25, zorder=1)

    ax.set_xlabel("FFR")
    ax.set_ylabel("CFR")
    ax.grid(False)

    _ensure_dir(save_dir)
    out1 = os.path.join(save_dir, "CFR_vs_FFR_CFRband.svg")
    fig.savefig(out1, transparent=True, bbox_inches="tight")
    print(f"Saved → {out1}")

    if show_plots:
        plt.show()
    plt.close(fig)

    # -------------------------
    # (B) FFR band emphasized
    # -------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ffr_tol = float(res_ffr.get("tol", 0.05))
    ax.axvspan(ffr_thresh - ffr_tol, ffr_thresh + ffr_tol, color="gray", alpha=0.33, zorder=0)

    mask_other = ~mask_ffr
    ax.scatter(x[mask_other], y[mask_other], **nonchosen, zorder=2)

    ax.scatter(
        x[mask_ffr], y[mask_ffr],
        marker="o", s=70, facecolors="k", edgecolors="k", linewidths=1.2, zorder=5
    )

    ax.axvline(ffr_thresh, color="k", linestyle="--", linewidth=2.4, alpha=1.0, zorder=1)
    ax.axhline(cfr_thresh, color="k", linestyle="--", linewidth=2.4, alpha=0.25, zorder=1)

    ax.set_xlabel("FFR")
    ax.set_ylabel("CFR")
    ax.grid(False)

    out2 = os.path.join(save_dir, "CFR_vs_FFR_FFRband.svg")
    fig.savefig(out2, transparent=True, bbox_inches="tight")
    print(f"Saved → {out2}")

    if show_plots:
        plt.show()
    plt.close(fig)

def plot_cfr_ffr_joint_band_plot(df_full, res_ffr, res_cfr, save_dir, show_plots=True,
                                 out_svg="CFR_vs_FFR_joint_bands.svg"):
    """
    Single CFR–FFR plot showing BOTH bands on one axes, matching the visual style of
    plot_cfr_ffr_two_band_plots (circle markers only).

    - Outside band: open circles (no fill)
    - Within band (FFR band OR CFR band): filled circles (darker)
    - Legend: only "Within band" vs "Outside band" (filled vs open)
    - Threshold lines: colored (CFR: viridis(0.33), FFR: viridis(0.66))
    - Optional light shading of the two bands in those same colors
    """
    d = df_full[["P_d/P_a", "CFR"]].dropna().copy()
    if len(d) == 0:
        print("[CFR–FFR joint band plot] No rows with both P_d/P_a and CFR. Skipping.")
        return

    # chosen indices from band slices (consistent with your other plotting functions)
    idx_ffr = set(res_ffr.get("slice", pd.DataFrame()).index) if isinstance(res_ffr, dict) else set()
    idx_cfr = set(res_cfr.get("slice", pd.DataFrame()).index) if isinstance(res_cfr, dict) else set()
    idx_inband = idx_ffr | idx_cfr

    ffr_thresh = float(res_ffr.get("thresh", 0.80))
    ffr_tol    = float(res_ffr.get("tol", 0.05))
    cfr_thresh = float(res_cfr.get("thresh", 2.00))
    cfr_tol    = float(res_cfr.get("tol", 0.25))

    mask_inband = d.index.isin(idx_inband)
    mask_outband = ~mask_inband

    x = d["P_d/P_a"].to_numpy(float)  # display as FFR
    y = d["CFR"].to_numpy(float)

    # requested line colors
    c_cfr = plt.get_cmap("viridis")(0.33)
    c_ffr = plt.get_cmap("viridis")(0.66)

    # marker styles (match your two-band plots: circles only)
    outside_style = dict(marker="o", s=70, facecolors="none", edgecolors="k", linewidths=1.6)
    inside_style  = dict(marker="o", s=70, facecolors="k", edgecolors="k", linewidths=1.2)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))

    # band shading (subtle; keeps the overall style similar)
    ax.axvspan(ffr_thresh - ffr_tol, ffr_thresh + ffr_tol, color=c_ffr, alpha=0.18, zorder=0)
    ax.axhspan(cfr_thresh - cfr_tol, cfr_thresh + cfr_tol, color=c_cfr, alpha=0.18, zorder=0)

    # outside band
    ax.scatter(x[mask_outband], y[mask_outband], **outside_style, zorder=2, label="Outside band")
    # within band (either)
    ax.scatter(x[mask_inband], y[mask_inband], **inside_style, zorder=3, label="Within band")

    # threshold lines (colored as requested)
    ax.axvline(ffr_thresh, color=c_ffr, linestyle="--", linewidth=2.4, zorder=1)
    ax.axhline(cfr_thresh, color=c_cfr, linestyle="--", linewidth=2.4, zorder=1)

    ax.set_xlabel("FFR")
    ax.set_ylabel("CFR")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12, loc="best")

    _ensure_dir(save_dir)
    outpath = os.path.join(save_dir, out_svg)
    fig.savefig(outpath, transparent=True, bbox_inches="tight")
    print(f"Saved → {outpath}")

    if show_plots:
        plt.show()
    plt.close(fig)

# =============================================================================
# TOP-LEVEL EXECUTION (matches your current behavior)
# =============================================================================
df_all = pd.read_csv(DATA_CSV)
df_full = df_all[df_all["Condition"] == "Hyperemic"].copy()

y_vars = ["WSS_TE_Area", "WSS_TE_Area_min", "WSS_LE_Area_min"]
x_var = "HMR"
covar = "Q_distal"

for y_var in y_vars:
    # Correlation block
    x_variable = x_var
    y_variable = y_var  # FFR
    _ = analyze_pair_simple(df_full[x_variable], df_full[y_variable], label=f"{y_variable} vs. {x_variable}")
    # print("")

    res_partial = pg.partial_corr(
        data=df_full,
        x=x_variable,
        y=y_variable,
        covar=covar,
        method="pearson",
    )
    print(f"     Partial Corr [{y_variable} and {x_variable}]")
    print(f"     {res_partial}")
    print("")

# # Local sensitivities near FFR=0.80±0.05
# res_ffr = local_sensitivity_band(
#     df_full,
#     x_var="HMR",
#     y_var="HSR",
#     z_var="P_d/P_a",
#     condition="Hyperemic",
#     exclude_no_stenosis=True,
#     stenosis_col="Stenosis Percentage",
#     location=None,
# )
# print("")

# # CFR band
# res_cfr = local_sensitivity_band(
#     df_full,
#     x_var="HMR",
#     y_var="HSR",
#     z_var="CFR",
#     condition="Hyperemic",
#     exclude_no_stenosis=True,
# )
# print("CFR p-values:", res_cfr.get("pvals", {}))

# -----------------------------------------------------------------------------
# OPTIONAL PLOTTING (was commented in your original)
# -----------------------------------------------------------------------------
# triangulated_band_map(
#     df_full,
#     x="HMR", y="HSR", z="P_d/P_a",
#     levels=[0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
#     threshold=0.80,
#     cmap_name="BuPu",
#     colorbar_label="FFR",
#     out_svg="FFR_bands_over_HMR_HSR.svg",
#     mask_maxedge_quantile=TRI_MASK_MAXEDGE_QUANTILE,
# )
#
# triangulated_band_map(
#     df_full,
#     x="HMR", y="HSR", z="CFR",
#     levels=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
#     threshold=2.0,
#     cmap_name="viridis",
#     colorbar_label="CFR",
#     out_svg="CFR_bands_over_HMR_HSR.svg",
#     mask_maxedge_quantile=TRI_MASK_MAXEDGE_QUANTILE,
# )
#
# plot_cfr_ffr_two_band_plots(
#     df_full=df_full,
#     res_ffr=res_ffr,
#     res_cfr=res_cfr,
#     save_dir=SAVE_DIR,
#     show_plots=SHOW_PLOTS,
# )

# # ---- add this call at the very bottom (after plot_cfr_ffr_two_band_plots) ----
# plot_cfr_ffr_joint_band_plot(
#     df_full=df_full,
#     res_ffr=res_ffr,
#     res_cfr=res_cfr,
#     save_dir=SAVE_DIR,
#     show_plots=SHOW_PLOTS,
# )