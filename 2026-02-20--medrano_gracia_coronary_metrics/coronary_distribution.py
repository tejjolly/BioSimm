#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import os, math, pandas as pd

# =============================
# CONFIG
# =============================
X_PAD_SIGMA = 4.0
DPI = 400
SAVE_SVG = True
OUTDIR = "images/"
os.makedirs(OUTDIR, exist_ok=True)

CMAP = plt.get_c_cmap("BuPu") if hasattr(plt, "get_c_cmap") else plt.get_cmap("BuPu")
COLOR_SINGLE = CMAP(0.50)
COLOR_TWO_A = CMAP(0.33)   # Female
COLOR_TWO_B = CMAP(0.66)   # Male
COLOR_DESIGN = "black"

LINEWIDTH_PDF = 2.2
ALPHA_FILL = 0.18
LINEWIDTH_DESIGN = 1.8

# =============================
# USER DESIGN VALUES
# =============================
DESIGN = {
    "angle_ladlcx_B_deg": 75.0,
    "angle_ladd1_B_deg": 51.0,
    "angle_lcxom1_B_deg": 55.0,
    "length_lmca_mm": 28.0,                 # 2.8 cm
    "diameter_lad_mm": 3.2,
    "diameter_lcx_mm": 3.0,
    "path_ostium_to_D1_mm": 63.0,           # (2.8 + 3.5) cm
    "path_ostium_to_OM1_mm": 69.0,          # (2.8 + 4.1) cm
}

# =============================
# STATS
# =============================
_seg_LMB = {"mean": 10.5, "sd": 5.3}
_seg_D1  = {"mean": 35.5, "sd": 15.2}
_seg_OM1 = {"mean": 41.3, "sd": 20.4}

def _sum_mean_sd(*parts):
    mu = float(sum(p["mean"] for p in parts))
    sd = float(math.sqrt(sum((p["sd"] ** 2) for p in parts)))
    return {"mean": mu, "sd": sd}

STATS = {
    "angle_ladlcx_B_deg": {
        "label": "LAD/LCX bifurcation angle",
        "unit": "deg",
        "groups": {"All": {"mean": 75.2, "sd": 23.3, "q1": 61.4, "q2": 73.1, "q3": 88.9}},
        "prefer_sex_split": False,
    },
    "angle_ladd1_B_deg": {
        "label": "LAD, D1 bifurcation angle",
        "unit": "deg",
        "groups": {"All": {"mean": 51.5, "sd": 16.5, "q1": 39.0, "q2": 50.9, "q3": 63.4}},
        "prefer_sex_split": False,
    },
    "angle_lcxom1_B_deg": {
        "label": "LCX, OM1 bifurcation angle (B)",
        "unit": "deg",
        "groups": {"All": {"mean": 55.4, "sd": 23.7, "q1": 38.4, "q2": 52.2, "q3": 68.7}},
        "prefer_sex_split": False,
    },
    "length_lmca_mm": {
        "label": "Left main length (ostium→LAD/LCX bifurcation)",
        "unit": "mm",
        "groups": {"All": {"mean": 10.2, "sd": 5.6, "q1": 7.0, "q2": 10.0, "q3": 13.0}},
        "prefer_sex_split": False,
    },
    "diameter_lad_mm": {
        "label": "LAD diameter",
        "unit": "mm",
        "groups": {
            "Female": {"mean": 3.0, "sd": 0.6, "q1": 2.5, "q2": 3.1, "q3": 3.4},
            "Male":   {"mean": 3.7, "sd": 0.6, "q1": 3.2, "q2": 3.7, "q3": 4.1},
        },
        "prefer_sex_split": True,
    },
    "diameter_lcx_mm": {
        "label": "LCX diameter",
        "unit": "mm",
        "groups": {
            "Female": {"mean": 2.8, "sd": 0.7, "q1": 2.3, "q2": 2.8, "q3": 3.3},
            "Male":   {"mean": 3.6, "sd": 0.6, "q1": 3.1, "q2": 3.6, "q3": 3.9},
        },
        "prefer_sex_split": True,
    },
    "path_ostium_to_D1_mm": {
        "label": "Path length: ostium→D1 bifurcation",
        "unit": "mm",
        "groups": {"All": _sum_mean_sd(_seg_LMB, _seg_D1)},  # no quartiles available
        "prefer_sex_split": False,
    },
    "path_ostium_to_OM1_mm": {
        "label": "Path length: ostium→OM1 bifurcation",
        "unit": "mm",
        "groups": {"All": _sum_mean_sd(_seg_LMB, _seg_OM1)},  # no quartiles available
        "prefer_sex_split": False,
    },
}

# =============================
# BOUNDS (physically valid support)
# =============================
# For these Medrano bifurcation angles, [0, 180] is typically appropriate.
# If you *know* yours are directional and can exceed 180, change upper to np.inf.
BOUNDS = {
    "angle_ladlcx_B_deg": (0.0, 180.0),
    "angle_ladd1_B_deg":  (0.0, 180.0),
    "angle_lcxom1_B_deg": (0.0, 180.0),
    "length_lmca_mm":     (0.0, np.inf),
    "diameter_lad_mm":    (0.0, np.inf),
    "diameter_lcx_mm":    (0.0, np.inf),
    "path_ostium_to_D1_mm": (0.0, np.inf),
    "path_ostium_to_OM1_mm": (0.0, np.inf),
}

# =============================
# HELPERS
# =============================
def _available_keys(d):
    return [k for k in ["mean", "sd", "q1", "q2", "q3"] if k in d and d[k] is not None]

def _truncnorm_dist(mu, sig, lb, ub):
    a = (lb - mu) / sig
    b = (ub - mu) / sig if np.isfinite(ub) else np.inf
    return stats.truncnorm(a=a, b=b, loc=mu, scale=sig)

def fit_truncnorm_from_stats(stats_dict, lb, ub):
    """
    Fit (mu, sig) of the *underlying* normal such that the TRUNCATED normal
    matches the provided mean/sd/q1/q2/q3 as closely as possible.
    """
    keys = _available_keys(stats_dict)
    if "sd" not in keys:
        raise ValueError("Need at least 'sd' for fitting.")

    target = {k: float(stats_dict[k]) for k in keys}

    # initialization: start near target mean, with target sd
    mu0 = target.get("mean", target.get("q2", 0.0))
    sig0 = max(target.get("sd", 1e-6), 1e-6)

    # weights: keep sd comparable; quantiles/means on same unit scale
    scale = max(target.get("sd", sig0), 1e-6)

    def obj(theta):
        mu = float(theta[0])
        sig = float(np.exp(theta[1]))  # enforce positivity

        # avoid degenerate sig
        sig = max(sig, 1e-8)

        dist = _truncnorm_dist(mu, sig, lb, ub)

        r = []
        if "mean" in target:
            r.append((dist.mean() - target["mean"]) / scale)
        if "sd" in target:
            r.append((dist.std() - target["sd"]) / max(target["sd"], 1e-6))
        if "q1" in target:
            r.append((dist.ppf(0.25) - target["q1"]) / scale)
        if "q2" in target:
            r.append((dist.ppf(0.50) - target["q2"]) / scale)
        if "q3" in target:
            r.append((dist.ppf(0.75) - target["q3"]) / scale)

        r = np.asarray(r, dtype=float)
        return float(np.sum(r * r))

    res = minimize(obj, x0=np.array([mu0, np.log(sig0)]), method="Nelder-Mead")

    if not res.success:
        # fallback: just use target mean/sd as underlying (still truncated in plotting)
        return float(mu0), float(sig0), {"success": False, "fit": "fallback_target_mean_sd", "fun": obj([mu0, np.log(sig0)])}

    mu = float(res.x[0])
    sig = float(np.exp(res.x[1]))
    return mu, sig, {"success": True, "fit": "truncnorm_lsq_mean_sd_q123", "fun": float(res.fun)}

def make_x_grid(mu_sig_list, lb, ub):
    mus = np.array([m for m, s in mu_sig_list], dtype=float)
    sigs = np.array([s for m, s in mu_sig_list], dtype=float)

    lo = float(np.min(mus - X_PAD_SIGMA * sigs))
    hi = float(np.max(mus + X_PAD_SIGMA * sigs))

    lo = max(lo, float(lb))
    hi = min(hi, float(ub)) if np.isfinite(ub) else hi

    # if bounds make window too tight, expand within bounds
    if hi <= lo:
        lo = float(lb)
        hi = float(ub) if np.isfinite(ub) else float(lb + 10.0)

    return np.linspace(lo, hi, 1200)

def plot_group_distribution(ax, x, dist, *, color, label):
    pdf = dist.pdf(x)
    ax.plot(x, pdf, color=color, lw=LINEWIDTH_PDF, label=label)
    ax.fill_between(x, 0, pdf, color=color, alpha=ALPHA_FILL)

# =============================
# PLOTTING
# =============================
for key, cfg in STATS.items():
    lb, ub = BOUNDS.get(key, (0.0, np.inf))

    groups = cfg["groups"]
    prefer_sex = cfg.get("prefer_sex_split", False)

    # Fit per group
    fits = {}
    mu_sig = []
    for g, s in groups.items():
        mu, sig, info = fit_truncnorm_from_stats(s, lb, ub)
        dist = _truncnorm_dist(mu, sig, lb, ub)
        fits[g] = (mu, sig, dist, info)
        mu_sig.append((mu, sig))

    x = make_x_grid(mu_sig, lb, ub)

    fig, ax = plt.subplots(figsize=(5, 3), dpi=DPI)

    # ---- plot distributions ----
    if prefer_sex and ("Female" in groups and "Male" in groups):
        for g in ["Female", "Male"]:
            _, _, dist, _ = fits[g]
            c = COLOR_TWO_A if g == "Female" else COLOR_TWO_B
            plot_group_distribution(ax, x, dist, color=c, label=g)
        ax.legend(frameon=False)

    else:
        # Single pooled distribution ("All") → no legend
        g = list(groups.keys())[0]
        _, _, dist, _ = fits[g]
        plot_group_distribution(ax, x, dist, color=COLOR_SINGLE, label=None)

    if key in DESIGN:
        x0 = DESIGN[key]
        ax.axvline(x0, color=COLOR_DESIGN, lw=LINEWIDTH_DESIGN, ls="--")

        ax.annotate(
            "Design",
            xy=(x0, ax.get_ylim()[1] * 0.96),  # anchor point on the line
            xytext=(-2, 0),  # 4 points to the right, 0 up/down
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="right",
            fontsize=11,
            color=COLOR_DESIGN,
        )

    # ax.set_title(cfg["label"])
    ax.set_xlabel(f"{cfg['label']} [{cfg['unit']}]")
    ax.set_ylabel("Density")
    fig.tight_layout()
    plt.show()
    if SAVE_SVG:
        fig.savefig(
            os.path.join(OUTDIR, f"{key}.svg"),
            bbox_inches="tight",
            transparent=True,
        )

    plt.close(fig)

# Save a manifest
pd.DataFrame(
    [{"metric": k, "svg": os.path.join(OUTDIR, f"{k}.svg")} for k in STATS]
).to_csv(os.path.join(OUTDIR, "manifest.csv"), index=False)

print(f"Wrote SVGs + manifest to: {OUTDIR}")