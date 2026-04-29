#!/usr/bin/env python3
"""
Local sensitivity (near FFR=0.80) using SALib Sobol on a quadratic surrogate.
No plots. Print-only. Throwaway script.
"""
# pip install SALib


import numpy as np
import pandas as pd
import sys

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/data/data_manuscript.csv"

SOURCE_EQ = "mine"          # set to None to skip
HYPEREMIC_ONLY = True       # keep only hyperemic rows (FFR context)
LAD_ONLY = False            # keep only LAD (set True if desired)
NO_STENOSIS_THRESHOLD = 0.05  # drop stenosis% <= this (0.05 = 5%)

# Local band around decision threshold
FFR_CENTER = 0.80
FFR_TOL = 0.1              # try 0.04–0.06 for robustness

# Sobol sample size (Saltelli requires N * (2d + 2) evals; d=2 here)
SALTELLI_N = 1000           # small is fine for d=2; raise if you want tighter CIs
SEED = 123

# Quantile-based local bounds from band data (with padding)
LOW_Q, HI_Q = 0.10, 0.90    # central 80% of band
PAD_FRAC = 0.05             # expand bounds by 5% of span

# Surrogate terms to include
INCLUDE_INTERACTION = False  # set True to add HMR*HSR term
INCLUDE_QUADRATIC = True     # quadratic terms HMR^2, HSR^2

# -----------------------------
# IMPORT SALib
# -----------------------------
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
except Exception as e:
    sys.stderr.write(
        "ERROR: SALib not available. Install with: pip install SALib\n"
    )
    raise

# -----------------------------
# LOAD & FILTER DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# rename for convenience
if 'P_d/P_a' in df.columns and 'FFR' not in df.columns:
    df = df.rename(columns={'P_d/P_a': 'FFR'})

# basic filters
if SOURCE_EQ is not None and 'source' in df.columns:
    df = df[df['source'].astype(str) == SOURCE_EQ]

if HYPEREMIC_ONLY and 'Condition' in df.columns:
    df = df[df['Condition'].astype(str) == 'Hyperemic']

if LAD_ONLY and 'Location' in df.columns:
    df = df[df['Location'].astype(str) == 'LAD']

if 'Stenosis Percentage' in df.columns and NO_STENOSIS_THRESHOLD is not None:
    df = df[df['Stenosis Percentage'] > NO_STENOSIS_THRESHOLD]

# keep only needed columns and drop NaNs
need_cols = ['FFR', 'HMR', 'HSR']
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

df = df[need_cols].dropna()
if df.empty:
    raise RuntimeError("No data left after filtering.")

# -----------------------------
# DEFINE LOCAL BAND
# -----------------------------
band = df[np.abs(df['FFR'] - FFR_CENTER) <= FFR_TOL].copy()
if len(band) < 8:
    raise RuntimeError(f"Not enough points in local band around {FFR_CENTER}±{FFR_TOL}. n={len(band)}")

# local stats
n_band = len(band)
hmr_rng = (band['HMR'].min(), band['HMR'].max())
hsr_rng = (band['HSR'].min(), band['HSR'].max())

print(f"[INFO] Local band FFR≈{FFR_CENTER}±{FFR_TOL}: n={n_band}")
print(f"       HMR range {hmr_rng[0]:.3f}–{hmr_rng[1]:.3f} | HSR range {hsr_rng[0]:.3f}–{hsr_rng[1]:.3f}")

# -----------------------------
# FIT QUADRATIC SURROGATE IN BAND
# -----------------------------
H = band['HMR'].to_numpy(float)
S = band['HSR'].to_numpy(float)
Y = band['FFR'].to_numpy(float)

# Design matrix
X_cols = [np.ones_like(H), H, S]
if INCLUDE_INTERACTION:
    X_cols.append(H * S)
if INCLUDE_QUADRATIC:
    X_cols.extend([H**2, S**2])
X = np.column_stack(X_cols)

# OLS
beta, *_ = np.linalg.lstsq(X, Y, rcond=None)

def ffr_hat(hmr, hsr):
    """Evaluate surrogate FFR(HMR,HSR)."""
    terms = [np.ones_like(hmr), hmr, hsr]
    if INCLUDE_INTERACTION:
        terms.append(hmr * hsr)
    if INCLUDE_QUADRATIC:
        terms.extend([hmr**2, hsr**2])
    mat = np.column_stack(terms)
    return (mat @ beta)

# Quick in-band fit quality
yhat = X @ beta
resid = Y - yhat
sse = float(np.sum(resid**2))
sst = float(np.sum((Y - Y.mean())**2))
r2 = 1 - sse / sst if sst > 0 else np.nan
print(f"[INFO] In-band surrogate fit: R^2 = {r2:.3f}, RMSE = {np.sqrt(sse/len(Y)):.4f}")

# -----------------------------
# LOCAL BOUNDS FOR SALib (from band quantiles + padding)
# -----------------------------
hmr_lo, hmr_hi = np.quantile(H, [LOW_Q, HI_Q])
hsr_lo, hsr_hi = np.quantile(S, [LOW_Q, HI_Q])

# pad bounds a bit
hmr_span = hmr_hi - hmr_lo
hsr_span = hsr_hi - hsr_lo
hmr_lo -= PAD_FRAC * hmr_span
hmr_hi += PAD_FRAC * hmr_span
hsr_lo -= PAD_FRAC * hsr_span
hsr_hi += PAD_FRAC * hsr_span

# sanity: ensure bounds are valid numbers and lo<hi
if not np.isfinite([hmr_lo, hmr_hi, hsr_lo, hsr_hi]).all() or hmr_lo >= hmr_hi or hsr_lo >= hsr_hi:
    raise RuntimeError("Invalid local bounds computed from band; adjust quantiles or padding.")

problem = {
    'num_vars': 2,
    'names': ['HMR', 'HSR'],
    'bounds': [[float(hmr_lo), float(hmr_hi)],
               [float(hsr_lo), float(hsr_hi)]]
}
print(f"[INFO] Local Sobol bounds: HMR[{hmr_lo:.3f},{hmr_hi:.3f}], HSR[{hsr_lo:.3f},{hsr_hi:.3f}]")

# -----------------------------
# SALib SAMPLING & ANALYSIS
# -----------------------------
rng = np.random.default_rng(SEED)
# SALib uses numpy's RNG internally; we still set a seed for consistency
X_samp = saltelli.sample(problem, SALTELLI_N, calc_second_order=True)

# Evaluate surrogate on samples
H_s = X_samp[:, 0]
S_s = X_samp[:, 1]
Y_s = ffr_hat(H_s, S_s)

Si = sobol.analyze(problem, Y_s, calc_second_order=True, print_to_console=False)

def _fmt_index(name_list, vals):
    return {name_list[i]: float(vals[i]) for i in range(len(name_list))}

names = problem['names']
S1 = _fmt_index(names, Si['S1'])
ST = _fmt_index(names, Si['ST'])
S1_conf = _fmt_index(names, Si['S1_conf'])
ST_conf = _fmt_index(names, Si['ST_conf'])
S2 = {f"{names[i]}×{names[j]}": float(Si['S2'][i, j])
      for i in range(len(names)) for j in range(i+1, len(names))}
S2_conf = {f"{names[i]}×{names[j]}": float(Si['S2_conf'][i, j])
           for i in range(len(names)) for j in range(i+1, len(names))}

print("\n=== Local Sobol (near FFR≈0.80) on quadratic surrogate ===")
print(f"S1 (first-order):     {S1}")
print(f"S1_conf:              {S1_conf}")
print(f"ST (total-order):     {ST}")
print(f"ST_conf:              {ST_conf}")
print(f"S2 (interaction):     {S2}")
print(f"S2_conf:              {S2_conf}")

# -----------------------------
# OPTIONAL: Local linear standardized slopes (cross-check)
# -----------------------------
# z-score H and S within the band to compare magnitudes
H_z = (H - H.mean()) / (H.std(ddof=0) + 1e-12)
S_z = (S - S.mean()) / (S.std(ddof=0) + 1e-12)
X_lin = np.column_stack([np.ones_like(H_z), H_z, S_z])
beta_lin, *_ = np.linalg.lstsq(X_lin, Y, rcond=None)
print("\n[Cross-check] Local standardized linear slopes in band:")
print(f"  HMR: {beta_lin[1]:+.3f}, HSR: {beta_lin[2]:+.3f} (intercept {beta_lin[0]:+.3f})")
