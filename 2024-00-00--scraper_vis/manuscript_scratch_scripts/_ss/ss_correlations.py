#!/usr/bin/env python3
import numpy as np
import pandas as pd
from math import isfinite

# -----------------------------
# CONFIG (edit as needed)
# -----------------------------
csv_path = "/data/data_manuscript.csv"

# Basic filters
hyperemic_only = True       # keep only hyperemic rows (for FFR analyses)
lad_only = False            # keep only LAD rows
no_stenosis_threshold = -1  # drop stenosis% <= this (0.05 = 5%); set to 0.0 to include all
drop_wss = True             # drop all WSS* columns to simplify the numeric set

# Outcome band for local correlations
use_ffr_band = True         # True → use FFR~0.80±tol; False → use CFR~2.0±tol
ffr_center, ffr_tol = 0.80, 0.05
cfr_center, cfr_tol = 2.00, 0.50

# Reporting
strong_corr_threshold = 0.40  # |r| > this will be printed from full corr matrix

# -----------------------------
# LOAD & PREP
# -----------------------------
df = pd.read_csv(csv_path)

# Keep string labels for these
if 'Condition' in df.columns:
    df['Condition'] = df['Condition'].astype(str)
if 'Location' in df.columns:
    df['Location'] = df['Location'].astype(str)
if 'source' in df.columns:
    df['source'] = df['source'].astype(str)

# FFR column convenience
if 'P_d/P_a' in df.columns and 'FFR' not in df.columns:
    df = df.rename(columns={'P_d/P_a': 'FFR'})



if hyperemic_only and 'Condition' in df.columns:
    df = df[df['Condition'] == 'Hyperemic']

if lad_only and 'Location' in df.columns:
    df = df[df['Location'] == 'LAD']

if 'Stenosis Percentage' in df.columns and no_stenosis_threshold is not None:
    df = df[df['Stenosis Percentage'] > no_stenosis_threshold]

# Drop clearly unused columns to simplify correlations
cols_to_drop = ['Condition', 'Geometry Number', 'R_micro', 'R_scale', 'discord', 'R_total']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

if drop_wss:
    df = df.drop(columns=df.filter(like='WSS').columns, errors='ignore')

# Keep only numeric for correlation matrix later
df_num = df.select_dtypes(include=[np.number]).copy()

# -----------------------------
# HELPERS
# -----------------------------
def safe_corr(a, b):
    """Pearson r with NaN handling; returns np.nan if insufficient data."""
    s = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(s) < 3:
        return np.nan, len(s)
    r = s['a'].corr(s['b'])
    return float(r), len(s)

def partial_corr(x, y, z):
    """Partial correlation r_xy.z via residuals (OLS) with NaN handling."""
    D = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    n = len(D)
    if n < 5:
        return np.nan, n
    X = np.c_[np.ones(n), D['z'].to_numpy()]
    # regress x ~ z
    bx, *_ = np.linalg.lstsq(X, D['x'].to_numpy(), rcond=None)
    rx = D['x'].to_numpy() - X @ bx
    # regress y ~ z
    by, *_ = np.linalg.lstsq(X, D['y'].to_numpy(), rcond=None)
    ry = D['y'].to_numpy() - X @ by
    r = np.corrcoef(rx, ry)[0, 1]
    return float(r), n

def describe_range(name, s):
    s = pd.Series(s).dropna()
    if s.empty:
        return f"{name}: [NA, NA] (n=0)"
    return f"{name}: [{s.min():.3f}, {s.max():.3f}] (n={len(s)})"

# -----------------------------
# CORE CHECKS
# -----------------------------
needed = ['HMR', 'HSR']
missing = [c for c in needed if c not in df_num.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

print("=== Dataset after filters ===")
print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
if 'Stenosis Percentage' in df.columns:
    print(describe_range("Stenosis%", df['Stenosis Percentage']))
print(describe_range("HMR", df['HMR']))
print(describe_range("HSR", df['HSR']))
if 'FFR' in df_num.columns:
    print(describe_range("FFR", df_num['FFR']))
if 'CFR' in df_num.columns:
    print(describe_range("CFR", df_num['CFR']))
print()

# 1) Global correlation HMR↔HSR
r_global, n_global = safe_corr(df_num['HMR'], df_num['HSR'])
print("1) Global corr(HMR, HSR): "
      f"r = {r_global:+.3f}  (n={n_global})")

# 2) Local band correlation (FFR or CFR)
if use_ffr_band and 'FFR' in df_num.columns:
    band = df_num[np.abs(df_num['FFR'] - ffr_center) <= ffr_tol]
    r_band, n_band = safe_corr(band['HMR'], band['HSR'])
    print(f"2) Local corr near FFR≈{ffr_center}±{ffr_tol}: "
          f"r = {r_band:+.3f}  (n={n_band})")
elif (not use_ffr_band) and 'CFR' in df_num.columns:
    band = df_num[np.abs(df_num['CFR'] - cfr_center) <= cfr_tol]
    r_band, n_band = safe_corr(band['HMR'], band['HSR'])
    print(f"2) Local corr near CFR≈{cfr_center}±{cfr_tol}: "
          f"r = {r_band:+.3f}  (n={n_band})")
else:
    print("2) Local band corr: outcome column not available.")
print()

# 3) Stratify by FFR bins (if available)
if 'FFR' in df_num.columns:
    bins = [0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["[0.60,0.70)", "[0.70,0.80)", "[0.80,0.90)", "[0.90,1.00]"]
    tmp = df_num[['HMR', 'HSR', 'FFR']].dropna().copy()
    tmp['FFR_bin'] = pd.cut(tmp['FFR'], bins=bins, labels=labels, right=False, include_lowest=True)
    print("3) corr(HMR,HSR) by FFR bin:")
    for lab, g in tmp.groupby('FFR_bin', dropna=False):
        r, n = safe_corr(g['HMR'], g['HSR'])
        print(f"   {str(lab):>12}: r = {r:+.3f} (n={n})")
    print()

# 4) Stratify by stenosis% bins (if available)
if 'Stenosis Percentage' in df.columns:
    sbins = [0.00, 0.05, 0.20, 0.40, 0.60, 0.80, 1.01]
    slabels = ["[0,5%)","[5,20%)","[20,40%)","[40,60%)","[60,80%)","[80,100%]"]
    tmp = df[['HMR','HSR','Stenosis Percentage']].dropna().copy()
    tmp['sten_bin'] = pd.cut(tmp['Stenosis Percentage'], bins=sbins, labels=slabels, right=False, include_lowest=True)
    print("4) corr(HMR,HSR) by Stenosis% bin:")
    for lab, g in tmp.groupby('sten_bin', dropna=False):
        r, n = safe_corr(g['HMR'], g['HSR'])
        print(f"   {str(lab):>10}: r = {r:+.3f} (n={n})")
    print()

# 5) Partial correlation corr(HMR,HSR | FFR or CFR)
if 'FFR' in df_num.columns and use_ffr_band:
    r_pc, n_pc = partial_corr(df_num['HMR'], df_num['HSR'], df_num['FFR'])
    print(f"5) Partial corr(HMR, HSR | FFR): r = {r_pc:+.3f} (n={n_pc})")
elif 'CFR' in df_num.columns and not use_ffr_band:
    r_pc, n_pc = partial_corr(df_num['HMR'], df_num['HSR'], df_num['CFR'])
    print(f"5) Partial corr(HMR, HSR | CFR): r = {r_pc:+.3f} (n={n_pc})")
else:
    print("5) Partial corr: conditioning variable not available.")
print()

# 6) Full numeric correlation matrix & strongest pairs
if df_num.shape[1] >= 2:
    corr_mat = df_num.corr()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    strong = upper.stack().sort_values(key=lambda s: s.abs(), ascending=False)
    strong = strong[np.abs(strong) > strong_corr_threshold]
    print(f"6) Strongest absolute correlations |r| > {strong_corr_threshold}:")
    if strong.empty:
        print("   (none)")
    else:
        for (a, b), r in strong.items():
            print(f"   {a:20} <-> {b:20} | r = {r:+.3f}")
else:
    print("6) Not enough numeric columns for a correlation matrix.")
