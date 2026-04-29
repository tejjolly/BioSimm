import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.stats import pearsonr

# Load data
df = pd.read_csv('/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/summary2.csv')

# Standardize column names to lowercase
df.columns = df.columns.str.lower()

# Drop rows where HMR > 5
df = df[df['hmr'] <= 5]

# Include only columns that contain 'wss'
wss_cols = [col for col in df.columns if 'wss' in col]
df = df[['hsr', 'hmr'] + wss_cols]

# Drop specific WSS metrics
to_drop = ['wss_avg_area_min', 'wss', 'wss_avg_area', 'wss_min']
df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

# Normalize _min and _max columns (excluding area_min and area_max) by wss_lmb
if 'wss_lmb' in df.columns:
    for col in df.columns:
        if (
            ('_min' in col or '_max' in col)
            and 'area_min' not in col and 'area_max' not in col
            and col != 'wss_lmb'
        ):
            df[col] = df[col] / df['wss_lmb']

    df.drop(columns='wss_lmb', inplace=True)

# Convert to numeric and drop strings
df = df.apply(pd.to_numeric, errors='coerce')

# Filter out NaNs or Infs
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =====================================================================
# REGRESSION AND PARTIAL CORRELATIONS
# =====================================================================
beta_data = []
partial_corr_data = []

# Loop through each WSS metric
for metric in [col for col in df.columns if col not in ['hsr', 'hmr']]:
    sub_df = df[['hsr', 'hmr', metric]].dropna()

    if len(sub_df) < 30:
        continue

    # Standardize predictors and response
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub_df[['hsr', 'hmr']])
    y_scaled = scaler.fit_transform(sub_df[[metric]])

    X_model = sm.add_constant(X_scaled)
    model = sm.OLS(y_scaled, X_model).fit()

    beta_data.append({
        'metric': metric,
        'beta_hsr': abs(model.params[1]),
        'beta_hmr': abs(model.params[2]),
        'r_squared': model.rsquared
    })

    # Partial correlations
    res_hsr = sm.OLS(sub_df['hsr'], sm.add_constant(sub_df['hmr'])).fit()
    res_metric_hsr = sm.OLS(sub_df[metric], sm.add_constant(sub_df['hmr'])).fit()
    partial_hsr = pearsonr(res_hsr.resid, res_metric_hsr.resid)[0]

    res_hmr = sm.OLS(sub_df['hmr'], sm.add_constant(sub_df['hsr'])).fit()
    res_metric_hmr = sm.OLS(sub_df[metric], sm.add_constant(sub_df['hsr'])).fit()
    partial_hmr = pearsonr(res_hmr.resid, res_metric_hmr.resid)[0]

    partial_corr_data.append({
        'metric': metric,
        'partial_hsr': abs(partial_hsr),
        'partial_hmr': abs(partial_hmr)
    })

# Convert to DataFrames
beta_df = pd.DataFrame(beta_data).sort_values('metric')
partial_df = pd.DataFrame(partial_corr_data).sort_values('metric')

# Define desired metric order
custom_order = [
    'wss_bif',
    'wss_le',
    'wss_te',
    'wss_area_bifur',
    'wss_le_area',
    'wss_te_area',
    'wss_le_min',
    'wss_te_min',
    'wss_le_area_min',
    'wss_te_area_min'
]

# Apply the order to both DataFrames
beta_df = beta_df.set_index('metric').loc[custom_order].reset_index()
partial_df = partial_df.set_index('metric').loc[custom_order].reset_index()

# =====================================================================
# PLOTTING: BAR PLOTS FOR REGRESSION COEFFS & PARTIAL CORRELATIONS
# =====================================================================
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(beta_df))
plt.bar(index, beta_df['beta_hsr'], bar_width, label='HSR')
plt.bar(index + bar_width, beta_df['beta_hmr'], bar_width, label='HMR')
plt.xticks(index + bar_width / 2, beta_df['metric'], rotation=45, ha='right')
plt.ylabel('Standardized Beta Coefficient')
plt.title('Standardized Beta Coefficients')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(index, partial_df['partial_hsr'], bar_width, label='HSR')
plt.bar(index + bar_width, partial_df['partial_hmr'], bar_width, label='HMR')
plt.xticks(index + bar_width / 2, partial_df['metric'], rotation=45, ha='right')
plt.ylabel('Partial Correlation')
plt.title('Partial Correlations')
plt.legend()
plt.tight_layout()
plt.show()

# Print summary tables
print("\n=== Standardized Regression Coefficients ===")
print(beta_df[['metric', 'r_squared', 'beta_hsr', 'beta_hmr']].to_string(index=False, float_format='%.3f'))

print("\n=== Partial Correlations ===")
print(partial_df[['metric', 'partial_hsr', 'partial_hmr']].to_string(index=False, float_format='%.3f'))

# =====================================================================
# NEW SECTION: 2D SCATTER PLOTS FOR HSR vs HMR, COLORED BY EACH WSS
# =====================================================================
# We'll create a separate figure for each WSS metric
# scaler = StandardScaler()
df[['hsr', 'hmr']] = scaler.fit_transform(df[['hsr', 'hmr']])

wss_metrics = [col for col in df.columns if col not in ['hsr', 'hmr']]



for metric in wss_metrics:
    sub_df = df[['hsr', 'hmr', metric]].dropna()
    if len(sub_df) == 0:
        continue

    plt.figure()
    scatter = plt.scatter(
        x=sub_df['hsr'],
        y=sub_df['hmr'],
        c=sub_df[metric],
        s=50,  # point size
        alpha=0.8  # transparency
    )
    plt.colorbar(scatter, label=metric)
    plt.xlabel('HSR')
    plt.ylabel('HMR')
    plt.title(f'HSR vs. HMR (Colored by {metric})')
    plt.tight_layout()
    plt.show()
