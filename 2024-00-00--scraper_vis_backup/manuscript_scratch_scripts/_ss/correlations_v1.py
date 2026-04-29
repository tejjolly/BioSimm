import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

def linear_vs_quadratic_F(x, y):
    """Returns (F, p, is_linear_bool) using nested OLS models."""
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    X_lin  = sm.add_constant(x)
    X_quad = sm.add_constant(np.c_[x, x**2])
    m_lin  = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    a = anova_lm(m_lin, m_quad)  # row 1 is the comparison
    F, p = a.loc[1, 'F'], a.loc[1, 'Pr(>F)']
    return F, float(p), (p >= 0.05)  # True -> treat as linear

def pearson_spearman(x, y):
    """Computes Pearson r,p and Spearman rho,p on aligned (non-NA) data."""
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return (r_p, p_p), (r_s, p_s), len(df)

def ols_with_residual_checks(x, y):
    """OLS for sensitivity bands; reports residual Shapiro and robust HC3 model."""
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    # Residual normality (OLS-only)
    W_resid, p_resid = stats.shapiro(m.resid) if len(m.resid) <= 5000 else (np.nan, np.nan)
    # Heteroscedasticity checks (on plain OLS residuals)
    import statsmodels.stats.api as sms
    bp_lm, bp_p, bp_f, bp_fp = sms.het_breuschpagan(m.resid, m.model.exog)
    white_lm, white_p, _, _   = sms.het_white(m.resid, m.model.exog)
    # Robust HC3 model for inference if needed
    m_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
    return {
        "ols": m, "ols_hc3": m_hc3,
        "shapiro_resid": (W_resid, p_resid),
        "bp_p": bp_p, "white_p": white_p
    }

def analyze_pair_simple(x, y, label="X vs Y"):
    # 1) Formal linearity test to choose Pearson vs Spearman
    F, pF, is_linear = linear_vs_quadratic_F(x, y)
    (r_p, p_p), (r_s, p_s), n = pearson_spearman(x, y)

    print(f"[{label}] Nested F-test (quad vs lin): F={F:.3g}, p={pF:.3g}  -> {'linear' if is_linear else 'nonlinear'}")
    if is_linear:
        print(f"Report Pearson (primary): r={r_p:+.3f}, p={p_p:.3g}, n={n}; Spearman (robustness): ρ={r_s:+.3f}, p={p_s:.3g}")
        return {"primary": ("pearson", r_p, p_p, n), "secondary": ("spearman", r_s, p_s, n)}
    else:
        print(f"Report Spearman (primary): ρ={r_s:+.3f}, p={p_s:.3g}, n={n}; Pearson (comparison): r={r_p:+.3f}, p={p_p:.3g}")
        return {"primary": ("spearman", r_s, p_s, n), "secondary": ("pearson", r_p, p_p, n)}

# Load and filter once
df_full = pd.read_csv('/data/data_manuscript.csv')
df_full = df_full[df_full['Condition'] == 'Hyperemic']

y_variable = 'P_d/P_a'
x_variable = 'HMR'

# Correlation (primary decided by F-test)
res = analyze_pair_simple(df_full[x_variable], df_full[y_variable], label=f"{y_variable} vs. {x_variable}")

# If you also want OLS (threshold-band sensitivities) with residual checks
ols_res = ols_with_residual_checks(df_full[x_variable], df_full[y_variable])
print("Shapiro–Wilk residual p:", ols_res["shapiro_resid"][1], "| White p:", ols_res["white_p"])
print(ols_res["ols_hc3"].summary())
