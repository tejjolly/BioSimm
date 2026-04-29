import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.api as sms
import pingouin as pg

# ── Existing helpers you already had ───────────────────────────────────────────
def linear_vs_quadratic_F(x, y):
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    X_lin  = sm.add_constant(x)
    X_quad = sm.add_constant(np.c_[x, x**2])
    m_lin  = sm.OLS(y, X_lin).fit()
    m_quad = sm.OLS(y, X_quad).fit()
    a = anova_lm(m_lin, m_quad)
    F, p = a.loc[1, 'F'], a.loc[1, 'Pr(>F)']
    return F, float(p), (p >= 0.05)  # True -> linear is adequate

def pearson_spearman(x, y):
    df = pd.concat([pd.Series(x, dtype=float), pd.Series(y, dtype=float)], axis=1).dropna()
    x = df.iloc[:,0].to_numpy()
    y = df.iloc[:,1].to_numpy()
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    return (r_p, p_p), (r_s, p_s), len(df)

def analyze_pair_simple(x, y, label="X vs Y"):
    F, pF, is_linear = linear_vs_quadratic_F(x, y)
    (r_p, p_p), (r_s, p_s), n = pearson_spearman(x, y)
    print(f"[{label}] Nested F-test (quad vs lin): F={F:.3g}, p={pF:.3g}  -> {'linear' if is_linear else 'nonlinear'}")
    if is_linear:
        print(f"Report Pearson (primary): R={r_p:+.3f}, p={p_p:.3g}, n={n}; Spearman (robustness): ρ={r_s:+.3f}, p={p_s:.3g}")
        return {"primary": ("pearson", r_p, p_p, n), "secondary": ("spearman", r_s, p_s, n)}
    else:
        print(f"Report Spearman (primary): ρ={r_s:+.3f}, p={p_s:.3g}, n={n}; Pearson (comparison): r={r_p:+.3f}, p={p_p:.3g}")
        return {"primary": ("spearman", r_s, p_s, n), "secondary": ("pearson", r_p, p_p, n)}

# ── New: local sensitivity within clinical bands ───────────────────────────────
def local_sensitivity_band(df,
                           x_var="HMR",
                           y_var="HSR",
                           z_var="P_d/P_a",          # 'P_d/P_a' for FFR or 'CFR'
                           condition="Hyperemic",
                           exclude_no_stenosis=True,
                           stenosis_col="Stenosis Percentage",
                           location=None):
    """
    Computes local standardized sensitivities β_x, β_y of z_var to x_var,y_var
    inside a narrow band around the clinical threshold:
      * FFR (P_d/P_a): 0.80 ± 0.04
      * CFR:           2.00 ± 0.25
    Uses conventional SEs if residuals pass normality + homoscedasticity,
    otherwise HC3 robust SEs. Prints n, which SEs were used, and p-values.
    Returns a dict with betas, CIs, p-values, etc.
    """
    # threshold & band by outcome
    if z_var == "CFR":
        thresh, tol = 2.0, 0.50
    else:  # treat anything else as FFR band (e.g., 'P_d/P_a')
        thresh, tol = 0.80, 0.05

    # base filter
    dff = df.loc[
        (df["Condition"] == condition)
        & df[x_var].notna()
        & df[y_var].notna()
        & df[z_var].notna()
    ].copy()

    if location is not None and "Location" in dff.columns:
        dff = dff.loc[dff["Location"] == location]

    # near-threshold slice
    near = dff.loc[(dff[z_var] - thresh).abs() <= tol].copy()

    if exclude_no_stenosis and (stenosis_col in near.columns):
        near = near.loc[near[stenosis_col] >= 0.05]

    n = len(near)
    # if n < 8:
    #     print(f"[Local sensitivity near {z_var}={thresh:.2f}±{tol}] Not enough points (n={n}).")
    #     return None

    # standardize predictors
    X1 = (near[x_var] - near[x_var].mean()) / near[x_var].std(ddof=1)
    X2 = (near[y_var] - near[y_var].mean()) / near[y_var].std(ddof=1)
    Y  = near[z_var].to_numpy(float)

    X_sm = sm.add_constant(np.c_[X1.to_numpy(float), X2.to_numpy(float)])

    # plain OLS for residuals / R², etc.
    m_ols = sm.OLS(Y, X_sm).fit()

    # Residual checks
    W_resid, p_norm = stats.shapiro(m_ols.resid) if len(m_ols.resid) <= 5000 else (np.nan, np.nan)
    bp_lm, p_bp, bp_f, p_bp_f = sms.het_breuschpagan(m_ols.resid, m_ols.model.exog)
    white_lm, p_white, _, _   = sms.het_white(m_ols.resid, m_ols.model.exog)

    # Decide inference SEs
    use_hc3 = False
    if not np.isnan(p_norm) and p_norm < 0.05:
        use_hc3 = True
    if p_bp < 0.05 or p_white < 0.05:
        use_hc3 = True

    m = sm.OLS(Y, X_sm).fit(cov_type="HC3") if use_hc3 else m_ols
    cov_label = "HC3 robust SEs" if use_hc3 else "conventional SEs"

    # coefficients: const, X1, X2  (X1,X2 are standardized)
    b0, b1_std, b2_std = m.params
    ci = m.conf_int(alpha=0.05)
    ci_b1_std = ci[1]  # [low, high]
    ci_b2_std = ci[2]

    # NEW: p-values (two-sided) and t-stats
    p_b1 = float(m.pvalues[1])
    p_b2 = float(m.pvalues[2])
    t_b1 = float(m.tvalues[1])
    t_b2 = float(m.tvalues[2])

    # Fit stats from plain OLS
    resid = Y - m_ols.fittedvalues
    rss = float(np.sum(resid**2))
    sst = float(np.sum((Y - Y.mean())**2))
    r2  = 1 - rss/sst if sst > 0 else np.nan
    p   = X_sm.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
    rmse = float(np.sqrt(rss / n))

    # Print concise summary WITH p-values
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
        "betas_std": {"const": b0, x_var: b1_std, y_var: b2_std},
        "ci_std": {x_var: tuple(ci_b1_std), y_var: tuple(ci_b2_std)},
        "pvals": {x_var: p_b1, y_var: p_b2},        # <-- added
        "tvals": {x_var: t_b1, y_var: t_b2},        # <-- added (optional)
        "tests": {"shapiro_p": p_norm, "bp_p": p_bp, "white_p": p_white},
        "used": cov_label,
        "r2": r2, "adj_r2": adj_r2, "rmse": rmse,
        "slice": near
    }



# ── Load once and use ──────────────────────────────────────────────────────────
df_full = pd.read_csv('../../data/data_manuscript.csv')
df_full = df_full[df_full['Condition'] == 'Hyperemic']

# Example: correlation block you already had
x_variable = 'HMR'
y_variable = ('WSS_TE_Area_min')   # FFR
res = analyze_pair_simple(df_full[x_variable], df_full[y_variable], label=f"{y_variable} vs. {x_variable}")
print('')
res_partial = pg.partial_corr(data=df_full,
                x=x_variable,
                y=y_variable,
                covar='Q_distal',
                method='pearson')
print(res_partial)
print('')
# Local sensitivities near FFR=0.80±0.10 (uses HMR & HSR as predictors by default)
res_ffr = local_sensitivity_band(
    df_full,
    x_var="HMR",
    y_var="HSR",
    z_var="P_d/P_a",        # for CFR, set z_var="CFR"
    condition="Hyperemic",
    exclude_no_stenosis=True,
    stenosis_col="Stenosis Percentage",
    location=None           # e.g., "LAD" to restrict
)
print('')
# If you also want the CFR band:
res_cfr = local_sensitivity_band(
    df_full,
    x_var="HMR",
    y_var="HSR",
    z_var="CFR",
    condition="Hyperemic",
    exclude_no_stenosis=True
)

# print("FFR p-values:", res_ffr["pvals"])  # {'HMR': 0.000358..., 'HSR': 0.000802...}

print("CFR p-values:", res_cfr["pvals"])  # {'HMR': 0.289..., 'HSR': 0.0217...}
