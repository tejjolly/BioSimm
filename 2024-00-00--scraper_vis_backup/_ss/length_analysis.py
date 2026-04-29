import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact


def plot_ffr_cfr_discordance_stacked(
    df_or_csv,
    *,
    condition_col="Condition",
    hyperemic_value="Hyperemic",
    ffr_col="P_d/P_a",
    cfr_col="CFR",
    length_col="Length",
    focal_threshold=1.5,
    ffr_thresh=0.80,
    cfr_thresh=2.0,
    use_test="chi2",   # "chi2" or "fisher"
    save_path=None,    # e.g., "ffr_cfr_discordance.svg"
):
    # ---- Load ----
    df = pd.read_csv(df_or_csv) if isinstance(df_or_csv, str) else df_or_csv.copy()

    # ---- Filter to rows that actually have CFR/FFR (usually hyperemic) ----
    if condition_col in df.columns:
        df = df[df[condition_col].astype(str).str.strip().eq(hyperemic_value)]

    # ---- Coerce numerics + drop missing ----
    for col in [ffr_col, cfr_col, length_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[ffr_col, cfr_col, length_col])

    # ---- Lesion type ----
    df["lesion_type"] = np.where(df[length_col] < focal_threshold, "focal lesion", "diffuse lesion")

    # ---- Discordance groups (define + as "abnormal") ----
    ffr_pos = df[ffr_col] <= ffr_thresh          # FFR+
    cfr_pos = df[cfr_col] < cfr_thresh           # CFR+

    # Keep only discordant:
    #   FFR+/CFR- : abnormal FFR, normal CFR
    #   FFR-/CFR+ : normal FFR, abnormal CFR
    # ---- Discordance groups (define + as "abnormal") ----
    ffr_pos = df[ffr_col] <= ffr_thresh  # FFR+
    cfr_pos = df[cfr_col] < cfr_thresh  # CFR+

    # Use pd.NA so the result stays "string-like" without NumPy float promotion issues
    df["group"] = np.where(
        ffr_pos & (~cfr_pos), "FFR+/CFR-\n",
        np.where((~ffr_pos) & cfr_pos, "FFR-/CFR+\n", pd.NA)
    )
    df = df.dropna(subset=["group"])

    # ---- Counts ----
    # rows: lesion_type, cols: group
    ct = pd.crosstab(df["lesion_type"], df["group"])
    # Ensure both lesion categories exist (even if 0)
    for lt in ["diffuse lesion", "focal lesion"]:
        if lt not in ct.index:
            ct.loc[lt] = 0
    ct = ct.loc[["diffuse lesion", "focal lesion"]]  # order like your example

    # Ensure both discordance groups exist (even if 0)
    for g in ["FFR-/CFR+\n", "FFR+/CFR-\n"]:
        if g not in ct.columns:
            ct[g] = 0
    # Put in a consistent left-to-right order (edit if you prefer the opposite)
    ct = ct[["FFR-/CFR+\n", "FFR+/CFR-\n"]]

    # ---- p-value from 2x2 table ----
    table_2x2 = ct.to_numpy()
    if use_test.lower() == "fisher":
        # Fisher expects [[a,b],[c,d]] but ordering doesn't matter for p
        _, pval = fisher_exact(table_2x2)
        p_label = "Fisher exact"
    else:
        chi2, pval, dof, exp = chi2_contingency(table_2x2)
        p_label = "Chi-square"

    # ---- Convert to percentages per column ----
    col_totals = ct.sum(axis=0).replace(0, np.nan)
    pct = (ct / col_totals) * 100

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(4, 5), dpi=600)

    x = np.arange(len(ct.columns))
    bottom = np.zeros(len(ct.columns))

    colors = {
        "diffuse lesion": "#5B8E94",  # teal-ish
        "focal lesion":   "#5A2D5C",  # purple-ish
    }

    for lesion in ["diffuse lesion", "focal lesion"]:
        heights = pct.loc[lesion].to_numpy()
        ax.bar(x, heights, bottom=bottom, color=colors[lesion], label=lesion)

        # segment labels: "count/total"
        for i, (h, btm) in enumerate(zip(heights, bottom)):
            if np.isnan(h) or h <= 0:
                continue
            count = int(ct.loc[lesion, ct.columns[i]])
            total = int(ct.iloc[:, i].sum())
            ax.text(
                x[i], btm + h/2,
                f"{count}/{total}",
                ha="center", va="center",
                color="white", fontsize=14, fontweight="bold"
            )

        bottom += np.nan_to_num(heights)

    # Axis formatting
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}(n={int(ct[g].sum())})" for g in ct.columns], fontsize=13)
    ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=13)


    # p-value text (match paper style)
    if pval < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {pval:.9g}"
    ax.set_title(p_text, fontsize=12, pad=8)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)

    plt.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax, ct, pval, p_label


df = pd.read_csv("../data/data_manuscript.csv")
# df = df[df['Condition']=='Hyperemic']
fig, ax, ct, pval, testname = plot_ffr_cfr_discordance_stacked(
    df,
    focal_threshold=1.5,
    ffr_thresh=0.80,
    cfr_thresh=2.0,
    use_test="chi2",           # chi2 or "fisher"
    save_path="ffr_cfr_discordance.svg",
)
print(ct)
print(testname, pval)