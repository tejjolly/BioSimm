#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ── 1) Load & trim ───────────────────────────────────────────────────────────
df = pd.read_csv(
    "/data/data.csv"
)

# Keep just the rows that have a discord label
df = df.dropna(subset=["discord"])

# If you still need FFR, add it; otherwise omit this whole block -------------

df["FFR"] = np.where(df["Condition"] == "Hyperemic", df["P_d/P_a"], np.nan)

# ── 2) Choose variables to plot (adjust to taste) ───────────────────────────
vars_to_plot = [
    "P_Loss_Coeff", "HMR", "BMR/HMR",
    "FFR", "CFR", "CFR/FFR"
]

# ── 3) Quick pairplot with Seaborn ──────────────────────────────────────────
sns.pairplot(
    data=df.dropna(subset=vars_to_plot),  # keep rows with all vars present
    vars=vars_to_plot,
    hue="discord",
    diag_kind="kde"        # default is fine; change to "hist" if you prefer
)

plt.tight_layout()
plt.show()
