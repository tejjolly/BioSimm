import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
import matplotlib.pyplot as plt
import os, pathlib
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

# ─── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(
    "/data/data.csv"
)
var = "P_Loss_Coeff"
data = df_raw[df_raw['source']=='mine']
data = data[data['Condition']=='Hyperemic']
# data = data[data['R_total']==0.24]
data = data[var].dropna()

# ─── ROUND TO 5% INCREMENTS & CLAMP AT 60% ─────────────────────────────────────
# data = (data / 0.05).round() * 0.05     # snap to nearest 0.05
# data[data > 0.60] = 0.60                # any >60% → 60%

# ─── Plot histogram + KDE ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

bin_width = 0.1
bins = np.arange(data.min(), data.max() + bin_width, bin_width)

sns.histplot(
    data,
    bins=bins,
    stat="count",
    alpha=0.8,
    color='#5E9096',
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

kde = gaussian_kde(data)
x_vals = np.linspace(bins[0], bins[-1], 500)                # align with bins
y_vals = kde(x_vals) * len(data) * bin_width
ax.plot(x_vals, y_vals, color='black', linewidth=1)

# ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax.set_title(f"{var}")
# ax.set_xlabel("$ζ_{L}$")
# ax.set_xlabel(var)
ax.set_ylabel("Count")
ax.grid(False)

plt.tight_layout()
plt.show()

# ─── PRINT SORTED PERCENTAGES ─────────────────────────────────────────────────
print("Unique values:")
print(sorted(data.unique()))
