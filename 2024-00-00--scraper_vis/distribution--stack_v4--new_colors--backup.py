import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

FONT_SIZE = 20
manuscript_simulations_only = True
my_simulations_only = False
kde_overlay = True

def bins_anchored_at_edge(data_min, data_max, thresh, bin_w):
    lo = thresh + bin_w * np.floor((data_min - thresh) / bin_w)
    hi = thresh + bin_w * np.ceil((data_max - thresh) / bin_w)
    # small +bin_w/2 to ensure inclusive upper edge due to FP
    return np.arange(lo, hi + bin_w/2, bin_w)

variables = ["HMR", "HSR"]
bin_widths = [0.75, 0.15]
threshs = [2.5, 0.8]


# Robust mapping R_total -> label with tolerance
def label_from_rtotal(rt, tol=1e-6):
    for val, lab in LEVELS:
        if np.isclose(rt, val, atol=tol, rtol=0):
            return lab
    return np.nan  # drop anything else



if manuscript_simulations_only:
    df_raw = pd.read_csv(
        "/Users/tejjolly/Documents/BioSimm/Simulations/"
        "Post_Processing/data/data_manuscript.csv"
    )
else:
    df_raw = pd.read_csv(
        "/Users/tejjolly/Documents/BioSimm/Simulations/"
        "Post_Processing/data/data.csv"
    )

# Labels we will actually use everywhere (hue_order, palette, etc.)
LEVELS = [
    (0.24, r"Base $R_{A\text{-}m}$"),
    (0.43, r"$R_{A\text{-}m}x2.7$"),
    (0.62, r"$R_{A\text{-}m}x4.0$"),
    (0.81, r"$R_{A\text{-}m}x6.3$")
]
hue_order = [lab for _, lab in LEVELS]

# Colormap picked once so the legend is consistent
vals = np.r_[0.25, np.linspace(0.6, 1.0, len(hue_order) - 1)]
color_map = mpl.cm.BuPu(vals)
palette = {lab: col for lab, col in zip(hue_order, color_map)}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

ymax_seen = 0
for ax, var, bin_width, thresh in zip(axes, variables, bin_widths, threshs):
    # Filter & tag
    df = (
        df_raw
        .pipe(lambda d: d[d["source"].eq("mine")] if my_simulations_only else d)
        .query("Condition == 'Hyperemic'")
        .dropna(subset=[var, "R_total"])
        .assign(Set=lambda d: d["R_total"].map(label_from_rtotal))
        .dropna(subset=["Set"])
    )

    # ---- Threshold summary (totals only) ----
    n_total = len(df)
    n_below = int((df[var] < thresh).sum())
    n_above = int((df[var] >= thresh).sum())
    p_below = 100.0 * n_below / n_total if n_total else 0.0
    p_above = 100.0 * n_above / n_total if n_total else 0.0

    print(f"\n{var} threshold summary (threshold = {thresh}):")
    print(f"  below: {n_below:4d} ({p_below:5.1f}%)   above: {n_above:4d} ({p_above:5.1f}%)   total: {n_total}")

    # Binning anchored at threshold
    bins = bins_anchored_at_edge(df[var].min(), df[var].max(), thresh=thresh, bin_w=bin_width)
    tick_bins = bins[::2]
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(tick_bins)
    ax.set_xticklabels([f"{b:.1f}" for b in tick_bins])

    sns.histplot(
        data=df,
        x=var,
        hue="Set",
        hue_order=hue_order,
        bins=bins,
        # stat="count",
        multiple="stack",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.75,
        ax=ax,
        legend=True  # we'll harvest the first legend and then remove both
    )

    ax.set_ylabel(None)

    if kde_overlay:
        for lab, group in df.groupby("Set"):
            kde = gaussian_kde(group[var])
            x_vals = np.linspace(bins[0], bins[-1], 500)
            y_vals = kde(x_vals) * len(group) * bin_width
            ax.plot(x_vals, y_vals, linewidth=3, color=palette[lab])

    # Reference line at the threshold
    ax.axvline(thresh, linestyle='--', linewidth=4, color='gray', alpha=1)

    # Per-axis cosmetics
    ax.grid(False)
    ax.set_xlabel(f"{var} [mmHg/cm/s]", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if y == 0 else f'{int(y)}'))

    # Track ymax to unify later (after seaborn draws)
    ymax_seen = max(ymax_seen, ax.get_ylim()[1])

# Hide right panel's y tick labels; rely on left only
axes[1].tick_params(labelleft=False)

# One shared y-label for the whole figure
fig.text(0.04, 0.5, "Count", va='center', rotation='vertical', fontsize=FONT_SIZE)

# Unify y-limits across both panels (rounded up nicely)
for ax in axes:
    ax.set_ylim(0, np.ceil(ymax_seen))

# Single, figure-level legend (take handles/labels from the left axis)
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    leg = ax.get_legend()
    if leg:
        leg.remove()
fig.legend(handles, labels, loc="upper center", ncol=len(hue_order), frameon=False)

plt.tight_layout(rect=(0.06, 0.0, 1.0, 0.88))  # leave space for shared y-label & top legend

fig.savefig("images/HSR_HMR_distribution_split.png", dpi=400, transparent=True)
fig.savefig("images/HSR_HMR_distribution_split.svg", transparent=True)
plt.show()
