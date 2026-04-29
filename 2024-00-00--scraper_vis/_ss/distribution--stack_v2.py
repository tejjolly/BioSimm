import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


# ─── Load ──────────────────────────────────────────────────────────────────────
manuscript_simulations_only = True
my_simulations_only = True
kde_overlay = False

def bins_anchored_at_edge(data_min, data_max, thresh, bin_w):
    lo = thresh + bin_w * np.floor((data_min - thresh) / bin_w)
    hi = thresh + bin_w * np.ceil((data_max - thresh) / bin_w)
    # small +bin_w/2 to ensure inclusive upper edge due to FP
    return np.arange(lo, hi + bin_w/2, bin_w)


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

# Robust mapping R_total -> label with tolerance
def label_from_rtotal(rt, tol=1e-6):
    for val, lab in LEVELS:
        if np.isclose(rt, val, atol=tol, rtol=0):
            return lab
    return np.nan  # drop anything else

variables = ["HSR", "HMR"]
bin_widths = [0.15, 0.75]
threshs = [0.8, 2.5]

for var, bin_width, thresh in zip(variables, bin_widths, threshs):
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
    n_total   = len(df)
    n_below   = int((df[var] < thresh).sum())
    n_above   = int((df[var] >= thresh).sum())
    p_below   = 100.0 * n_below / n_total if n_total else 0.0
    p_above   = 100.0 * n_above / n_total if n_total else 0.0

    print(f"\n{var} threshold summary (threshold = {thresh}):")
    print(f"  below: {n_below:4d} ({p_below:5.1f}%)   above: {n_above:4d} ({p_above:5.1f}%)   total: {n_total}")

    # Colormap — choose ONE of the two lines below:
    # Full spectrum (may be very light at the bottom / very dark at top):
    # color_map = mpl.cm.Reds(np.linspace(0, 1, len(hue_order)))
    # Visibility-safe mid–deep range (recommended for white/transparent backgrounds):
    color_map = mpl.cm.BuPu(np.linspace(0.25, 1, len(hue_order)))
    fig, ax = plt.subplots(figsize=(6, 3.85))
    palette = {lab: col for lab, col in zip(hue_order, color_map)}
    bins = bins_anchored_at_edge(df[var].min(), df[var].max(), thresh=thresh, bin_w=bin_width)
    tick_bins = bins[::2]  # downsample if you want fewer ticks
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xticks(tick_bins)
    ax.set_xticklabels([f"{b:.1f}" for b in tick_bins])  # <-- labels from tick_bins

    sns.histplot(
        data=df,
        x=var,
        hue="Set",
        hue_order=hue_order,           # least → most diseased; last stacks on top
        bins=bins,
        stat="count",
        multiple="stack",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        alpha=1,
        ax=ax,
        legend=True
    )

    # sns.rugplot(
    #     data=df,
    #     x=var,
    #     # hue="Set",
    #     # hue_order=hue_order,
    #     color='black',
    #     ax=ax,
    #     palette=palette,
    #     height=0.15
    # )

    # Optional KDE overlays (match colors)

    if kde_overlay:
        for lab, group in df.groupby("Set"):
            kde = gaussian_kde(group[var])
            x_vals = np.linspace(bins[0], bins[-1], 500)
            y_vals = kde(x_vals) * len(group) * bin_width
            ax.plot(x_vals, y_vals, linewidth=3, color=palette[lab])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_ylabel("Count", fontsize=20)
    ax.set_ylim([0,8])
    ax.set_xlabel(f"{var} [mmHg/cm/s]", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(False)
    legend = ax.get_legend()
    if legend:
        legend.set_title(None)

    # Reference line (adjust as needed)
    ax.axvline(thresh, linestyle='--', linewidth=4, color='black', alpha=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if y == 0 else f'{int(y)}'))

    plt.tight_layout()
    fig.savefig(f"../images/{var}_distribution_split.png", dpi=400, transparent=True)
    fig.savefig(f"../images/{var}_distribution_split.svg", transparent=True)
    plt.show()