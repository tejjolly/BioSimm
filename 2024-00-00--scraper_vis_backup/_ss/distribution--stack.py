import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

# ─── Load & tag ────────────────────────────────────────────────────────────────
# THIS IS THE SET OF SIMULATIONS FOR THE MANUSCRIPT

manuscript_simulations_only = True
my_simulations_only = True
kde_overlay = False

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


variables = ["HMR"]#, 'HSR', 'HMR']
bin_widths = [0.6]#, 0.1, 0.5]
base_name = r"$\mathrm{Base}\ R_{A\text{-}m}$"
vary_name = r"$\mathrm{Elevated}\ R_{A\text{-}m}$"

for var, bin_width in zip(variables, bin_widths):
    df = (
        df_raw[df_raw["source"] == "mine"]
              .query("Condition == 'Hyperemic'")
              .dropna(subset=[var, "R_total"])
              .assign(Set=lambda d: np.where(d["R_total"] == 0.24,
                                             base_name,
                                             vary_name))
    )

    counts = df["Set"].value_counts()
    print(f"\nFor variable {var} (mine only):")
    print(f"{base_name}: {counts.get(base_name, 0)}")
    print(f"{vary_name}: {counts.get(vary_name, 0)}")

    # ─── Histogram (stacked: old bottom, new top) ──────────────────────────────────
    # bin_width = 0.1

    fig, ax = plt.subplots(figsize=(6, 3.85))
    palette = {base_name: "#5E9096",      # bottom colour
               vary_name:  "#5E2F5C"}      # top colour

    bins = np.arange(df[var].min(), df[var].max() + bin_width, bin_width)

    sns.histplot(
        data=df,
        x=var,
        hue="Set",
        hue_order=[vary_name, base_name],
        bins=bins,
        stat="count",
        multiple="stack",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
        ax=ax,
        legend=True
    )

    # Optional KDE overlays
    if kde_overlay:
        for set_name, group in df.groupby("Set"):
            kde = gaussian_kde(group[var])
            x_vals = np.linspace(bins[0], bins[-1], 500)
            y_vals = kde(x_vals) * len(group) * bin_width
            ax.plot(x_vals, y_vals,
                    # label=f"{set_name}",
                    linewidth=3,
                    color=palette[set_name])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    ax.set_ylabel("Count", fontsize = 20)
    ax.set_xlabel(f'{var} [mmHg/cm/s]', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    # ax.set_title(f"{var} distribution—old vs. new simulations")
    ax.grid(False)
    # ax.legend(fontsize=18)
    legend = ax.get_legend()
    legend.set_title(None)
    ax.axvline(2.5,linestyle='--',linewidth=4,color='gray',alpha=1)

    plt.tight_layout()
    fig.savefig(f"images/{var}_distribution.png", dpi=400, transparent=True)
    fig.savefig(f"images/{var}_distribution.svg", transparent=True)

    plt.show()


#─── Part 2:  use *all* rows, no filtering ────────────────────────────────────
if not my_simulations_only:
    for var, bin_width in zip(variables, bin_widths):
        df_all = (
            df_raw
            # keep every row from every source
            # only drop rows where *this* variable is NaN (needed for KDE / hist)
            .query("Condition == 'Hyperemic'")
            .dropna(subset=[var])
            .assign(Set=lambda d: np.where(d["source"] == "mine",
                                           base_name,
                                           vary_name))
        )

        counts = df_all["Set"].value_counts()
        print(f"\nFor variable {var} (all sources):")
        print(f"{base_name}: {counts.get(base_name, 0)}")
        print(f"{vary_name}: {counts.get(vary_name, 0)}")

        # ── Plot histogram (stacked) ───────────────────────────────────────────────
        # fig, ax = plt.subplots(figsize=(8.25, 4))
        fig, ax = plt.subplots(figsize=(6, 4))
        palette = {base_name: "#5E9096", vary_name: "#5E2F5C"}
        # palette = {base_name: "#556092", vary_name: "#FFFFFF"}

        bins = np.arange(df_all[var].min(),
                         df_all[var].max() + bin_width,
                         bin_width)

        sns.histplot(
            data=df_all,
            x=var,
            hue="Set",
            hue_order=[vary_name, base_name],   # top = “Elevated”, bottom = “Base”
            bins=bins,
            stat="count",
            multiple="stack",
            palette=palette,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            ax=ax,
            legend=True
        )

        # Optional KDE overlays
        for set_name, group in df_all.groupby("Set"):
            kde = gaussian_kde(group[var])
            x_vals = np.linspace(bins[0], bins[-1], 500)
            y_vals = kde(x_vals) * len(group) * bin_width
            ax.plot(x_vals, y_vals, linewidth=3, color=palette[set_name])

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_ylabel("Count", fontsize=20)
        ax.set_xlim([0.5, 8])
        ax.set_xlabel(f'{var} [mmHg/cm/s]', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(False)

        plt.tight_layout()
        fig.savefig(f"images/{var}_distribution_allSources.png",
                    dpi=600, transparent=True)
        fig.savefig(f"images/{var}_distribution_allSources.svg",transparent=True)

        plt.show()
