#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TOGGLES — change here in the IDE
# ============================================================
RUN_PHASES = [2, 3]            # e.g. [1,2,3] or [2,3] or [3]
RUN_GEOMETRIES = ["13", "37"]  # can be ["13"] or ["13", "37"]
PLOT_VELOCITY = False           # turn off to hide velocity + secondary y in phase 2
# ============================================================

# ------------------------------------------------------------
# PATHS / CONFIG
# ------------------------------------------------------------
BASE = "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests"

# extractor (phase 1)
PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"
EXTRACT_SCRIPT = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
    "2025-11-05--multi-tests/extract_concentration.py"
)

# hmr data csv
HMR_CSV = os.path.join(BASE, "hmr_data.csv")

# run suffixes per geometry
RUN_SUFFIXES = {
    "13": ["24", "62", "81_v2", "100_v3"],
    "37": ["24", "62", "81", "100"],
}

# arc-length file per geometry
ARC_PATHS = {
    "13": os.path.join(BASE, "path_LCA_1_arclen.csv"),
    "37": os.path.join(BASE, "path_LCA_1_arclen.csv"),  # change if different
}

# linear-fit segment per geometry
SEGMENTS = {
    "13": (105, 140),
    "37": (25, 90),
}
# ------------------------------------------------------------


def phase1_extract():
    """Call pvpython extractor that writes concentration.csv per run."""
    print("[PHASE 1] running pvpython extractor …")
    subprocess.run([PV_PYTHON, EXTRACT_SCRIPT], check=True)
    print("[PHASE 1] done.")


def compute_tag_for_geom(geom):
    """
    Helper: compute TAG slopes for one geometry.
    Returns a list of dicts, one per case in that geometry:
    {
      case, geom, suffix, TAG_slope, R2, t_max, C_inlet
    }
    """
    arc_path = ARC_PATHS[geom]
    arc_df = pd.read_csv(arc_path)
    arc = arc_df["ArcLength"].to_numpy()
    n_arc = len(arc)

    i0, i1 = SEGMENTS[geom]
    rows = []

    for suffix in RUN_SUFFIXES[geom]:
        case = f"g{geom}_r{suffix}"
        csv_path = os.path.join(BASE, case, "concentration.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARN] {csv_path} not found, skipping {case}")
            continue

        df = pd.read_csv(csv_path)
        inlet_df = df.groupby("Time").head(1)
        idx_max = inlet_df["Concentration"].idxmax()
        t_max = inlet_df.loc[idx_max, "Time"]
        c_inlet = inlet_df.loc[idx_max, "Concentration"]

        prof = df[df["Time"] == t_max].reset_index(drop=True)
        prof = prof.iloc[:n_arc].reset_index(drop=True)

        x = arc[i0:i1+1]
        y = prof.loc[i0:i1, "Concentration"].to_numpy()

        # fit
        m, b = np.polyfit(x, y, 1)

        # R^2
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        rows.append({
            "case": case,
            "geom": geom,
            "suffix": suffix,
            "TAG_slope": m,
            "R2": r2,
            "t_max": t_max,
            "C_inlet": c_inlet,
            "x_seg": x,
            "y_seg": y,
            "y_fit": y_pred,
        })
        print(f"  {case}: TAG_slope={m}, R2={r2}")

    return rows


def phase2_plot_concentration():
    """
    For each geometry:
      - read the concentration.csv for each run
      - pick the time of max inlet concentration
      - plot concentration vs arc length
      - (optionally) plot velocity magnitude on secondary y
      - show point index on top
      - overlay fitted slope segment (from same data)
    """
    print("[PHASE 2] plotting concentration (and maybe velocity)")

    for geom in RUN_GEOMETRIES:
        print(f"[PHASE 2] geometry g{geom}")
        arc_path = ARC_PATHS[geom]
        arc_df = pd.read_csv(arc_path)
        arc_all = arc_df["ArcLength"].to_numpy()
        n_arc = len(arc_all)
        i0, i1 = SEGMENTS[geom]

        # compute TAG for this geom so we can overlay the fit
        tag_rows = compute_tag_for_geom(geom)
        tag_map = {row["case"]: row for row in tag_rows}

        profiles = {}
        for suffix in RUN_SUFFIXES[geom]:
            case = f"g{geom}_r{suffix}"
            csv_path = os.path.join(BASE, case, "concentration.csv")
            if not os.path.exists(csv_path):
                print(f"  [WARN] {csv_path} not found, skipping {case}")
                continue

            df = pd.read_csv(csv_path)
            inlet_df = df.groupby("Time").head(1)
            idx_max = inlet_df["Concentration"].idxmax()
            t_max = inlet_df.loc[idx_max, "Time"]
            c_max = inlet_df.loc[idx_max, "Concentration"]
            print(f"  {case}: t_max={t_max}, C_inlet={c_max}")

            prof = df[df["Time"] == t_max].reset_index(drop=True)
            prof = prof.iloc[:n_arc].reset_index(drop=True)
            profiles[case] = prof

        if not profiles:
            continue

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax_top = ax1.twiny()

        # only make secondary y if velocity is on
        if PLOT_VELOCITY:
            ax2 = ax1.twinx()
        else:
            ax2 = None

        arc = arc_all
        npts = len(arc)

        for case, prof in profiles.items():
            x_arc = arc
            conc = prof["Concentration"]

            line = ax1.plot(x_arc, conc, label=case, alpha=0.7)[0]
            color = line.get_color()

            if PLOT_VELOCITY:
                vel_mag = np.sqrt(
                    prof["Velocity:0"]**2 +
                    prof["Velocity:1"]**2 +
                    prof["Velocity:2"]**2
                )
                ax2.plot(x_arc, vel_mag, linestyle=":", color=color)

            # overlay the fitted line segment for this case
            if case in tag_map:
                x_seg = tag_map[case]["x_seg"]
                y_fit = tag_map[case]["y_fit"]
                ax1.plot(x_seg, y_fit, linestyle="--", color=color, linewidth=3)

        ax1.set_xlabel("Arc length along centerline")
        ax1.set_ylabel("Concentration (at time of max inlet)")
        if PLOT_VELOCITY:
            ax2.set_ylabel("Velocity magnitude")
        ax1.set_title(f"Centerline concentration snapshot (g{geom})")

        # top x-axis = point index
        ax_top.set_xlim(ax1.get_xlim())
        tick_idx = np.arange(0, npts, 20)
        tick_pos = arc[tick_idx]
        ax_top.set_xticks(tick_pos)
        ax_top.set_xticklabels([str(i) for i in tick_idx])
        ax_top.set_xlabel("Point index")
        ax1.set_ylim(bottom=0)
        ax1.legend(loc="upper right")
        fig.tight_layout()
        plt.show()

    print("[PHASE 2] done.")


def phase3_calc_tafe_and_plot_hmr():
    """
    - compute TAG slope for every (geometry, run_suffix)
    - load hmr_data.csv
    - merge on (g, r)
    - plot HMR vs TAG, label each point
    - print R2 for each fit (already printed in compute_tag_for_geom)
    """
    print("[PHASE 3] computing TAG-like slopes and plotting vs HMR")

    hmr_df = pd.read_csv(HMR_CSV)
    hmr_df["case"] = hmr_df.apply(
        lambda row: f"g{row['g']}_r{row['r']}", axis=1
    )

    # gather TAG from all geometries
    all_rows = []
    for geom in RUN_GEOMETRIES:
        print(f"[PHASE 3] geometry g{geom}")
        tag_rows = compute_tag_for_geom(geom)
        all_rows.extend(tag_rows)

    tag_df = pd.DataFrame(all_rows)
    merged = pd.merge(tag_df, hmr_df, on="case", how="left")

    # joint plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # keep only rows that have both values
    merged_clean = merged.dropna(subset=["HMR", "TAG_slope"]).copy()

    # one color per geometry
    geom_list = sorted(merged_clean["g"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    geom_color = {g: cmap(i % 10) for i, g in enumerate(geom_list)}

    # scatter + labels (color by geometry)
    for _, row in merged_clean.iterrows():
        hmr = float(row["HMR"])
        tag = float(row["TAG_slope"])
        label = str(row["case"])
        gval = row["g"]

        c = geom_color[gval]
        ax.scatter(hmr, tag, s=50, color=c, edgecolor="white", linewidths=0.6)
        ax.text(hmr + 0.03, tag - 0.005, label, fontsize=8, va="top", color=c)

    # legend: one entry per geometry
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=geom_color[g], markeredgecolor="k",
               markeredgewidth=0.5, label=f"g{g}")
        for g in geom_list
    ]
    ax.legend(handles=legend_handles, title="Geometry", loc="best")

    # style to emphasize "4th quadrant-ish" look
    ax.set_xlabel("HMR")
    ax.set_ylabel("TAG slope (conc / length)")
    ax.set_title("HMR vs TAG (all geometries)")

    # x-axis on top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # remove bottom and right spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(False)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    if 1 in RUN_PHASES:
        phase1_extract()
    if 2 in RUN_PHASES:
        phase2_plot_concentration()
    if 3 in RUN_PHASES:
        phase3_calc_tafe_and_plot_hmr()