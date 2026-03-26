#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# TOGGLES — change here in the IDE
# ============================================================
RUN_PHASES = [2]            # e.g. [1,2,3] or [2,3] or [3]
RUN_GEOMETRIES = ["13", "37"]  # order matters: 1st -> BuPu, 2nd -> OrRd
PLOT_VELOCITY = False          # show velocity + secondary y in phase 2
THRESHOLD = 0.1                # slope: from max conc down to <= this
FIT_GEOMETRY = True            # fit HMR vs TAG per geometry in phase 3
NORMALIZE_CONC = True          # normalize phase-2 curves by their own max
# label mapping for phase 3 legend
GEOM_LABELS = {
    "13": "stenosis",
    "37": "no stenosis",
}
# ============================================================

# ------------------------------------------------------------
# PATHS / CONFIG
# ------------------------------------------------------------
BASE = "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests"

PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"
EXTRACT_SCRIPT = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
    "2025-11-05--multi-tests/extract_concentration.py"
)

HMR_CSV = os.path.join(BASE, "hmr_data.csv")

# run suffixes per geometry (your latest)
RUN_SUFFIXES = {
    "13": ["24", "62", "81_v2", "100_v3"],
    "37": ["24", "62", "81", "100"],
}

# arc-length file per geometry
ARC_PATHS = {
    "13": os.path.join(BASE, "path_LCA_1_arclen.csv"),
    "37": os.path.join(BASE, "path_LCA_1_arclen.csv"),
}

clevel = [0.2, 1]
# ------------------------------------------------------------


def suffix_to_rmicro(sfx: str) -> float:
    base = sfx.split("_")[0]   # '81_v2' -> '81'
    return float(base) / 100.0


def make_geom_cmap(geom_index: int):
    if geom_index == 0:
        return plt.get_cmap("BuPu")
    else:
        return plt.get_cmap("OrRd")


def phase1_extract():
    print("[PHASE 1] running pvpython extractor …")
    subprocess.run([PV_PYTHON, EXTRACT_SCRIPT], check=True)
    print("[PHASE 1] done.")


def compute_tag_threshold_for_geom(geom: str, threshold: float = 0.1):
    arc_df = pd.read_csv(ARC_PATHS[geom])
    arc_all = arc_df["ArcLength"].to_numpy()
    n_arc = len(arc_all)

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

        conc_all = prof["Concentration"].to_numpy()
        arc = arc_all[:len(conc_all)]

        # index of max conc along line
        imax = int(np.nanargmax(conc_all))

        # find cutoff
        icut = None
        for i in range(imax + 1, len(conc_all)):
            c = conc_all[i]
            if c == 0:
                continue
            if c <= threshold:
                icut = i
                break

        if icut is None:
            nz = np.where(conc_all != 0)[0]
            if len(nz) == 0:
                print(f"  [WARN] all zeros for {case}, skipping")
                continue
            icut = int(nz[-1])
            if icut <= imax:
                icut = len(conc_all) - 1

        seg_x = arc[imax:icut + 1]
        seg_y = conc_all[imax:icut + 1]
        mask = seg_y != 0
        seg_x = seg_x[mask]
        seg_y = seg_y[mask]

        if len(seg_x) < 2:
            print(f"  [WARN] segment too short for {case}, skipping")
            continue

        m, b = np.polyfit(seg_x, seg_y, 1)
        y_fit = m * seg_x + b

        ss_res = np.sum((seg_y - y_fit) ** 2)
        ss_tot = np.sum((seg_y - np.mean(seg_y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        print(f"  {case}: slope={m}, R2={r2}, seg_len={len(seg_x)}")

        rows.append({
            "case": case,
            "geom": geom,
            "suffix": suffix,
            "TAG_slope": m,
            "R2": r2,
            "t_max": t_max,
            "C_inlet": c_inlet,
            "x_seg": seg_x,
            "y_fit": y_fit,
        })

    return rows


def phase2_plot_concentration():
    print("[PHASE 2] plotting concentration (and maybe velocity)")

    for g_idx, geom in enumerate(RUN_GEOMETRIES):
        print(f"[PHASE 2] geometry g{geom}")

        tag_rows = compute_tag_threshold_for_geom(geom, threshold=THRESHOLD)
        tag_map = {row["case"]: row for row in tag_rows}

        arc_df = pd.read_csv(ARC_PATHS[geom])
        arc_all = arc_df["ArcLength"].to_numpy()
        n_arc = len(arc_all)

        suffixes = RUN_SUFFIXES[geom]
        r_values = [suffix_to_rmicro(sfx) for sfx in suffixes]
        ordered = sorted(zip(suffixes, r_values), key=lambda x: x[1])
        cmap = make_geom_cmap(g_idx)
        c_levels = np.linspace(clevel[0], clevel[1], len(ordered))
        suffix_to_color = {
            sfx: cmap(level)
            for (sfx, _), level in zip(ordered, c_levels)
        }

        # load profiles
        profiles = {}
        for suffix in suffixes:
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
        ax2 = ax1.twinx() if PLOT_VELOCITY else None

        arc = arc_all
        npts = len(arc)

        for suffix in suffixes:
            case = f"g{geom}_r{suffix}"
            if case not in profiles:
                continue
            prof = profiles[case]
            x_arc = arc
            conc = prof["Concentration"].to_numpy()
            color = suffix_to_color[suffix]

            if NORMALIZE_CONC:
                cmax = conc.max()
                conc_plot = conc / cmax if cmax > 0 else conc
            else:
                conc_plot = conc

            ax1.plot(x_arc, conc_plot, label=case, alpha=0.7, color=color)

            if PLOT_VELOCITY and ax2 is not None:
                vel_mag = np.sqrt(
                    prof["Velocity:0"]**2 +
                    prof["Velocity:1"]**2 +
                    prof["Velocity:2"]**2
                )
                ax2.plot(x_arc, vel_mag, linestyle=":", color=color, linewidth=2)

            # overlay fit
            if case in tag_map:
                x_seg = tag_map[case]["x_seg"]
                y_fit = tag_map[case]["y_fit"]
                if NORMALIZE_CONC and conc.max() > 0:
                    y_fit = y_fit / conc.max()
                ax1.plot(
                    x_seg,
                    y_fit,
                    linestyle="--",
                    color=color,
                    linewidth=3,
                )

        ax1.set_xlabel("Centerline length [cm]")
        label_y = "Concentration"
        if NORMALIZE_CONC:
            label_y += " (normalized)"
        ax1.set_ylabel(label_y)
        if PLOT_VELOCITY and ax2 is not None:
            ax2.set_ylabel("Velocity magnitude")
        # ax1.set_title(f"Centerline concentration snapshot (g{geom})")
        ax1.set_ylim(bottom=0)
        if NORMALIZE_CONC:
            ax1.set_ylim(top=1)
        ax1.set_xlim(left=0)

        # top x-axis = point index
        ax_top.set_xlim(ax1.get_xlim())
        tick_idx = np.arange(0, npts, 20)
        tick_pos = arc[tick_idx]
        ax_top.set_xticks(tick_pos)
        ax_top.set_xticklabels([str(i) for i in tick_idx])
        ax_top.set_xlabel("Point index")

        # legend: R_micro
        handles = []
        for (sfx, r) in ordered:
            handles.append(
                Line2D([0], [0], color=suffix_to_color[sfx], lw=3,
                       label=f"R_micro = {r:.2f}")
            )
        ax1.legend(handles=handles, loc="upper right")

        fig.tight_layout()
        plt.show()

    print("[PHASE 2] done.")


def phase3_calc_tafe_and_plot_hmr():
    print("[PHASE 3] computing TAG-like slopes and plotting vs HMR")

    hmr_df = pd.read_csv(HMR_CSV)
    hmr_df["case"] = hmr_df.apply(
        lambda row: f"g{row['g']}_r{row['r']}", axis=1
    )

    all_rows = []
    for geom in RUN_GEOMETRIES:
        print(f"[PHASE 3] geometry g{geom}")
        tag_rows = compute_tag_threshold_for_geom(geom, threshold=THRESHOLD)
        all_rows.extend(tag_rows)

    tag_df = pd.DataFrame(all_rows)
    merged = pd.merge(tag_df, hmr_df, on="case", how="left")
    merged_clean = merged.dropna(subset=["HMR", "TAG_slope"]).copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    # ---- NEW: single color per geometry ----
    geom_base_colors = {}
    for g_idx, geom in enumerate(RUN_GEOMETRIES):
        # pick one shade from that geometry's cmap
        geom_base_colors[geom] = make_geom_cmap(g_idx)(0.6)

    # scatter
    for _, row in merged_clean.iterrows():
        hmr = float(row["HMR"])
        tag = float(row["TAG_slope"])
        geom = row["geom"]

        base_color = geom_base_colors[geom]

        label = GEOM_LABELS.get(geom, f"g{geom}")
        if label.lower() == "stenosis":
            marker = "^"   # triangle
        else:
            marker = "o"   # circle

        ax.scatter(
            hmr,
            tag,
            s=50,
            color=base_color,
            edgecolor="white",
            linewidths=0.6,
            marker=marker,
        )

    # optional per-geometry fit (use same base color)
    if FIT_GEOMETRY:
        for g_idx, geom in enumerate(RUN_GEOMETRIES):
            sub = merged_clean[merged_clean["geom"] == geom]
            if len(sub) >= 2:
                X = sub["HMR"].to_numpy()
                Y = sub["TAG_slope"].to_numpy()
                m, b = np.polyfit(X, Y, 1)
                y_line = m * np.linspace(X.min(), X.max(), 50) + b
                x_line = np.linspace(X.min(), X.max(), 50)
                ax.plot(
                    x_line,
                    y_line,
                    color=geom_base_colors[geom],
                    linewidth=1.5,
                )

    # legend: stenosis vs no stenosis, same colors as above
    legend_handles = []
    for g_idx, geom in enumerate(RUN_PHASES and RUN_GEOMETRIES):
        display_name = GEOM_LABELS.get(geom, f"g{geom}")
        if display_name.lower() == "stenosis":
            marker = "^"
        else:
            marker = "o"
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                markerfacecolor=geom_base_colors[geom],
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=display_name,
            )
        )
    ax.legend(handles=legend_handles, loc="best")

    # cosmetics
    ax.set_xlabel("HMR [mmHg/cm/s]")
    ax.set_ylabel("TAG slope [Concentration / length]")
    #ax.set_title("HMR vs TAG")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("/Users/tejjolly/Documents/BioSimm/Meetings/2025-11-06--research_update/figures/TAGvHMR.svg",format="svg",transparent=True, dpi=600)
    plt.show()

    print("[PHASE 3] done.")


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