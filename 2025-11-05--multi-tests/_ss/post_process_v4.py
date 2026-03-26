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
RUN_PHASES = [2, 3]          # e.g. [1,2,3] or [2,3] or [3]
RUN_GEOMETRIES = ["13", "37", "60"]      # order matters
PLOT_VELOCITY = False         # show velocity + secondary y in phase 2
THRESHOLD = 0.1              # slope: from max conc down to <= this
FIT_GEOMETRY = True          # fit HMR vs TAG per geometry in phase 3
NORMALIZE_CONC = False        # normalize phase-2 curves by their own max

# dyes / AIFs
DYES = ["N", "B"]            # B -> slow AIF, N -> fast AIF
DYE_LABELS = {
    "B": "B",
    "N": "N",
}

# label mapping for phase 3 legend
GEOM_LABELS = {
    "13": "Stenosis",
    "37": "No stenosis",
    "60": "Patient specific",
}

PHASE2_FONTSIZE = 16         # controls only Phase-2 figure text

# ============================================================

# ------------------------------------------------------------
# PATHS / CONFIG
# ------------------------------------------------------------
BASE = "/Volumes/biosimm-Tej-Jolly/2025-11-16--TAG"

PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"
EXTRACT_SCRIPT = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
    "2025-11-05--multi-tests/extract_concentration.py"
)

HMR_CSV = os.path.join(BASE, "hmr_data.csv")

# run suffixes per geometry (your latest)
RUN_SUFFIXES = {
    "13": ["24", "43", "62", "100"],
    "37": ["24", "43", "62", "100"],
    "60": ["24", "43", "62", "100"]
}

# arc-length file per geometry
ARC_PATHS = {
    "13": os.path.join(BASE, "g13_arclen.csv"),
    "37": os.path.join(BASE, "g37_arclen.csv"),
    "60": os.path.join(BASE, "g60_arclen.csv"),
    # "60": os.path.join(BASE, "path_patient_arclen.csv"),  # example
}

# concentration directory: e.g. BASE/concentrations/g13_r43_dB_concentration.csv
CONC_DIR = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2025-11-05--multi-tests/concentrations"

clevel = [0.2, 1]
# ------------------------------------------------------------


def suffix_to_rmicro(sfx: str) -> float:
    base = sfx.split("_")[0]   # '81_v2' -> '81'
    return float(base) / 100.0


def make_geom_cmap(geom_index: int):
    # kept for backward compatibility; not used for Phase 2 anymore
    if geom_index == 1:
        return plt.get_cmap("BuPu")
    else:
        return plt.get_cmap("OrRd")


def phase1_extract():
    print("[PHASE 1] running pvpython extractor …")
    subprocess.run([PV_PYTHON, EXTRACT_SCRIPT], check=True)
    print("[PHASE 1] done.")


def _conc_csv_path(geom: str, suffix: str, dye: str) -> str:
    """
    Build path like:
    BASE/concentrations/g13_r43_dB_concentration.csv
    """
    case_base = f"g{geom}_r{suffix}_d{dye}"
    fname = f"{case_base}_concentration.csv"
    return os.path.join(CONC_DIR, fname)


def compute_tag_threshold_for_geom(geom: str, dye: str, threshold: float = 0.1):
    """
    Compute TAG-like slope for each run suffix in a given geometry & dye.

    ASSUMPTION: Each concentration CSV is already a single snapshot at c_max
    (no Time column needed). We just treat it as concentration vs arc length.
    """
    arc_df = pd.read_csv(ARC_PATHS[geom])
    arc_all = arc_df["ArcLength"].to_numpy()
    n_arc = len(arc_all)

    rows = []

    for suffix in RUN_SUFFIXES[geom]:
        case_base = f"g{geom}_r{suffix}"
        case_dye = f"{case_base}_d{dye}"
        csv_path = _conc_csv_path(geom, suffix, dye)

        if not os.path.exists(csv_path):
            print(f"  [WARN] {csv_path} not found, skipping {case_dye}")
            continue

        # UPDATED: no Time logic, just one snapshot per file
        df = pd.read_csv(csv_path)
        if "Concentration" not in df.columns:
            print(f"  [WARN] no 'Concentration' column in {csv_path}, skipping")
            continue

        # Use first n_arc points along the centerline
        prof = df.iloc[:n_arc].reset_index(drop=True)

        conc_all = prof["Concentration"].to_numpy()
        if conc_all.size == 0 or np.all(conc_all == 0):
            print(f"  [WARN] all zeros or empty for {case_dye}, skipping")
            continue

        arc = arc_all[:len(conc_all)]

        # index of max conc along line
        imax = int(np.nanargmax(conc_all))

        # find cutoff downstream where concentration falls below threshold
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
                print(f"  [WARN] all zeros for {case_dye}, skipping")
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
            print(f"  [WARN] segment too short for {case_dye}, skipping")
            continue

        m, b = np.polyfit(seg_x, seg_y, 1)
        y_fit = m * seg_x + b

        ss_res = np.sum((seg_y - y_fit) ** 2)
        ss_tot = np.sum((seg_y - np.mean(seg_y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        print(f"  {case_dye}: slope={m}, R2={r2}, seg_len={len(seg_x)}")

        # t_max is no longer meaningful; keep placeholder for compatibility
        t_max = np.nan
        # assume first point is inlet concentration at this snapshot
        c_inlet = float(conc_all[0])

        rows.append({
            "case": case_base,      # matches HMR case key (no dye)
            "case_dye": case_dye,   # unique per geom/r/dye
            "geom": geom,
            "suffix": suffix,
            "dye": dye,
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

    for geom in RUN_GEOMETRIES:
        print(f"[PHASE 2] geometry g{geom}")

        # arc-length for this geometry
        arc_df = pd.read_csv(ARC_PATHS[geom])
        arc_all = arc_df["ArcLength"].to_numpy()
        n_arc = len(arc_all)

        suffixes = RUN_SUFFIXES[geom]
        r_values = [suffix_to_rmicro(sfx) for sfx in suffixes]
        ordered = sorted(zip(suffixes, r_values), key=lambda x: x[1])

        # Use BuPu for ALL concentration plots
        cmap = plt.get_cmap("BuPu")
        c_levels = np.linspace(clevel[0], clevel[1], len(ordered))
        suffix_to_color = {
            sfx: cmap(level)
            for (sfx, _), level in zip(ordered, c_levels)
        }

        for dye in DYES:
            print(f"[PHASE 2]  dye {dye}")
            tag_rows = compute_tag_threshold_for_geom(geom, dye, threshold=THRESHOLD)
            tag_map = {row["case_dye"]: row for row in tag_rows}

            # --- local font sizing just for Phase 2
            with plt.rc_context({
                "font.size": PHASE2_FONTSIZE,
                "axes.labelsize": PHASE2_FONTSIZE,
                "xtick.labelsize": PHASE2_FONTSIZE,
                "ytick.labelsize": PHASE2_FONTSIZE,
                "legend.fontsize": max(6, PHASE2_FONTSIZE - 2),
            }):
                fig, ax1 = plt.subplots(figsize=(8, 5))
                ax2 = ax1.twinx() if PLOT_VELOCITY else None

                arc = arc_all

                for suffix in suffixes:
                    case_base = f"g{geom}_r{suffix}"
                    case_dye = f"{case_base}_d{dye}"
                    csv_path = _conc_csv_path(geom, suffix, dye)

                    if not os.path.exists(csv_path):
                        print(f"  [WARN] {csv_path} not found, skipping {case_dye}")
                        continue

                    prof_all = pd.read_csv(csv_path)
                    if "Concentration" not in prof_all.columns:
                        print(f"  [WARN] no 'Concentration' column in {csv_path}, skipping")
                        continue

                    # UPDATED: treat entire file as single snapshot along centerline
                    prof = prof_all.iloc[:n_arc].reset_index(drop=True)

                    conc = prof["Concentration"].to_numpy()
                    if conc.size == 0:
                        print(f"  [WARN] empty concentration profile for {case_dye}, skipping")
                        continue

                    c_max = conc.max()
                    print(f"  {case_dye}: C_max_snapshot={c_max}")

                    x_arc = arc[:len(conc)]
                    color = suffix_to_color[suffix]

                    if c_max > 0 and NORMALIZE_CONC:
                        conc_plot = conc / c_max
                    else:
                        conc_plot = conc

                    ax1.plot(
                        x_arc, conc_plot,
                        label=case_base,
                        alpha=0.7,
                        color=color,
                        linewidth=3
                    )

                    if PLOT_VELOCITY and ax2 is not None:
                        if all(col in prof.columns for col in ["Velocity:0", "Velocity:1", "Velocity:2"]):
                            vel_mag = np.sqrt(
                                prof["Velocity:0"]**2 +
                                prof["Velocity:1"]**2 +
                                prof["Velocity:2"]**2
                            )
                            ax2.plot(x_arc, vel_mag, linestyle=":", color=color, linewidth=2)
                        else:
                            print(f"  [WARN] velocity columns missing for {case_dye}, skipping velocity plot")

                    # overlay fit
                    if case_dye in tag_map:
                        x_seg = tag_map[case_dye]["x_seg"]
                        y_fit = tag_map[case_dye]["y_fit"]
                        if c_max > 0 and NORMALIZE_CONC:
                            y_fit = y_fit / c_max
                        ax1.plot(x_seg, y_fit, linestyle="--", color=color, linewidth=1.5)

                ax1.set_xlabel("Centerline length [cm]")
                label_y = "Concentration" + (" (normalized)" if NORMALIZE_CONC else "")
                ax1.set_ylabel(label_y)
                if PLOT_VELOCITY and ax2 is not None:
                    ax2.set_ylabel("Velocity magnitude")
                ax1.set_ylim(bottom=0)
                if NORMALIZE_CONC:
                    ax1.set_ylim(top=1)
                ax1.set_xlim(left=0)

                # legend keyed by R_micro
                handles = [
                    Line2D([0], [0], color=suffix_to_color[sfx], lw=3,
                           label=f"R_micro = {r:.2f}")
                    for (sfx, r) in ordered
                ]
                ax1.legend(handles=handles, loc="best")

                # Title: Dye-B / Dye-N
                geom_name = GEOM_LABELS.get(geom, f"g{geom}")
                ax1.set_title(f"{geom_name}, Dye-{dye}")

                fig.tight_layout()
                out_name = f"tag_{geom}_d{dye}.svg"
                if NORMALIZE_CONC:
                    plt.savefig(
                        f"/Users/tejjolly/Documents/BioSimm/Meetings/2025-11-20--research_update/figures/normalized/{out_name}",
                        format="svg", transparent=True, dpi=600
                    )
                else:
                    plt.savefig(
                        f"/Users/tejjolly/Documents/BioSimm/Meetings/2025-11-20--research_update/figures/non-normalized/{out_name}",
                        format="svg", transparent=True, dpi=600
                    )
                plt.show()

    print("[PHASE 2] done.")


def phase3_calc_tafe_and_plot_hmr():
    print("[PHASE 3] computing TAG-like slopes and plotting vs HMR")

    hmr_df = pd.read_csv(HMR_CSV)
    # HMR case key does NOT include dye
    hmr_df["case"] = hmr_df.apply(lambda row: f"g{row['g']}_r{row['r']}", axis=1)

    all_rows = []
    for geom in RUN_GEOMETRIES:
        for dye in DYES:
            print(f"[PHASE 3] geometry g{geom}, dye {dye}")
            tag_rows = compute_tag_threshold_for_geom(geom, dye, threshold=THRESHOLD)
            all_rows.extend(tag_rows)

    tag_df = pd.DataFrame(all_rows)
    if tag_df.empty:
        print("[PHASE 3] no TAG rows found, aborting.")
        return

    merged = pd.merge(tag_df, hmr_df, on="case", how="left")
    merged_clean = merged.dropna(subset=["HMR", "TAG_slope"]).copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    # --- series are (geom × dye) ---
    series_keys = sorted(
        {(row["geom"], row["dye"]) for _, row in merged_clean.iterrows()}
    )
    series_cmap_names = ["Blues", "Greens", "Reds", "Greys", "Purples", "Oranges"]

    series_colors = {}     # scatter colors per (geom, dye)
    series_fit_colors = {} # fit-line colors per (geom, dye)

    for (geom, dye), cmap_name in zip(series_keys, series_cmap_names):
        cmap = plt.get_cmap(cmap_name)
        series_colors[(geom, dye)] = cmap(0.6)
        series_fit_colors[(geom, dye)] = cmap(0.9)

    # scatter, colored by (geom, dye)
    for _, row in merged_clean.iterrows():
        hmr = float(row["HMR"])
        tag = float(row["TAG_slope"])
        geom = row["geom"]
        dye = row["dye"]

        color = series_colors.get((geom, dye), "k")

        ax.scatter(
            hmr, tag,
            s=50,
            color=color,
            edgecolor="white",
            linewidths=0.6
        )

    # optional per-(geom, dye) fit
    if FIT_GEOMETRY:
        for geom, dye in series_keys:
            sub = merged_clean[
                (merged_clean["geom"] == geom) &
                (merged_clean["dye"] == dye)
            ]
            if len(sub) >= 2:
                X = sub["HMR"].to_numpy()
                Y = sub["TAG_slope"].to_numpy()
                m, b = np.polyfit(X, Y, 1)
                y_pred = m * X + b
                ss_res = np.sum((Y - y_pred) ** 2)
                ss_tot = np.sum((Y - np.mean(Y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                print(f"[PHASE 3] fit g{geom}, dye {dye}: slope={m}, intercept={b}, R2={r2}")

                x_line = np.linspace(X.min(), X.max(), 50)
                y_line = m * x_line + b
                line_color = series_fit_colors.get((geom, dye), "k")
                ax.plot(x_line, y_line, color=line_color, linewidth=1.5)

    # legend: one entry per (geom, dye) series
    legend_handles = []
    for (geom, dye), color in series_colors.items():
        geom_name = GEOM_LABELS.get(geom, f"g{geom}")
        dye_name = DYE_LABELS.get(dye, dye)
        label = f"{geom_name}, {dye_name}"
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=label,
            )
        )
    ax.legend(handles=legend_handles, title=None, loc="best")

    ax.set_xlabel("HMR [mmHg/cm/s]")
    ax.set_ylabel("TAG slope [Concentration / length]")

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        "/Users/tejjolly/Documents/BioSimm/Meetings/2025-11-20--research_update/figures/TAGvHMR.svg",
        format="svg", transparent=True, dpi=600
    )
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
