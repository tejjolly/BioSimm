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
THRESHOLD = 0.0              # slope: from max conc down to <= this
FIT_GEOMETRY = True          # fit HMR vs TAG per geometry in phase 3
NORMALIZE_CONC = False        # normalize phase-2 curves by their own max
NORMALIZED_TAG = False

PHASE2_FONTSIZE = 16         # controls only Phase-2 figure text

# Global-ish figure size controls
FIGSIZE_PHASE2 = (3, 2)      # for phase-2 concentration plots (original 8x5)
FIGSIZE_PHASE3 = (3, 2)      # for TAG vs HMR plots (combined + individual) (original 6x4)

PHASE2_CLEAN = True   # removes all axis text/labels/ticks/legend/title
PHASE2_LEGEND = True
PLOT_TITLES = False

# dyes / AIFs
DYES = ["N", "B"]            # B -> slow AIF, N -> fast AIF
DYE_LABELS = {
    "B": "flat dye input @ time t",
    "N": "sharp dye input @ time t",
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

    Returns a list of dicts, one per (geom, suffix, dye), with:
      - TAG_slope        : raw slope (Concentration / length)
      - TAG_slope_norm   : slope using concentration normalized by group_max
      - R2               : R^2 for raw fit
      - R2_norm          : R^2 for normalized fit
      - x_seg            : x locations for fitted segment
      - y_fit            : fitted raw values
      - y_fit_norm       : fitted normalized values
      - group_max        : max concentration over all suffixes in this (geom, dye) group
    """
    arc_df = pd.read_csv(ARC_PATHS[geom])
    arc_all = arc_df["ArcLength"].to_numpy()
    n_arc = len(arc_all)

    temp_entries = []
    group_max = 0.0

    # -------- FIRST PASS: collect segments and find group_max --------
    for suffix in RUN_SUFFIXES[geom]:
        case_base = f"g{geom}_r{suffix}"
        case_dye = f"{case_base}_d{dye}"
        csv_path = _conc_csv_path(geom, suffix, dye)

        if not os.path.exists(csv_path):
            print(f"  [WARN] {csv_path} not found, skipping {case_dye}")
            continue

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

        # update group-wide max for this (geom, dye)
        c_max = float(np.nanmax(conc_all))
        if c_max > group_max:
            group_max = c_max

        arc = arc_all[:len(conc_all)]

        # index of max conc along line
        imax = int(np.nanargmax(conc_all))

        # find cutoff downstream where concentration falls below threshold (raw units)
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
        seg_y_raw = conc_all[imax:icut + 1]
        mask = seg_y_raw != 0
        seg_x = seg_x[mask]
        seg_y_raw = seg_y_raw[mask]

        if len(seg_x) < 2:
            print(f"  [WARN] segment too short for {case_dye}, skipping")
            continue

        # store for second-pass fitting
        C_inlet = float(conc_all[0])  # first point as inlet conc at snapshot
        temp_entries.append({
            "case": case_base,      # matches HMR case key (no dye)
            "case_dye": case_dye,   # unique per geom/r/dye
            "geom": geom,
            "suffix": suffix,
            "dye": dye,
            "t_max": np.nan,        # no longer meaningful; kept for compatibility
            "C_inlet": C_inlet,
            "x_seg": seg_x,
            "seg_y_raw": seg_y_raw,
        })

    if not temp_entries:
        return []

    if group_max <= 0:
        print(f"  [WARN] group_max <= 0 for geom {geom}, dye {dye}; no normalized slopes will be meaningful.")
        # we'll still compute raw slopes below; normalized will be NaN

    # -------- SECOND PASS: compute raw & normalized slopes --------
    rows = []
    for entry in temp_entries:
        seg_x = entry["x_seg"]
        seg_y_raw = entry["seg_y_raw"]

        # ---- raw fit ----
        m_raw, b_raw = np.polyfit(seg_x, seg_y_raw, 1)
        y_fit_raw = m_raw * seg_x + b_raw

        ss_res_raw = np.sum((seg_y_raw - y_fit_raw) ** 2)
        ss_tot_raw = np.sum((seg_y_raw - np.mean(seg_y_raw)) ** 2)
        r2_raw = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw != 0 else np.nan

        # ---- normalized fit (per geom+dye group) ----
        if group_max > 0:
            seg_y_norm = seg_y_raw / group_max
            m_norm, b_norm = np.polyfit(seg_x, seg_y_norm, 1)
            y_fit_norm = m_norm * seg_x + b_norm

            ss_res_norm = np.sum((seg_y_norm - y_fit_norm) ** 2)
            ss_tot_norm = np.sum((seg_y_norm - np.mean(seg_y_norm)) ** 2)
            r2_norm = 1 - ss_res_norm / ss_tot_norm if ss_tot_norm != 0 else np.nan
        else:
            m_norm = np.nan
            b_norm = np.nan
            y_fit_norm = None
            r2_norm = np.nan

        print(
            f"  {entry['case_dye']}: "
            f"raw slope={m_raw:.4g}, norm slope={m_norm:.4g}, "
            f"R2_raw={r2_raw:.3f}, R2_norm={r2_norm:.3f}, "
            f"seg_len={len(seg_x)}"
        )

        rows.append({
            "case": entry["case"],
            "case_dye": entry["case_dye"],
            "geom": entry["geom"],
            "suffix": entry["suffix"],
            "dye": entry["dye"],
            "TAG_slope": m_raw,
            "TAG_slope_norm": m_norm,
            "R2": r2_raw,
            "R2_norm": r2_norm,
            "t_max": entry["t_max"],
            "C_inlet": entry["C_inlet"],
            "x_seg": seg_x,
            "y_fit": y_fit_raw,
            "y_fit_norm": y_fit_norm,
            "group_max": group_max,
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

            # TAG rows (raw concentrations, used only for fit overlay)
            tag_rows = compute_tag_threshold_for_geom(geom, dye, threshold=THRESHOLD)
            tag_map = {row["case_dye"]: row for row in tag_rows}

            # --------------------------------------------------------
            # FIRST PASS: load all profiles and find group_max per geom+dye
            # --------------------------------------------------------
            conc_data = {}   # suffix -> dict with conc, x_arc, prof, c_max
            group_max = 0.0

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

                # treat entire file as single snapshot along centerline
                prof = prof_all.iloc[:n_arc].reset_index(drop=True)
                conc = prof["Concentration"].to_numpy()
                if conc.size == 0:
                    print(f"  [WARN] empty concentration profile for {case_dye}, skipping")
                    continue

                c_max = float(conc.max())
                print(f"  {case_dye}: C_max_snapshot={c_max}")

                x_arc = arc_all[:len(conc)]
                conc_data[suffix] = {
                    "case_base": case_base,
                    "case_dye": case_dye,
                    "x_arc": x_arc,
                    "conc": conc,
                    "prof": prof,
                }

                if c_max > group_max:
                    group_max = c_max

            if not conc_data:
                print(f"  [WARN] no valid profiles for geometry g{geom}, dye {dye}")
                continue

            # --------------------------------------------------------
            # SECOND PASS: plot using per-(geom,dye) group_max
            # --------------------------------------------------------
            with plt.rc_context({
                "font.size": PHASE2_FONTSIZE,
                "axes.labelsize": PHASE2_FONTSIZE,
                "xtick.labelsize": PHASE2_FONTSIZE,
                "ytick.labelsize": PHASE2_FONTSIZE,
                "legend.fontsize": max(6, PHASE2_FONTSIZE - 2),
            }):
                fig, ax1 = plt.subplots(figsize=FIGSIZE_PHASE2)
                ax2 = ax1.twinx() if PLOT_VELOCITY else None

                for suffix in suffixes:
                    if suffix not in conc_data:
                        continue

                    data = conc_data[suffix]
                    case_base = data["case_base"]
                    case_dye = data["case_dye"]
                    x_arc = data["x_arc"]
                    conc = data["conc"]
                    prof = data["prof"]
                    color = suffix_to_color[suffix]

                    if NORMALIZE_CONC and group_max > 0:
                        conc_plot = conc / group_max
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

                    # overlay TAG fit (computed in raw units, scaled here if normalized)
                    if case_dye in tag_map:
                        x_seg = tag_map[case_dye]["x_seg"]
                        y_fit = tag_map[case_dye]["y_fit"]
                        if NORMALIZE_CONC and group_max > 0:
                            y_fit_plot = y_fit / group_max
                        else:
                            y_fit_plot = y_fit
                        ax1.plot(x_seg, y_fit_plot, linestyle="--", color=color, linewidth=1.5)
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
                if PHASE2_LEGEND:
                    ax1.legend(handles=handles, loc="best")

                # Title: geom label + dye
                geom_name = GEOM_LABELS.get(geom, f"g{geom}")
                dye_name = DYE_LABELS.get(dye, dye)
                if PLOT_TITLES:
                    ax1.set_title(f"{geom_name}, {dye_name}")
                # ------------------------------------------------------------
                # PHASE 2 CLEAN MODE (for inset usage)
                # ------------------------------------------------------------
                if PHASE2_CLEAN:
                    # Turn off ticks AND tick labels
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

                    if ax2 is not None:
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                        ax2.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

                    # Remove axis labels
                    ax1.set_xlabel(None)
                    ax1.set_ylabel(None)
                    if ax2 is not None:
                        ax2.set_ylabel(None)

                    # Remove title
                    ax1.set_title("")

                    # Remove legend if present
                    leg = ax1.get_legend()
                    if leg is not None:
                        leg.remove()

                    # ---- Spines: keep ONLY bottom + left ----
                    for spine in ["bottom", "left"]:
                        ax1.spines[spine].set_visible(True)

                    for spine in ["top", "right"]:
                        ax1.spines[spine].set_visible(True)

                    if ax2 is not None:
                        # twin axis shares top + right spines → hide all of them
                        for spine in ["top", "right", "bottom", "left"]:
                            if spine in ax2.spines:
                                ax2.spines[spine].set_visible(False)
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

    # --- gather TAG rows across all geom + dye ---
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

    # --- choose which TAG metric to use ---
    tag_col = "TAG_slope_norm" if NORMALIZED_TAG else "TAG_slope"
    if tag_col not in merged.columns:
        print(f"[PHASE 3] column {tag_col} not found, aborting.")
        return

    merged_clean = merged.dropna(subset=["HMR", tag_col]).copy()
    if merged_clean.empty:
        print(f"[PHASE 3] no rows with both HMR and {tag_col}, aborting.")
        return

    # --- output directory / labels depending on normalized toggle ---
    base_fig_dir = "/Users/tejjolly/Documents/BioSimm/Meetings/2025-11-20--research_update/figures"
    subdir = "normalized" if NORMALIZED_TAG else "non-normalized"
    out_dir = os.path.join(base_fig_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    if NORMALIZED_TAG:
        y_label = "Normalized TAG slope [1/cm]"
    else:
        # y_label = "TAG slope [Concentration / cm]"
        y_label = "TAG slope"

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

    # ============================================================
    # COMBINED PLOT: all series together
    # ============================================================
    fig, ax = plt.subplots(figsize=FIGSIZE_PHASE3)

    # scatter, colored by (geom, dye)
    for _, row in merged_clean.iterrows():
        hmr = float(row["HMR"])
        tag_val = float(row[tag_col])
        geom = row["geom"]
        dye = row["dye"]

        color = series_colors.get((geom, dye), "k")

        ax.scatter(
            hmr, tag_val,
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
                Y = sub[tag_col].to_numpy()

                m, b = np.polyfit(X, Y, 1)
                y_pred = m * X + b
                ss_res = np.sum((Y - y_pred) ** 2)
                ss_tot = np.sum((Y - np.mean(Y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                print(f"[PHASE 3] fit ALL g{geom}, dye {dye}: slope={m}, intercept={b}, R2={r2}")

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
    ax.set_ylabel(y_label)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    combined_fname = os.path.join(out_dir, "TAGvHMR_all.svg")
    plt.savefig(combined_fname, format="svg", transparent=True, dpi=600)
    plt.show()

    # ============================================================
    # INDIVIDUAL PLOTS: one per (geom, dye) series
    # ============================================================
    for geom, dye in series_keys:
        sub = merged_clean[
            (merged_clean["geom"] == geom) &
            (merged_clean["dye"] == dye)
            ]
        if sub.empty:
            continue

        fig_s, ax_s = plt.subplots(figsize=FIGSIZE_PHASE3)

        # Extract R_micro (suffix) and sort
        suffixes = sub["suffix"].astype(str).tolist()
        r_values = [suffix_to_rmicro(sfx) for sfx in suffixes]
        ordered = sorted(zip(suffixes, r_values), key=lambda x: x[1])

        # Colormap: span entire BuPu range
        cmap = plt.get_cmap("BuPu")
        c_levels = np.linspace(0.2, 0.95, len(ordered))  # avoid too light/dark
        suffix_to_color = {
            sfx: cmap(level)
            for (sfx, _), level in zip(ordered, c_levels)
        }

        # scatter point-by-point, each with its own color
        for _, row in sub.iterrows():
            hmr = float(row["HMR"])
            tag_val = float(row[tag_col])
            sfx = str(row["suffix"])
            color = suffix_to_color.get(sfx, cmap(0.5))

            ax_s.scatter(
                hmr, tag_val,
                s=60,
                color=color,
                edgecolor="white",
                linewidths=0.6
            )

        # Optional: fit line (use darkest BuPu in range)
        if FIT_GEOMETRY and len(sub) >= 2:
            X = sub["HMR"].to_numpy()
            Y = sub[tag_col].to_numpy()

            m, b = np.polyfit(X, Y, 1)
            x_line = np.linspace(X.min(), X.max(), 50)
            y_line = m * x_line + b

            ax_s.plot(x_line, y_line, color=cmap(0.95), linewidth=1.6)

        # labels / styling (same convention)
        ax_s.set_xlabel("HMR [mmHg/cm/s]")
        ax_s.set_ylabel(y_label)

        ax_s.xaxis.set_label_position('top')
        ax_s.xaxis.tick_top()
        ax_s.spines['bottom'].set_visible(False)
        ax_s.spines['right'].set_visible(False)
        ax_s.grid(False)

        geom_name = GEOM_LABELS.get(geom, f"g{geom}")
        dye_name = DYE_LABELS.get(dye, dye)
        if PLOT_TITLES:
            ax_s.set_title(f"{geom_name}, {dye_name}")
        ax_s.set_ylim([-0.40,0])
        plt.tight_layout()
        fname = os.path.join(out_dir, f"TAGvHMR_g{geom}_d{dye}.svg")
        plt.savefig(fname, format="svg", transparent=True, dpi=600)
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
