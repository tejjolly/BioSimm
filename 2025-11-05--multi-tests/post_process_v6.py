#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# TOGGLES — change here in the IDE
# ============================================================
BASE = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance"
RUN_PHASES = [1, 2]          # e.g. [1,2,3] or [2,3] or [3]
RUN_GEOMETRIES = ["87"]      # order matters
PLOT_VELOCITY = False         # show velocity + secondary y in phase 2
THRESHOLD = 0            # slope: from max conc down to <= this
FIT_GEOMETRY = False          # fit HMR vs TAG per geometry in phase 3
NORMALIZE_CONC = True        # normalize phase-2 curves by their own max
NORMALIZED_TAG = False
PHASE1_MANUAL_TIME = None    # set a physical time to override the auto t_max lookup in phase 1
REVERSE_CENTERLINE_DIRECTION = True  # if True, treat the end of the current arc/profile as the physical inlet
PHASE2_ANIMATION_TIME_RANGE = (5.16, 26.66)  # set to (t_start, t_end) to save one frame per available time in that range
PHASE2_ANIMATION_OUTPUT_DIR = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
    "2025-11-05--multi-tests/images/animation"
)
PHASE2_MANUAL_SNAPSHOT_DIR = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
    "2025-11-05--multi-tests/images"
)
PHASE2_ANIMATION_DPI = 300
PHASE2_OUTPUT_FORMATS = ("png", "svg")
OVERWRITE_EXISTING = False  # if False, skip CSVs/frames that already exist; if True, regenerate them

PHASE2_FONTSIZE = 16         # controls only Phase-2 figure text

# Global-ish figure size controls
FIGSIZE_PHASE2 = (8, 5)      # for phase-2 concentration plots (original 8x5)
FIGSIZE_PHASE3 = (8, 5)      # for TAG vs HMR plots (combined + individual) (original 6x4)

PHASE2_CLEAN = False   # removes all axis text/labels/ticks/legend/title
PHASE2_LEGEND = False
PLOT_TITLES = False
PHASE2_BUPU_RANGE = (0.9, 1.0)  # min/max fraction of the BuPu colormap used in phase 2

# dyes / AIFs
DYES = [""]                  # empty string means no dye label in the case name
DYE_LABELS = {
    "": "no dye label",
    "B": "flat dye input @ time t",
    "N": "sharp dye input @ time t",
}

# label mapping for phase 3 legend
GEOM_LABELS = {
    "13": "Stenosis",
    "37": "No stenosis",
    "60": "Patient specific",
    "80": "Mass balance",
}

PHASE2_FONTSIZE = 16         # controls only Phase-2 figure text

# ============================================================

# ------------------------------------------------------------
# PATHS / CONFIG
# ------------------------------------------------------------
PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"
EXTRACT_SCRIPT = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
    "2025-11-05--multi-tests/extract_concentration_v2.py"
)

HMR_CSV = os.path.join(BASE, "hmr_data.csv")

# extractor defaults; override per case below if needed
DEFAULT_CHILD_FOLDER = "96-procs"
CASE_DIR_OVERRIDES = {}
CHILD_FOLDER_OVERRIDES = {}

# run suffixes per geometry (supports values like "24", "24_v3", "62_fast")
RUN_SUFFIXES = {
    "87": ["24"],
}

# arc-length file per geometry
ARC_PATHS = {
    "87": [
        os.path.join(BASE, "centerline_LCA.csv"),
        (
            "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
            "2025-11-05--multi-tests/centerlines/centerline_LCA.csv"
        ),
    ],
}

# concentration directory: e.g. BASE/concentrations/g13_r43_dB_concentration.csv
CONC_DIR = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2025-11-05--multi-tests/concentrations"

# ------------------------------------------------------------


def make_cmap_levels(level_range, count: int):
    start, stop = level_range
    if count <= 0:
        return np.array([])
    if count == 1:
        return np.array([(start + stop) / 2.0])
    return np.linspace(start, stop, count)


def suffix_to_rmicro(sfx: str) -> float:
    if not sfx:
        return np.nan
    match = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", sfx)
    if match is None:
        return np.nan
    return float(match.group(1)) / 100.0


def build_case_base(geom: str, suffix: str = "") -> str:
    case = f"g{geom}"
    if suffix:
        case += f"_r{suffix}"
    return case


def build_case_name(geom: str, suffix: str = "", dye: str = "") -> str:
    case = build_case_base(geom, suffix)
    if dye:
        case += f"_d{dye}"
    return case


def build_plot_stem(prefix: str, geom: str, dye: str = "") -> str:
    stem = f"{prefix}_{geom}"
    if dye:
        stem += f"_d{dye}"
    return stem


def build_extract_case_specs():
    case_specs = []

    for geom in RUN_GEOMETRIES:
        suffixes = RUN_SUFFIXES.get(geom)
        if suffixes is None:
            raise KeyError(f"No RUN_SUFFIXES configured for geometry {geom}")

        for suffix in suffixes:
            for dye in DYES:
                case_name = build_case_name(geom, suffix, dye)
                case_base = build_case_base(geom, suffix)
                case_dir = CASE_DIR_OVERRIDES.get(
                    case_name,
                    CASE_DIR_OVERRIDES.get(case_base, case_name),
                )
                child_folder = CHILD_FOLDER_OVERRIDES.get(
                    case_name,
                    CHILD_FOLDER_OVERRIDES.get(case_base, DEFAULT_CHILD_FOLDER),
                )
                case_specs.append({
                    "geom": geom,
                    "run_suffix": suffix,
                    "dye": dye,
                    "case_dir": case_dir,
                    "child_folder": child_folder,
                })

    return case_specs


def resolve_existing_path(path_config):
    if isinstance(path_config, (list, tuple)):
        for path in path_config:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find any configured path in {path_config}")
    return path_config


def load_arc_values(path_config):
    csv_path = resolve_existing_path(path_config)
    arc_df = pd.read_csv(csv_path)

    if "ArcLength" in arc_df.columns:
        arc_values = arc_df["ArcLength"].to_numpy(dtype=float)
        arc_values = arc_values[np.isfinite(arc_values)]
        if arc_values.size > 0:
            return orient_arc_values(arc_values)

    point_column_sets = [
        ("Points_0", "Points_1", "Points_2"),
        ("Points:0", "Points:1", "Points:2"),
    ]
    point_cols = next(
        (cols for cols in point_column_sets if all(col in arc_df.columns for col in cols)),
        None,
    )
    if point_cols is None:
        raise ValueError(
            f"Could not find ArcLength or point columns in {csv_path}: {list(arc_df.columns)}"
        )

    points = arc_df.loc[:, point_cols].to_numpy(dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        raise ValueError(f"No finite centerline points found in {csv_path}")

    arc = np.zeros(len(points))
    for i in range(1, len(points)):
        arc[i] = arc[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    return orient_arc_values(arc)


def orient_arc_values(arc_values):
    arc = np.asarray(arc_values, dtype=float)
    arc = arc[np.isfinite(arc)]
    if arc.size == 0:
        return arc

    if arc[-1] >= arc[0]:
        arc = arc - arc[0]
    else:
        arc = arc[0] - arc
    if REVERSE_CENTERLINE_DIRECTION:
        return arc[-1] - arc[::-1]
    return arc


def load_centerline_profile(df: pd.DataFrame, n_arc: int) -> pd.DataFrame:
    prof = df.iloc[:n_arc].reset_index(drop=True)
    if REVERSE_CENTERLINE_DIRECTION:
        prof = prof.iloc[::-1].reset_index(drop=True)
    return prof


def format_suffix_label(geom: str, suffix: str) -> str:
    if not suffix:
        return build_case_base(geom, suffix)
    r_value = suffix_to_rmicro(suffix)
    if np.isfinite(r_value):
        return f"R_micro = {r_value:.2f}"
    return build_case_base(geom, suffix)


def sort_suffixes(suffixes):
    def sort_key(sfx):
        if not sfx:
            return (0, -1.0, "")
        r_value = suffix_to_rmicro(sfx)
        if np.isfinite(r_value):
            return (1, r_value, sfx)
        return (2, float("inf"), sfx)

    return sorted(suffixes, key=sort_key)


def make_geom_cmap(geom_index: int):
    # kept for backward compatibility; not used for Phase 2 anymore
    if geom_index == 1:
        return plt.get_cmap("BuPu")
    else:
        return plt.get_cmap("OrRd")


def phase1_extract():
    print("[PHASE 1] running pvpython extractor …")
    case_specs = build_extract_case_specs()
    print(f"[PHASE 1] prepared {len(case_specs)} case spec(s) from post-process config")

    cmd = [
        PV_PYTHON,
        EXTRACT_SCRIPT,
        "--case-specs-json",
        json.dumps(case_specs, separators=(",", ":")),
    ]
    if OVERWRITE_EXISTING:
        cmd.append("--overwrite")
    if phase2_animation_enabled():
        cmd.append("--save-all-timesteps")
        t_start, t_end = _phase2_animation_time_bounds()
        cmd.extend(["--time-range", str(t_start), str(t_end)])
        print(
            "[PHASE 1] animation mode active; exporting time-series concentration CSVs "
            f"for range [{t_start}, {t_end}]"
        )
    elif PHASE1_MANUAL_TIME is not None:
        cmd.extend(["--manual-time", str(PHASE1_MANUAL_TIME)])
        print(f"[PHASE 1] overriding auto t_max with manual time {PHASE1_MANUAL_TIME}")
    subprocess.run(cmd, check=True)
    print("[PHASE 1] done.")


def _conc_csv_path(geom: str, suffix: str, dye: str, full_series: bool = False) -> str:
    """
    Build path like:
    BASE/concentrations/g13_r43_dB_concentration.csv
    or, for no suffix / no dye cases:
    BASE/concentrations/g80_concentration.csv
    """
    case_name = build_case_name(geom, suffix, dye)
    if full_series:
        fname = f"{case_name}_concentration_timeseries.csv"
    else:
        fname = f"{case_name}_concentration.csv"
    return os.path.join(CONC_DIR, fname)


def phase2_animation_enabled() -> bool:
    return PHASE2_ANIMATION_TIME_RANGE is not None


def _phase2_animation_time_bounds():
    if not phase2_animation_enabled():
        return None, None
    t0, t1 = PHASE2_ANIMATION_TIME_RANGE
    return (t0, t1) if t0 <= t1 else (t1, t0)


def phase2_manual_snapshot_enabled() -> bool:
    return (not phase2_animation_enabled()) and (PHASE1_MANUAL_TIME is not None)


def first_existing_column(columns, candidates):
    return next((col for col in candidates if col in columns), None)


def load_time_series_profiles(geom: str, suffix: str, dye: str, arc_all):
    case_dye = build_case_name(geom, suffix, dye)
    csv_path = _conc_csv_path(geom, suffix, dye, full_series=True)
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found, skipping {case_dye}")
        return []

    df = pd.read_csv(csv_path)
    if "Concentration" not in df.columns:
        print(f"  [WARN] no 'Concentration' column in {csv_path}, skipping")
        return []
    time_col = first_existing_column(df.columns, ["Time", "time"])
    if time_col is None:
        print(f"  [WARN] no 'Time' column in {csv_path}; rerun phase 1 in animation mode")
        return []

    n_arc = len(arc_all)
    timestep_col = first_existing_column(df.columns, ["TimeStep", "Time Step", "time_step"])
    group_col = timestep_col if timestep_col is not None else time_col
    frame_ids = df[group_col].drop_duplicates().tolist()

    frames = []
    for frame_id in frame_ids:
        frame_raw = df.loc[df[group_col] == frame_id].reset_index(drop=True)
        if frame_raw.empty:
            continue

        prof = load_centerline_profile(frame_raw, n_arc)
        conc = prof["Concentration"].to_numpy()
        if conc.size == 0:
            continue

        x_arc = arc_all[:len(conc)]
        time_value = float(frame_raw[time_col].iloc[0])
        if timestep_col is not None:
            frame_key = int(frame_raw[timestep_col].iloc[0])
        else:
            frame_key = len(frames)

        frames.append({
            "frame_key": frame_key,
            "time": time_value,
            "x_arc": x_arc,
            "conc": conc,
            "prof": prof,
        })

    return frames


def select_time_range_frames(frames):
    t_start, t_end = _phase2_animation_time_bounds()
    if t_start is None:
        return frames
    selected = [frame for frame in frames if t_start <= frame["time"] <= t_end]
    return selected


def format_time_token(time_value: float) -> str:
    return f"{time_value:010.4f}".replace("-", "m").replace(".", "p")


def build_output_paths(directory: str, stem: str, formats):
    return [os.path.join(directory, f"{stem}.{fmt}") for fmt in formats]


def outputs_exist(paths) -> bool:
    return all(os.path.exists(path) for path in paths)


def save_figure_in_formats(fig, directory: str, stem: str, formats, dpi: int, transparent: bool = True):
    for fmt in formats:
        out_path = os.path.join(directory, f"{stem}.{fmt}")
        fig.savefig(out_path, format=fmt, dpi=dpi, transparent=transparent)


def apply_phase2_clean_mode(ax1, ax2):
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    if ax2 is not None:
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    ax1.set_xlabel(None)
    ax1.set_ylabel(None)
    if ax2 is not None:
        ax2.set_ylabel(None)

    ax1.set_title("")

    leg = ax1.get_legend()
    if leg is not None:
        leg.remove()

    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_visible(True)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(True)

    if ax2 is not None:
        for spine in ["top", "right", "bottom", "left"]:
            if spine in ax2.spines:
                ax2.spines[spine].set_visible(False)


def archive_existing_animation_frames():
    archive_dir = os.path.join(PHASE2_ANIMATION_OUTPUT_DIR, "_ss")
    os.makedirs(archive_dir, exist_ok=True)

    moved_count = 0
    for name in os.listdir(PHASE2_ANIMATION_OUTPUT_DIR):
        src_path = os.path.join(PHASE2_ANIMATION_OUTPUT_DIR, name)
        if name == "_ss" or not os.path.isfile(src_path):
            continue

        dst_path = os.path.join(archive_dir, name)
        if os.path.exists(dst_path):
            stem, ext = os.path.splitext(name)
            counter = 1
            while True:
                candidate = os.path.join(archive_dir, f"{stem}_{counter:03d}{ext}")
                if not os.path.exists(candidate):
                    dst_path = candidate
                    break
                counter += 1

        shutil.move(src_path, dst_path)
        moved_count += 1

    if moved_count > 0:
        print(f"[PHASE 2] archived {moved_count} existing frame(s) to {archive_dir}")


def phase2_save_animation_frames():
    print("[PHASE 2] saving animation frames")
    os.makedirs(PHASE2_ANIMATION_OUTPUT_DIR, exist_ok=True)
    if OVERWRITE_EXISTING:
        archive_existing_animation_frames()
    t_start, t_end = _phase2_animation_time_bounds()
    print(f"[PHASE 2] time range = [{t_start}, {t_end}]")

    for geom in RUN_GEOMETRIES:
        print(f"[PHASE 2] geometry g{geom}")
        arc_all = load_arc_values(ARC_PATHS[geom])
        suffixes = RUN_SUFFIXES[geom]
        ordered = sort_suffixes(suffixes)

        cmap = plt.get_cmap("BuPu")
        c_levels = make_cmap_levels(PHASE2_BUPU_RANGE, len(ordered))
        suffix_to_color = {
            sfx: cmap(level)
            for sfx, level in zip(ordered, c_levels)
        }

        for dye in DYES:
            print(f"[PHASE 2]  dye {dye}")
            series_frames = {}
            group_max = 0.0
            velocity_max = 0.0
            frame_catalog = {}

            for suffix in suffixes:
                case_base = build_case_base(geom, suffix)
                case_dye = build_case_name(geom, suffix, dye)
                frames = load_time_series_profiles(geom, suffix, dye, arc_all)
                frames = select_time_range_frames(frames)
                if not frames:
                    print(f"  [WARN] no frames found in range for {case_dye}")
                    continue

                frame_map = {}
                for frame in frames:
                    frame_map[frame["frame_key"]] = frame
                    frame_catalog[frame["frame_key"]] = frame["time"]

                    conc_max = float(np.nanmax(frame["conc"]))
                    if conc_max > group_max:
                        group_max = conc_max

                    if PLOT_VELOCITY and all(
                        col in frame["prof"].columns for col in ["Velocity:0", "Velocity:1", "Velocity:2"]
                    ):
                        vel_mag = np.sqrt(
                            frame["prof"]["Velocity:0"] ** 2 +
                            frame["prof"]["Velocity:1"] ** 2 +
                            frame["prof"]["Velocity:2"] ** 2
                        ).to_numpy()
                        vel_frame_max = float(np.nanmax(vel_mag))
                        if vel_frame_max > velocity_max:
                            velocity_max = vel_frame_max

                series_frames[suffix] = {
                    "case_base": case_base,
                    "case_dye": case_dye,
                    "frames": frame_map,
                }

            if not series_frames:
                print(f"  [WARN] no time-series data available for geometry g{geom}, dye {dye}")
                continue

            if group_max <= 0:
                print(f"  [WARN] non-positive concentration max across selected range for g{geom}, dye {dye}")
                continue

            y_top = 1.0 if NORMALIZE_CONC else group_max * 1.02
            frame_keys = sorted(frame_catalog, key=lambda key: frame_catalog[key])
            print(f"  [INFO] saving {len(frame_keys)} frames to {PHASE2_ANIMATION_OUTPUT_DIR}")

            with plt.rc_context({
                "font.size": PHASE2_FONTSIZE,
                "axes.labelsize": PHASE2_FONTSIZE,
                "xtick.labelsize": PHASE2_FONTSIZE,
                "ytick.labelsize": PHASE2_FONTSIZE,
                "legend.fontsize": max(6, PHASE2_FONTSIZE - 2),
            }):
                for frame_index, frame_key in enumerate(frame_keys):
                    frame_time = frame_catalog[frame_key]
                    stem = build_plot_stem("tag", geom, dye)
                    frame_stem = f"{stem}_frame_{frame_index:04d}_t{format_time_token(frame_time)}"
                    out_paths = build_output_paths(
                        PHASE2_ANIMATION_OUTPUT_DIR, frame_stem, PHASE2_OUTPUT_FORMATS
                    )
                    if outputs_exist(out_paths) and not OVERWRITE_EXISTING:
                        print(f"  [SKIP] frame already exists: {frame_stem}")
                        continue

                    fig, ax1 = plt.subplots(figsize=FIGSIZE_PHASE2)
                    ax2 = ax1.twinx() if PLOT_VELOCITY else None

                    for suffix in suffixes:
                        if suffix not in series_frames:
                            continue
                        frame = series_frames[suffix]["frames"].get(frame_key)
                        if frame is None:
                            continue

                        color = suffix_to_color[suffix]
                        conc = frame["conc"]
                        prof = frame["prof"]
                        x_arc = frame["x_arc"]
                        if NORMALIZE_CONC and group_max > 0:
                            conc_plot = conc / group_max
                        else:
                            conc_plot = conc

                        ax1.plot(
                            x_arc, conc_plot,
                            label=series_frames[suffix]["case_base"],
                            alpha=1.0,
                            color=color,
                            linewidth=3,
                        )

                        if PLOT_VELOCITY and ax2 is not None:
                            if all(col in prof.columns for col in ["Velocity:0", "Velocity:1", "Velocity:2"]):
                                vel_mag = np.sqrt(
                                    prof["Velocity:0"] ** 2 +
                                    prof["Velocity:1"] ** 2 +
                                    prof["Velocity:2"] ** 2
                                )
                                ax2.plot(x_arc, vel_mag, linestyle=":", color=color, linewidth=2)

                    ax1.set_xlabel("Centerline length [cm]")
                    label_y = "Concentration" + (" (normalized to range max)" if NORMALIZE_CONC else "")
                    ax1.set_ylabel(label_y)
                    if PLOT_VELOCITY and ax2 is not None:
                        ax2.set_ylabel("Velocity magnitude")
                        if velocity_max > 0:
                            ax2.set_ylim(0, velocity_max * 1.02)

                    ax1.set_ylim(0, y_top)
                    ax1.set_xlim(left=0)

                    handles = [
                        Line2D([0], [0], color=suffix_to_color[sfx], lw=3,
                               label=format_suffix_label(geom, sfx))
                        for sfx in ordered
                        if sfx in series_frames
                    ]
                    if PHASE2_LEGEND and handles:
                        ax1.legend(handles=handles, loc="best")

                    geom_name = GEOM_LABELS.get(geom, f"g{geom}")
                    dye_name = DYE_LABELS.get(dye, dye or "no dye label")
                    if PLOT_TITLES:
                        title = geom_name if not dye else f"{geom_name}, {dye_name}"
                        ax1.set_title(f"{title}  t = {frame_time:.4f}")

                    if PHASE2_CLEAN:
                        apply_phase2_clean_mode(ax1, ax2)

                    fig.tight_layout()
                    save_figure_in_formats(
                        fig,
                        PHASE2_ANIMATION_OUTPUT_DIR,
                        frame_stem,
                        PHASE2_OUTPUT_FORMATS,
                        dpi=PHASE2_ANIMATION_DPI,
                        transparent=True,
                    )
                    plt.close(fig)

    print("[PHASE 2] animation frames done.")


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
    arc_all = load_arc_values(ARC_PATHS[geom])
    n_arc = len(arc_all)

    temp_entries = []
    group_max = 0.0

    # -------- FIRST PASS: collect segments and find group_max --------
    for suffix in RUN_SUFFIXES[geom]:
        case_base = build_case_base(geom, suffix)
        case_dye = build_case_name(geom, suffix, dye)
        csv_path = _conc_csv_path(geom, suffix, dye)

        if not os.path.exists(csv_path):
            print(f"  [WARN] {csv_path} not found, skipping {case_dye}")
            continue

        df = pd.read_csv(csv_path)
        if "Concentration" not in df.columns:
            print(f"  [WARN] no 'Concentration' column in {csv_path}, skipping")
            continue

        # Use first n_arc points along the centerline, re-oriented if needed
        prof = load_centerline_profile(df, n_arc)

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
    if phase2_animation_enabled():
        phase2_save_animation_frames()
        return

    print("[PHASE 2] plotting concentration (and maybe velocity)")

    for geom in RUN_GEOMETRIES:
        print(f"[PHASE 2] geometry g{geom}")

        # arc-length for this geometry
        arc_all = load_arc_values(ARC_PATHS[geom])
        n_arc = len(arc_all)

        suffixes = RUN_SUFFIXES[geom]
        ordered = sort_suffixes(suffixes)

        # Use BuPu for ALL concentration plots
        cmap = plt.get_cmap("BuPu")
        c_levels = make_cmap_levels(PHASE2_BUPU_RANGE, len(ordered))
        suffix_to_color = {
            sfx: cmap(level)
            for sfx, level in zip(ordered, c_levels)
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
                case_base = build_case_base(geom, suffix)
                case_dye = build_case_name(geom, suffix, dye)
                csv_path = _conc_csv_path(geom, suffix, dye)

                if not os.path.exists(csv_path):
                    print(f"  [WARN] {csv_path} not found, skipping {case_dye}")
                    continue

                prof_all = pd.read_csv(csv_path)
                if "Concentration" not in prof_all.columns:
                    print(f"  [WARN] no 'Concentration' column in {csv_path}, skipping")
                    continue

                # treat entire file as single snapshot along centerline, re-oriented if needed
                prof = load_centerline_profile(prof_all, n_arc)
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
                        alpha=1.0,
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
                        ax1.plot(x_seg, y_fit_plot, linestyle="--", color=color, linewidth=0.01)
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
                           label=format_suffix_label(geom, sfx))
                    for sfx in ordered
                ]
                if PHASE2_LEGEND:
                    ax1.legend(handles=handles, loc="best")

                # Title: geom label + dye
                geom_name = GEOM_LABELS.get(geom, f"g{geom}")
                dye_name = DYE_LABELS.get(dye, dye or "no dye label")
                if PLOT_TITLES:
                    title = geom_name if not dye else f"{geom_name}, {dye_name}"
                    ax1.set_title(title)
                # ------------------------------------------------------------
                # PHASE 2 CLEAN MODE (for inset usage)
                # ------------------------------------------------------------
                if PHASE2_CLEAN:
                    apply_phase2_clean_mode(ax1, ax2)
                fig.tight_layout()
                out_name = f"{build_plot_stem('tag', geom, dye)}.svg"
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
                if phase2_manual_snapshot_enabled():
                    os.makedirs(PHASE2_MANUAL_SNAPSHOT_DIR, exist_ok=True)
                    manual_stem = (
                        f"{build_plot_stem('tag', geom, dye)}_t{format_time_token(PHASE1_MANUAL_TIME)}"
                    )
                    manual_out_paths = build_output_paths(
                        PHASE2_MANUAL_SNAPSHOT_DIR, manual_stem, PHASE2_OUTPUT_FORMATS
                    )
                    if OVERWRITE_EXISTING or not outputs_exist(manual_out_paths):
                        save_figure_in_formats(
                            fig,
                            PHASE2_MANUAL_SNAPSHOT_DIR,
                            manual_stem,
                            PHASE2_OUTPUT_FORMATS,
                            dpi=PHASE2_ANIMATION_DPI,
                            transparent=True,
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
        dye_name = DYE_LABELS.get(dye, dye or "no dye label")
        label = geom_name if not dye else f"{geom_name}, {dye_name}"
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
        ordered = sort_suffixes(suffixes)

        # Colormap: span entire BuPu range
        cmap = plt.get_cmap("BuPu")
        c_levels = np.linspace(0.2, 0.95, len(ordered))  # avoid too light/dark
        suffix_to_color = {
            sfx: cmap(level)
            for sfx, level in zip(ordered, c_levels)
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
        dye_name = DYE_LABELS.get(dye, dye or "no dye label")
        if PLOT_TITLES:
            title = geom_name if not dye else f"{geom_name}, {dye_name}"
            ax_s.set_title(title)
        ax_s.set_ylim([-0.40,0])
        plt.tight_layout()
        fname = os.path.join(out_dir, f"TAGvHMR_{build_case_name(geom, '', dye)}.svg")
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
