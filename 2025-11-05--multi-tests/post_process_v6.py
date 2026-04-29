#!/usr/bin/env python3
import csv
import json
import os
import shutil
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from helper_scripts.case_utils import (
    add_output_paths_to_case_specs,
    build_case_base,
    build_case_name,
    concentration_csv_path,
    expand_case_specs,
    resolve_existing_path,
    suffix_to_rmicro,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# TOGGLES — change here in the IDE
# ============================================================
# BASE = "/Volumes/maxone/2026-02-03--mass_balance"
BASE = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance"
RUN_PHASES = [1,2]          # e.g. [1,2,3] or [2,3] or [3]
CASES = [
    "g87_r24",
]
PLOT_VELOCITY = False         # show velocity + secondary y in phase 2
THRESHOLD = 0            # slope: from max conc down to <= this
FIT_GEOMETRY = False          # fit HMR vs TAG per geometry in phase 3
NORMALIZE_CONC = True        # normalize phase-2 curves by their own max
NORMALIZED_TAG = False
PHASE1_MANUAL_TIME = None    # set a physical time to override the auto t_max lookup in phase 1
REVERSE_CENTERLINE_DIRECTION = True  # if True, treat the end of the current arc/profile as the physical inlet
# PHASE2_ANIMATION_TIME_RANGE = (1.72, 24.08)  # set to (t_start, t_end) to save one frame per available time in that range
PHASE2_ANIMATION_TIME_RANGE = (4.3, 26.6)  # set to (t_start, t_end) to save one frame per available time in that range
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

# Legacy Phase-3 grouping knobs. Phase 1/2 use CASES above.
RUN_GEOMETRIES = ["87"]      # order matters for phase 3
RUN_SUFFIXES = {
    "87": ["24"],
}
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

# ============================================================

# ------------------------------------------------------------
# PATHS / CONFIG
# ------------------------------------------------------------
PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "helper_scripts", "extract_centerline_concentration.py")

HMR_CSV = os.path.join(BASE, "hmr_data.csv")

# extractor defaults; override per case below if needed
DEFAULT_CHILD_FOLDER = "96-procs"
CASE_DIR_OVERRIDES = {}
CHILD_FOLDER_OVERRIDES = {}

TAG_OUTPUT_FOLDER = "TAG"
CONCENTRATION_SUBDIR = "concentrations"
PHASE2_ANIMATION_SUBDIR = "animation"
PHASE2_STATIC_SUBDIR = "plots"
PHASE2_MANUAL_SNAPSHOT_SUBDIR = "snapshots"
PHASE3_SUBDIR = "tag_vs_hmr"
PHASE2_ANIMATION_METADATA_CSV = "_frame_metadata.csv"
PHASE2_FRAME_PROGRESS_EVERY = 25

ARC_PATH = os.path.join(BASE, "centerline_LCA.csv")

# ------------------------------------------------------------


def normalize_phase12_case(raw_case):
    if isinstance(raw_case, str):
        raw_case = {
            "case_dir": raw_case,
            "label": raw_case,
        }

    case_dir = str(raw_case["case_dir"])
    case_id = str(raw_case.get("case_id", os.path.basename(os.path.normpath(case_dir))))
    if case_id.startswith("g"):
        case_id = case_id[1:]

    spec = {
        "geom": str(raw_case.get("geom", case_id)),
        "run_suffix": str(raw_case.get("run_suffix", "")),
        "dye": str(raw_case.get("dye", "")),
        "case_dir": case_dir,
        "child_folder": str(raw_case.get("child_folder", DEFAULT_CHILD_FOLDER)),
        "label": str(raw_case.get("label", case_dir)),
    }
    if raw_case.get("step_override") is not None:
        spec["step_override"] = int(raw_case["step_override"])
    if raw_case.get("time_override") is not None:
        spec["time_override"] = float(raw_case["time_override"])
    return spec


def build_extract_case_specs():
    return add_output_paths_to_case_specs(
        [normalize_phase12_case(case) for case in CASES],
        BASE,
        tag_output_folder=TAG_OUTPUT_FOLDER,
        concentration_subdir=CONCENTRATION_SUBDIR,
    )


def build_phase3_case_specs():
    return add_output_paths_to_case_specs(
        expand_case_specs(
            RUN_GEOMETRIES,
            RUN_SUFFIXES,
            DYES,
            default_child_folder=DEFAULT_CHILD_FOLDER,
            case_dir_overrides=CASE_DIR_OVERRIDES,
            child_folder_overrides=CHILD_FOLDER_OVERRIDES,
        ),
        BASE,
        tag_output_folder=TAG_OUTPUT_FOLDER,
        concentration_subdir=CONCENTRATION_SUBDIR,
    )


_CASE_SPECS_CACHE = None
_PHASE3_CASE_SPECS_CACHE = None


def get_case_specs():
    global _CASE_SPECS_CACHE
    if _CASE_SPECS_CACHE is None:
        _CASE_SPECS_CACHE = build_extract_case_specs()
    return _CASE_SPECS_CACHE


def get_phase3_case_specs():
    global _PHASE3_CASE_SPECS_CACHE
    if _PHASE3_CASE_SPECS_CACHE is None:
        _PHASE3_CASE_SPECS_CACHE = build_phase3_case_specs()
    return _PHASE3_CASE_SPECS_CACHE


def case_spec_key(geom: str, suffix: str, dye: str):
    return (str(geom), str(suffix), str(dye))


def get_phase3_case_spec(geom: str, suffix: str, dye: str):
    key = case_spec_key(geom, suffix, dye)
    case_specs = {
        case_spec_key(spec["geom"], spec["run_suffix"], spec["dye"]): spec
        for spec in get_phase3_case_specs()
    }
    if key not in case_specs:
        raise KeyError(f"No phase-3 case spec configured for geom={geom}, suffix={suffix}, dye={dye}")
    return case_specs[key]


def concentration_csv_for_case_spec(spec, full_series: bool = False):
    return concentration_csv_path(
        spec["concentration_dir"],
        spec["geom"],
        spec["run_suffix"],
        spec["dye"],
        full_series=full_series,
    )


def concentration_csv_for_phase3_case(geom: str, suffix: str, dye: str, full_series: bool = False):
    return concentration_csv_for_case_spec(
        get_phase3_case_spec(geom, suffix, dye),
        full_series=full_series,
    )


def output_dir_for_case_spec(spec, *subdirs):
    return os.path.join(spec["tag_dir"], *subdirs)


def output_dir_for_phase3_geom_dye(geom: str, dye: str, *subdirs):
    suffixes = RUN_SUFFIXES.get(geom, [])
    if not suffixes:
        raise KeyError(f"No run suffixes configured for geometry {geom}")
    spec = get_phase3_case_spec(geom, suffixes[0], dye)
    return output_dir_for_case_spec(spec, *subdirs)


def case_display_name(spec):
    return spec.get("label") or build_case_name(spec["geom"], spec["run_suffix"], spec["dye"])


def tag_plot_stem(dye: str = ""):
    return "tag" if not dye else f"tag_d{dye}"


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


def sort_suffixes(suffixes):
    def sort_key(sfx):
        if not sfx:
            return (0, -1.0, "")
        r_value = suffix_to_rmicro(sfx)
        if np.isfinite(r_value):
            return (1, r_value, sfx)
        return (2, float("inf"), sfx)

    return sorted(suffixes, key=sort_key)


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


def load_time_series_profiles(case_spec, arc_all):
    case_label = case_display_name(case_spec)
    csv_path = concentration_csv_for_case_spec(case_spec, full_series=True)
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found, skipping {case_label}")
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


def animation_metadata_path(animation_output_dir: str):
    return os.path.join(animation_output_dir, PHASE2_ANIMATION_METADATA_CSV)


def load_saved_animation_group_max(animation_output_dir: str):
    metadata_path = animation_metadata_path(animation_output_dir)
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)

    if row is None or "group_max" not in row:
        return None

    try:
        return float(row["group_max"])
    except (TypeError, ValueError):
        return None


def write_animation_group_max(animation_output_dir: str, group_max: float):
    metadata_path = animation_metadata_path(animation_output_dir)
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_max"])
        writer.writeheader()
        writer.writerow({"group_max": f"{group_max:.17g}"})


def has_existing_animation_frame_files(animation_output_dir: str):
    frame_extensions = tuple(f".{fmt.lower()}" for fmt in PHASE2_OUTPUT_FORMATS)
    for name in os.listdir(animation_output_dir):
        path = os.path.join(animation_output_dir, name)
        if os.path.isfile(path) and name.lower().endswith(frame_extensions):
            return True
    return False


def animation_group_max_is_stale(animation_output_dir: str, group_max: float):
    saved_group_max = load_saved_animation_group_max(animation_output_dir)
    if saved_group_max is None:
        if has_existing_animation_frame_files(animation_output_dir):
            print("  [INFO] Existing animation frames have no group_max metadata; regenerating.")
            return True
        return False

    if np.isclose(saved_group_max, group_max, rtol=1.0e-12, atol=1.0e-12):
        return False

    print(
        "  [INFO] Animation group_max changed "
        f"from {saved_group_max:.12g} to {group_max:.12g}; regenerating frames."
    )
    return True


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


def archive_existing_animation_frames(animation_output_dir: str):
    archive_dir = os.path.join(animation_output_dir, "_ss")
    os.makedirs(archive_dir, exist_ok=True)

    moved_count = 0
    for name in os.listdir(animation_output_dir):
        src_path = os.path.join(animation_output_dir, name)
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


def maybe_print_frame_progress(processed, total, saved, skipped, force=False):
    if total <= 0:
        return
    if not force and processed % PHASE2_FRAME_PROGRESS_EVERY != 0:
        return
    print(
        f"  [PROGRESS] Frames {processed}/{total} processed; "
        f"saved={saved}, skipped={skipped}",
        flush=True,
    )


def phase2_save_animation_frames():
    print("[PHASE 2] saving animation frames")
    t_start, t_end = _phase2_animation_time_bounds()
    print(f"[PHASE 2] time range = [{t_start}, {t_end}]")
    arc_all = load_arc_values(ARC_PATH)
    case_specs = get_case_specs()
    cmap = plt.get_cmap("BuPu")
    color = cmap((PHASE2_BUPU_RANGE[0] + PHASE2_BUPU_RANGE[1]) / 2.0)

    for case_spec in case_specs:
        case_label = case_display_name(case_spec)
        dye = case_spec.get("dye", "")
        print(f"[PHASE 2] case {case_label}")
        animation_output_dir = output_dir_for_case_spec(case_spec, PHASE2_ANIMATION_SUBDIR)
        os.makedirs(animation_output_dir, exist_ok=True)
        if OVERWRITE_EXISTING:
            archive_existing_animation_frames(animation_output_dir)

        frames = select_time_range_frames(load_time_series_profiles(case_spec, arc_all))
        if not frames:
            print(f"  [WARN] no frames found in range for {case_label}")
            continue

        group_max = max(float(np.nanmax(frame["conc"])) for frame in frames)
        velocity_max = 0.0
        if PLOT_VELOCITY:
            for frame in frames:
                if all(col in frame["prof"].columns for col in ["Velocity:0", "Velocity:1", "Velocity:2"]):
                    vel_mag = np.sqrt(
                        frame["prof"]["Velocity:0"] ** 2 +
                        frame["prof"]["Velocity:1"] ** 2 +
                        frame["prof"]["Velocity:2"] ** 2
                    ).to_numpy()
                    velocity_max = max(velocity_max, float(np.nanmax(vel_mag)))

        if group_max <= 0:
            print(f"  [WARN] non-positive concentration max across selected range for {case_label}")
            continue

        if not OVERWRITE_EXISTING and animation_group_max_is_stale(animation_output_dir, group_max):
            archive_existing_animation_frames(animation_output_dir)

        write_animation_group_max(animation_output_dir, group_max)

        y_top = 1.0 if NORMALIZE_CONC else group_max * 1.02
        frames = sorted(frames, key=lambda frame: frame["time"])
        total_frames = len(frames)
        saved_frames = 0
        skipped_frames = 0
        print(f"  [INFO] saving {total_frames} frames to {animation_output_dir}")

        with plt.rc_context({
            "font.size": PHASE2_FONTSIZE,
            "axes.labelsize": PHASE2_FONTSIZE,
            "xtick.labelsize": PHASE2_FONTSIZE,
            "ytick.labelsize": PHASE2_FONTSIZE,
            "legend.fontsize": max(6, PHASE2_FONTSIZE - 2),
        }):
            for frame_index, frame in enumerate(frames):
                frame_time = frame["time"]
                stem = tag_plot_stem(dye)
                frame_stem = f"{stem}_frame_{frame_index:04d}_t{format_time_token(frame_time)}"
                out_paths = build_output_paths(
                    animation_output_dir, frame_stem, PHASE2_OUTPUT_FORMATS
                )
                processed_frames = frame_index + 1
                if outputs_exist(out_paths) and not OVERWRITE_EXISTING:
                    skipped_frames += 1
                    maybe_print_frame_progress(
                        processed_frames,
                        total_frames,
                        saved_frames,
                        skipped_frames,
                        force=processed_frames == total_frames,
                    )
                    continue

                fig, ax1 = plt.subplots(figsize=FIGSIZE_PHASE2)
                ax2 = ax1.twinx() if PLOT_VELOCITY else None

                conc = frame["conc"]
                prof = frame["prof"]
                x_arc = frame["x_arc"]
                conc_plot = conc / group_max if NORMALIZE_CONC and group_max > 0 else conc

                ax1.plot(
                    x_arc,
                    conc_plot,
                    label=case_label,
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
                label_y = "Concentration" + (" (normalized)" if NORMALIZE_CONC else "")
                ax1.set_ylabel(label_y)
                if PLOT_VELOCITY and ax2 is not None:
                    ax2.set_ylabel("Velocity magnitude")
                    if velocity_max > 0:
                        ax2.set_ylim(0, velocity_max * 1.02)

                ax1.set_ylim(0, y_top)
                ax1.set_xlim(left=0)

                if PHASE2_LEGEND:
                    ax1.legend(loc="best")

                if PLOT_TITLES:
                    ax1.set_title(f"{case_label}  t = {frame_time:.4f}")

                if PHASE2_CLEAN:
                    apply_phase2_clean_mode(ax1, ax2)

                fig.tight_layout()
                save_figure_in_formats(
                    fig,
                    animation_output_dir,
                    frame_stem,
                    PHASE2_OUTPUT_FORMATS,
                    dpi=PHASE2_ANIMATION_DPI,
                    transparent=True,
                )
                plt.close(fig)
                saved_frames += 1
                maybe_print_frame_progress(
                    processed_frames,
                    total_frames,
                    saved_frames,
                    skipped_frames,
                    force=processed_frames == total_frames,
                )

        write_animation_group_max(animation_output_dir, group_max)

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
    arc_all = load_arc_values(ARC_PATH)
    n_arc = len(arc_all)

    temp_entries = []
    group_max = 0.0

    # -------- FIRST PASS: collect segments and find group_max --------
    for suffix in RUN_SUFFIXES[geom]:
        case_base = build_case_base(geom, suffix)
        case_dye = build_case_name(geom, suffix, dye)
        csv_path = concentration_csv_for_phase3_case(geom, suffix, dye)

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
    arc_all = load_arc_values(ARC_PATH)
    n_arc = len(arc_all)
    cmap = plt.get_cmap("BuPu")
    color = cmap((PHASE2_BUPU_RANGE[0] + PHASE2_BUPU_RANGE[1]) / 2.0)

    for case_spec in get_case_specs():
        case_label = case_display_name(case_spec)
        dye = case_spec.get("dye", "")
        print(f"[PHASE 2] case {case_label}")

        csv_path = concentration_csv_for_case_spec(case_spec)
        if not os.path.exists(csv_path):
            print(f"  [WARN] {csv_path} not found, skipping {case_label}")
            continue

        prof_all = pd.read_csv(csv_path)
        if "Concentration" not in prof_all.columns:
            print(f"  [WARN] no 'Concentration' column in {csv_path}, skipping")
            continue

        prof = load_centerline_profile(prof_all, n_arc)
        conc = prof["Concentration"].to_numpy()
        if conc.size == 0:
            print(f"  [WARN] empty concentration profile for {case_label}, skipping")
            continue

        group_max = float(np.nanmax(conc))
        print(f"  {case_label}: C_max_snapshot={group_max}")
        x_arc = arc_all[:len(conc)]

        with plt.rc_context({
            "font.size": PHASE2_FONTSIZE,
            "axes.labelsize": PHASE2_FONTSIZE,
            "xtick.labelsize": PHASE2_FONTSIZE,
            "ytick.labelsize": PHASE2_FONTSIZE,
            "legend.fontsize": max(6, PHASE2_FONTSIZE - 2),
        }):
            fig, ax1 = plt.subplots(figsize=FIGSIZE_PHASE2)
            ax2 = ax1.twinx() if PLOT_VELOCITY else None

            conc_plot = conc / group_max if NORMALIZE_CONC and group_max > 0 else conc
            ax1.plot(
                x_arc,
                conc_plot,
                label=case_label,
                alpha=1.0,
                color=color,
                linewidth=3,
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
                    print(f"  [WARN] velocity columns missing for {case_label}, skipping velocity plot")

            ax1.set_xlabel("Centerline length [cm]")
            label_y = "Concentration" + (" (normalized)" if NORMALIZE_CONC else "")
            ax1.set_ylabel(label_y)
            if PLOT_VELOCITY and ax2 is not None:
                ax2.set_ylabel("Velocity magnitude")
            ax1.set_ylim(bottom=0)
            if NORMALIZE_CONC:
                ax1.set_ylim(top=1)
            ax1.set_xlim(left=0)

            if PHASE2_LEGEND:
                ax1.legend(loc="best")

            if PLOT_TITLES:
                ax1.set_title(case_label)

            if PHASE2_CLEAN:
                apply_phase2_clean_mode(ax1, ax2)

            fig.tight_layout()
            scale_subdir = "normalized" if NORMALIZE_CONC else "non-normalized"
            static_output_dir = output_dir_for_case_spec(
                case_spec,
                PHASE2_STATIC_SUBDIR,
                scale_subdir,
            )
            os.makedirs(static_output_dir, exist_ok=True)
            out_name = f"{tag_plot_stem(dye)}.svg"
            plt.savefig(
                os.path.join(static_output_dir, out_name),
                format="svg",
                transparent=True,
                dpi=600,
            )
            if phase2_manual_snapshot_enabled():
                manual_snapshot_dir = output_dir_for_case_spec(
                    case_spec,
                    PHASE2_MANUAL_SNAPSHOT_SUBDIR,
                )
                os.makedirs(manual_snapshot_dir, exist_ok=True)
                manual_stem = f"{tag_plot_stem(dye)}_t{format_time_token(PHASE1_MANUAL_TIME)}"
                manual_out_paths = build_output_paths(
                    manual_snapshot_dir, manual_stem, PHASE2_OUTPUT_FORMATS
                )
                if OVERWRITE_EXISTING or not outputs_exist(manual_out_paths):
                    save_figure_in_formats(
                        fig,
                        manual_snapshot_dir,
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
    subdir = "normalized" if NORMALIZED_TAG else "non-normalized"
    out_dir = output_dir_for_phase3_geom_dye(
        RUN_GEOMETRIES[0],
        DYES[0],
        PHASE3_SUBDIR,
        subdir,
    )
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
        series_out_dir = output_dir_for_phase3_geom_dye(geom, dye, PHASE3_SUBDIR, subdir)
        os.makedirs(series_out_dir, exist_ok=True)
        fname = os.path.join(series_out_dir, f"TAGvHMR{('_d' + dye) if dye else ''}.svg")
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
