# export_centerline_conc_batch_verbose.py
import argparse
import csv
import json
import os
import sys
from glob import glob
from pathlib import Path

PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-time",
        type=float,
        default=None,
        help="Physical time to use instead of auto-selecting the max inlet-concentration time.",
    )
    parser.add_argument(
        "--manual-step",
        type=int,
        default=None,
        help="Result step to use directly instead of looking it up from time or t_max.",
    )
    parser.add_argument(
        "--save-all-timesteps",
        action="store_true",
        help="Save a full time-series concentration CSV instead of a single snapshot.",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("T_START", "T_END"),
        default=None,
        help="When saving a time series, include only timesteps whose physical times fall in this range.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSVs instead of skipping them.",
    )
    parser.add_argument(
        "--case-specs-json",
        type=str,
        default=None,
        help="JSON list of case-spec dictionaries to process instead of the built-in CASE_SPECS.",
    )
    return parser.parse_known_args(sys.argv[1:])[0]


CLI_ARGS = parse_cli_args()

# Make sure we're under pvpython *before* importing paraview.simple
if "paraview" not in sys.executable.lower():
    print(f"[INFO] Relaunching under pvpython: {PV_PYTHON}")
    cmd = [PV_PYTHON] + sys.argv
    os.execv(PV_PYTHON, cmd)

from paraview.simple import *

ROOT_DIR = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance"
CENTERLINE_CANDIDATES = [
    os.path.join(ROOT_DIR, "centerline_LCA.vtp"),
    (
        "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/"
        "2025-11-05--multi-tests/centerlines/centerline_LCA.vtp"
    ),
]
CONCENTRATION_DIR = (
    "/Users/tejjolly/Documents/BioSimm/Simulations/"
    "Post_Processing/2025-11-05--multi-tests/concentrations"
)

DEFAULT_CASE_SPECS = [
    {
        "geom": "80",
        "run_suffix": "",
        "dye": "",
        "case_dir": "g80",
        "child_folder": "96-procs",
        # "step_override": 514,
    },
]

# True -> save only one snapshot instead of the full time series
SAVE_ONLY_TMAX = True

# If set to an integer, use that result step directly instead of auto-picking
# the max inlet-concentration step from B_HF_Concentration_average.txt.
MANUAL_STEP = None

# If set to a float, use the nearest available result time instead of auto-picking
# the max inlet-concentration time from B_HF_Concentration_average.txt.
MANUAL_TIME = None


def load_case_specs():
    if not CLI_ARGS.case_specs_json:
        return DEFAULT_CASE_SPECS

    try:
        raw_specs = json.loads(CLI_ARGS.case_specs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse --case-specs-json: {exc}") from exc

    if not isinstance(raw_specs, list):
        raise ValueError("--case-specs-json must decode to a list of case specs")

    case_specs = []
    for idx, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Case spec #{idx} is not a dictionary: {raw_spec!r}")
        if "geom" not in raw_spec:
            raise ValueError(f"Case spec #{idx} is missing required key 'geom'")

        spec = dict(raw_spec)
        spec["geom"] = str(raw_spec["geom"])
        spec["run_suffix"] = str(raw_spec.get("run_suffix", ""))
        spec["dye"] = str(raw_spec.get("dye", ""))

        if raw_spec.get("case_dir") is not None:
            spec["case_dir"] = str(raw_spec["case_dir"])
        if raw_spec.get("child_folder") is not None:
            spec["child_folder"] = str(raw_spec["child_folder"])
        if raw_spec.get("step_override") is not None:
            spec["step_override"] = int(raw_spec["step_override"])
        if raw_spec.get("time_override") is not None:
            spec["time_override"] = float(raw_spec["time_override"])

        case_specs.append(spec)

    print(f"[INFO] Loaded {len(case_specs)} case spec(s) from --case-specs-json")
    return case_specs


CASE_SPECS = load_case_specs()


def resolve_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of: {paths}")


def build_case_base(geom, run_suffix=""):
    case = f"g{geom}"
    if run_suffix:
        case += f"_r{run_suffix}"
    return case


def build_case_name(geom, run_suffix="", dye=""):
    case = build_case_base(geom, run_suffix)
    if dye:
        case += f"_d{dye}"
    return case


def build_output_csv_path(case_name, save_only_tmax):
    suffix = "_concentration.csv" if save_only_tmax else "_concentration_timeseries.csv"
    return os.path.join(CONCENTRATION_DIR, f"{case_name}{suffix}")


def load_avgfile_rows(path):
    with open(path, "r") as f:
        lines = f.readlines()

    header_found = False
    inlet_index = None
    rows = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "step":
            header_found = True
            inlet_index = next(
                (i for i, part in enumerate(parts) if "inlet" in part.lower()),
                None,
            )
            if inlet_index is None:
                print(f"  [WARN] No inlet column found in header: {parts}")
                return []
            continue

        if not header_found:
            continue

        try:
            step_idx = int(parts[0])
            time_val = float(parts[1])
            inlet_c = float(parts[inlet_index])
        except Exception:
            continue

        rows.append({
            "step": step_idx,
            "time": time_val,
            "inlet_c": inlet_c,
        })

    return rows


def find_tmax_from_avgfile(path):
    """
    Parse B_HF_Concentration_average.txt and return:
    - the timestep index (1-based),
    - the physical time value,
    - the max inlet concentration value.
    """
    rows = load_avgfile_rows(path)
    if not rows:
        return None, None, None

    best_row = max(rows, key=lambda row: row["inlet_c"])
    best_step = best_row["step"]
    best_time = best_row["time"]
    max_val = best_row["inlet_c"]

    print(
        f"  [INFO] Max inlet concentration = {max_val} "
        f"at step={best_step}, time={best_time}"
    )
    return best_step, best_time, max_val


def find_step_for_time(path, target_time):
    """
    Parse B_HF_Concentration_average.txt and return the step whose physical time
    is closest to target_time.
    """
    rows = load_avgfile_rows(path)
    if not rows:
        return None, None, None

    best_row = min(rows, key=lambda row: abs(row["time"] - target_time))
    matched_step = best_row["step"]
    matched_time = best_row["time"]
    inlet_c = best_row["inlet_c"]

    print(
        f"  [INFO] Requested time={target_time}; using nearest available "
        f"time={matched_time} at step={matched_step} (|dt|={abs(matched_time - target_time)})"
    )
    return matched_step, matched_time, inlet_c


def find_steps_in_time_range(path, time_range):
    rows = load_avgfile_rows(path)
    if not rows:
        return []

    t0, t1 = time_range
    t_start, t_end = (t0, t1) if t0 <= t1 else (t1, t0)
    selected = [row for row in rows if t_start <= row["time"] <= t_end]

    print(
        f"  [INFO] Found {len(selected)} timestep(s) in requested time range "
        f"[{t_start}, {t_end}]"
    )
    return selected


def resolve_vtu_path(results_dir, step_idx):
    candidate_names = [
        f"result_{step_idx:04d}.vtu",
        f"result_{step_idx:03d}.vtu",
        f"result_{step_idx}.vtu",
    ]

    for name in candidate_names:
        test_path = os.path.join(results_dir, name)
        if os.path.exists(test_path):
            return test_path
    return None


def relabel_saved_timeseries_csv(csv_path, selected_rows):
    if not selected_rows or not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        return

    frame_col = None
    for candidate in ["TimeStep", "Time Step", "time_step", "Time", "time"]:
        if candidate in fieldnames:
            frame_col = candidate
            break
    if frame_col is None:
        print(f"[WARN] Could not find a frame column in {csv_path}; leaving saved times unchanged")
        return

    saved_frame_ids = []
    seen_ids = set()
    for row in rows:
        frame_id = row.get(frame_col, "")
        if frame_id not in seen_ids:
            seen_ids.add(frame_id)
            saved_frame_ids.append(frame_id)

    if len(saved_frame_ids) != len(selected_rows):
        print(
            f"[WARN] Saved frame count ({len(saved_frame_ids)}) does not match selected step count "
            f"({len(selected_rows)}) in {csv_path}; leaving saved times unchanged"
        )
        return

    step_map = {
        saved_id: int(row["step"])
        for saved_id, row in zip(saved_frame_ids, selected_rows)
    }
    time_map = {
        saved_id: float(row["time"])
        for saved_id, row in zip(saved_frame_ids, selected_rows)
    }

    timestep_col = next(
        (candidate for candidate in ["TimeStep", "Time Step", "time_step"] if candidate in fieldnames),
        "TimeStep",
    )
    time_col = next(
        (candidate for candidate in ["Time", "time"] if candidate in fieldnames),
        "Time",
    )

    if timestep_col not in fieldnames:
        fieldnames.insert(0, timestep_col)
    if time_col not in fieldnames:
        insert_at = 1 if timestep_col in fieldnames else 0
        fieldnames.insert(insert_at, time_col)

    for row in rows:
        frame_id = row.get(frame_col, "")
        row[timestep_col] = step_map[frame_id]
        row[time_col] = f"{time_map[frame_id]:.12g}"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Relabeled saved times in {csv_path} using physical times from the average file")


def load_saved_timeseries_metadata(csv_path):
    if not os.path.exists(csv_path):
        return [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        return [], []

    timestep_col = next(
        (candidate for candidate in ["TimeStep", "Time Step", "time_step"] if candidate in fieldnames),
        None,
    )
    time_col = next(
        (candidate for candidate in ["Time", "time"] if candidate in fieldnames),
        None,
    )

    saved_steps = []
    saved_times = []
    seen_frames = set()
    frame_key_col = timestep_col or time_col
    if frame_key_col is None:
        return [], []

    for row in rows:
        frame_key = row.get(frame_key_col, "")
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)

        if timestep_col is not None:
            try:
                saved_steps.append(int(float(row[timestep_col])))
            except Exception:
                saved_steps.append(None)
        if time_col is not None:
            try:
                saved_times.append(float(row[time_col]))
            except Exception:
                saved_times.append(None)

    return saved_steps, saved_times


def infer_timeseries_frame_columns(fieldnames):
    timestep_col = next(
        (candidate for candidate in ["TimeStep", "Time Step", "time_step"] if candidate in fieldnames),
        None,
    )
    time_col = next(
        (candidate for candidate in ["Time", "time"] if candidate in fieldnames),
        None,
    )
    return timestep_col, time_col


def missing_selected_rows_from_timeseries_csv(csv_path, selected_rows, atol=1e-9):
    if not selected_rows:
        return []

    saved_steps, saved_times = load_saved_timeseries_metadata(csv_path)
    if saved_steps and all(step is not None for step in saved_steps):
        saved_step_set = set(saved_steps)
        return [row for row in selected_rows if int(row["step"]) not in saved_step_set]

    if saved_times and all(time is not None for time in saved_times):
        missing_rows = []
        for row in selected_rows:
            target_time = float(row["time"])
            if not any(abs(saved_time - target_time) <= atol for saved_time in saved_times):
                missing_rows.append(row)
        return missing_rows

    return list(selected_rows)


def load_timeseries_csv_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def frame_sort_key(frame_key):
    kind, value = frame_key
    return (0, value) if kind == "step" else (1, value)


def group_rows_by_frame(rows, timestep_col, time_col):
    groups = {}
    for row in rows:
        if timestep_col is not None and row.get(timestep_col, "") != "":
            frame_key = ("step", int(float(row[timestep_col])))
        elif time_col is not None and row.get(time_col, "") != "":
            frame_key = ("time", float(row[time_col]))
        else:
            frame_key = ("row", len(groups))
        groups.setdefault(frame_key, []).append(row)
    return groups


def merge_timeseries_csv(existing_csv_path, new_csv_path):
    if not os.path.exists(existing_csv_path):
        os.replace(new_csv_path, existing_csv_path)
        return
    if not os.path.exists(new_csv_path):
        return

    existing_fieldnames, existing_rows = load_timeseries_csv_rows(existing_csv_path)
    new_fieldnames, new_rows = load_timeseries_csv_rows(new_csv_path)

    fieldnames = list(existing_fieldnames)
    for field in new_fieldnames:
        if field not in fieldnames:
            fieldnames.append(field)

    timestep_col, time_col = infer_timeseries_frame_columns(fieldnames)
    merged_groups = group_rows_by_frame(existing_rows, timestep_col, time_col)
    new_groups = group_rows_by_frame(new_rows, timestep_col, time_col)

    for frame_key, rows in new_groups.items():
        merged_groups[frame_key] = rows

    merged_frame_keys = sorted(merged_groups.keys(), key=frame_sort_key)
    merged_rows = []
    for frame_key in merged_frame_keys:
        merged_rows.extend(merged_groups[frame_key])

    with open(existing_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


centerline_path = resolve_existing_path(CENTERLINE_CANDIDATES)
centerline = OpenDataFile(centerline_path)
os.makedirs(CONCENTRATION_DIR, exist_ok=True)

for spec in CASE_SPECS:
    geom = spec["geom"]
    run_suffix = spec.get("run_suffix", "")
    dye = spec.get("dye", "")
    child_folder = spec.get("child_folder", "96-procs")

    case_name = build_case_name(geom, run_suffix, dye)
    case_dir = os.path.join(ROOT_DIR, spec.get("case_dir", case_name))
    results_dir = os.path.join(case_dir, child_folder)

    print(f"[INFO] Processing {case_name} from {results_dir} ...")

    vtu_files = sorted(glob(os.path.join(results_dir, "result_*.vtu")))
    if not vtu_files:
        print(f"[WARN] No result_*.vtu in {results_dir}, skipping")
        continue

    save_only_tmax = SAVE_ONLY_TMAX and not CLI_ARGS.save_all_timesteps
    out_csv = build_output_csv_path(case_name, save_only_tmax)
    time_range = tuple(CLI_ARGS.time_range) if CLI_ARGS.time_range is not None else None
    selected_rows_for_range = None
    missing_rows_for_range = None

    if not save_only_tmax and time_range is not None:
        avgfile = os.path.join(results_dir, "B_HF_Concentration_average.txt")
        if not os.path.exists(avgfile):
            print(f"[WARN] {avgfile} not found, skipping {case_name}")
            continue
        selected_rows_for_range = find_steps_in_time_range(avgfile, time_range)
        if not selected_rows_for_range:
            print(f"[WARN] No steps found in time range {time_range} for {case_name}, skipping")
            continue

    if os.path.exists(out_csv) and not CLI_ARGS.overwrite:
        if (
            not save_only_tmax and
            selected_rows_for_range is not None
        ):
            missing_rows_for_range = missing_selected_rows_from_timeseries_csv(out_csv, selected_rows_for_range)
            if not missing_rows_for_range:
                print(f"[SKIP] Existing time-series CSV already covers requested range: {out_csv}")
                continue
            print(
                f"[INFO] Existing time-series CSV is missing {len(missing_rows_for_range)} timestep(s); "
                f"exporting only the missing data into {out_csv}"
            )
        else:
            print(f"[SKIP] Output already exists: {out_csv}")
            continue

    if save_only_tmax:
        step_override = spec.get("step_override")
        if step_override is None:
            step_override = CLI_ARGS.manual_step if CLI_ARGS.manual_step is not None else MANUAL_STEP

        time_override = spec.get("time_override")
        if time_override is None:
            time_override = CLI_ARGS.manual_time if CLI_ARGS.manual_time is not None else MANUAL_TIME

        if step_override is not None:
            step_idx = int(step_override)
            print(f"  [INFO] Using manual step override: {step_idx}")
        else:
            avgfile = os.path.join(results_dir, "B_HF_Concentration_average.txt")
            if not os.path.exists(avgfile):
                print(f"[WARN] {avgfile} not found, skipping {case_name}")
                continue

            if time_override is not None:
                step_idx, matched_time, matched_val = find_step_for_time(avgfile, float(time_override))
                if step_idx is None:
                    print(f"[WARN] Could not find a step near time={time_override} for {case_name}, skipping")
                    continue
            else:
                step_idx, t_max, max_val = find_tmax_from_avgfile(avgfile)
                if step_idx is None:
                    print(f"[WARN] Could not find t_max for {case_name}, skipping")
                    continue

            if step_idx is None:
                print(f"[WARN] Could not resolve a result step for {case_name}, skipping")
                continue

        vtu_path = resolve_vtu_path(results_dir, step_idx)
        if vtu_path is None:
            print(f"[WARN] Could not find VTU for step {step_idx} in {results_dir}")
            continue

        print(f"  [INFO] Using VTU file: {os.path.basename(vtu_path)} for step {step_idx}")

        volume = OpenDataFile(vtu_path)
        resampled = ResampleWithDataset(
            SourceDataArrays=volume,
            DestinationMesh=centerline,
        )
        resampled.UpdatePipeline()

        print(f"[INFO] Saving ONLY t_max to {out_csv}")
        SaveData(
            out_csv,
            proxy=resampled,
            WriteTimeSteps=0,
            ChooseArraysToWrite=1,
            AddTime=0,
            AddTimeStep=0,
            AddMetaData=1,
            PointDataArrays=[
                "Concentration",
                "Pressure",
                "Velocity",
            ],
        )
    else:
        selected_vtu_files = vtu_files
        selected_rows = None
        save_target_csv = out_csv
        merge_after_save = False
        if time_range is not None:
            selected_rows = selected_rows_for_range

            if missing_rows_for_range is not None:
                selected_rows = missing_rows_for_range
                if out_csv.endswith(".csv"):
                    save_target_csv = out_csv[:-4] + "_missing.csv"
                else:
                    save_target_csv = f"{out_csv}_missing.csv"
                merge_after_save = True

            selected_vtu_files = []
            for row in selected_rows:
                vtu_path = resolve_vtu_path(results_dir, int(row["step"]))
                if vtu_path is None:
                    print(f"[WARN] Could not find VTU for step {row['step']} in {results_dir}")
                    continue
                selected_vtu_files.append(vtu_path)

            if not selected_vtu_files:
                print(f"[WARN] No VTU files resolved in time range {time_range} for {case_name}, skipping")
                continue

            print(
                f"  [INFO] Loading {len(selected_vtu_files)} timestep(s) for {case_name} "
                f"in time range [{min(time_range)}, {max(time_range)}]"
            )
        else:
            print(f"  [INFO] Loading full time series for {case_name}")

        volume = OpenDataFile(selected_vtu_files)

        anim = GetAnimationScene()
        anim.UpdateAnimationUsingDataTimeSteps()

        resampled = ResampleWithDataset(
            SourceDataArrays=volume,
            DestinationMesh=centerline,
        )
        resampled.UpdatePipeline()

        print(f"[INFO] Saving ALL timesteps to {save_target_csv}")
        SaveData(
            save_target_csv,
            proxy=resampled,
            WriteTimeSteps=1,
            ChooseArraysToWrite=1,
            AddTime=1,
            AddTimeStep=1,
            AddMetaData=1,
            PointDataArrays=[
                "Concentration",
                "Pressure",
                "Velocity",
            ],
        )
        if selected_rows is not None:
            relabel_saved_timeseries_csv(save_target_csv, selected_rows)
            if merge_after_save:
                merge_timeseries_csv(out_csv, save_target_csv)
                os.remove(save_target_csv)

    print(f"[OK] {case_name} done")

print("[DONE] all configured cases")
