# export_centerline_conc_batch_verbose.py
import argparse
import json
import os
import sys
from glob import glob

from case_utils import build_case_name, concentration_csv_path, resolve_existing_path
from timeseries_csv_utils import (
    append_timeseries_csv,
    merge_timeseries_csv,
    missing_selected_rows_from_timeseries_csv,
    relabel_saved_timeseries_csv,
)

PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


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
        required=True,
        help="JSON list of case-spec dictionaries to process.",
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
    os.path.join(PROJECT_DIR, "centerlines", "centerline_LCA.vtp"),
]


def load_case_specs():
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
        if raw_spec.get("results_dir") is not None:
            spec["results_dir"] = str(raw_spec["results_dir"])
        if raw_spec.get("tag_dir") is not None:
            spec["tag_dir"] = str(raw_spec["tag_dir"])
        if raw_spec.get("concentration_dir") is not None:
            spec["concentration_dir"] = str(raw_spec["concentration_dir"])
        if raw_spec.get("step_override") is not None:
            spec["step_override"] = int(raw_spec["step_override"])
        if raw_spec.get("time_override") is not None:
            spec["time_override"] = float(raw_spec["time_override"])

        case_specs.append(spec)

    print(f"[INFO] Loaded {len(case_specs)} case spec(s) from --case-specs-json")
    return case_specs


CASE_SPECS = load_case_specs()


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


def delete_paraview_proxy(proxy):
    try:
        Delete(proxy)
    except Exception:
        pass


def save_resampled_timeseries_csv(csv_path, volume):
    resampled = ResampleWithDataset(
        SourceDataArrays=volume,
        DestinationMesh=centerline,
    )
    resampled.UpdatePipeline()

    SaveData(
        csv_path,
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
    delete_paraview_proxy(resampled)


def iter_chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield start, items[start:start + chunk_size]


def make_unique_sidecar_path(path, suffix):
    candidate = f"{path}{suffix}"
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{path}{suffix}.{counter:03d}"
        counter += 1
    return candidate


def save_selected_timesteps_with_progress(work_items, save_target_csv, chunk_size=100):
    total = len(work_items)

    chunk_count = (total + chunk_size - 1) // chunk_size
    for chunk_idx, (start_idx, chunk) in enumerate(iter_chunks(work_items, chunk_size), start=1):
        rows = [row for row, _ in chunk]
        vtu_paths = [vtu_path for _, vtu_path in chunk]
        first_row = rows[0]
        last_row = rows[-1]
        first_step = int(first_row["step"])
        last_step = int(last_row["step"])
        first_time = float(first_row["time"])
        last_time = float(last_row["time"])

        if save_target_csv.endswith(".csv"):
            part_csv = save_target_csv[:-4] + f"_part_{chunk_idx:04d}.csv"
        else:
            part_csv = f"{save_target_csv}_part_{chunk_idx:04d}.csv"

        if os.path.exists(part_csv):
            os.remove(part_csv)

        print(
            f"[PROGRESS] Saving chunk {chunk_idx}/{chunk_count} "
            f"(timesteps {start_idx + 1}-{start_idx + len(chunk)}/{total}): "
            f"steps={first_step}-{last_step}, times={first_time:.12g}-{last_time:.12g}",
            flush=True,
        )

        volume = OpenDataFile(vtu_paths)
        save_resampled_timeseries_csv(part_csv, volume)
        delete_paraview_proxy(volume)

        relabel_saved_timeseries_csv(part_csv, rows)
        append_timeseries_csv(save_target_csv, part_csv)
        if os.path.exists(part_csv):
            os.remove(part_csv)

    print(f"[INFO] Finished saving {total}/{total} timestep(s) to {save_target_csv}", flush=True)


centerline_path = resolve_existing_path(CENTERLINE_CANDIDATES)
centerline = OpenDataFile(centerline_path)

for spec in CASE_SPECS:
    geom = spec["geom"]
    run_suffix = spec.get("run_suffix", "")
    dye = spec.get("dye", "")
    child_folder = spec.get("child_folder", "96-procs")

    case_name = build_case_name(geom, run_suffix, dye)
    case_dir = spec.get("case_dir", case_name)
    if not os.path.isabs(case_dir):
        case_dir = os.path.join(ROOT_DIR, case_dir)
    results_dir = spec.get("results_dir", os.path.join(case_dir, child_folder))
    concentration_dir = spec.get(
        "concentration_dir",
        os.path.join(case_dir, "TAG", "concentrations"),
    )
    os.makedirs(concentration_dir, exist_ok=True)

    print(f"[INFO] Processing {case_name} from {results_dir} ...")

    vtu_files = sorted(glob(os.path.join(results_dir, "result_*.vtu")))
    if not vtu_files:
        print(f"[WARN] No result_*.vtu in {results_dir}, skipping")
        continue

    save_only_tmax = not CLI_ARGS.save_all_timesteps
    out_csv = concentration_csv_path(
        concentration_dir,
        geom,
        run_suffix,
        dye,
        full_series=not save_only_tmax,
    )
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
            step_override = CLI_ARGS.manual_step

        time_override = spec.get("time_override")
        if time_override is None:
            time_override = CLI_ARGS.manual_time

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
        selected_work_items = None
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
                if os.path.exists(save_target_csv):
                    remaining_rows = missing_selected_rows_from_timeseries_csv(save_target_csv, selected_rows)
                    recovered_count = len(selected_rows) - len(remaining_rows)
                    if recovered_count > 0:
                        print(
                            f"[INFO] Resuming from existing partial CSV: recovered "
                            f"{recovered_count} timestep(s) from {save_target_csv}"
                        )
                        selected_rows = remaining_rows
                    else:
                        archive_path = make_unique_sidecar_path(save_target_csv, ".unusable")
                        os.replace(save_target_csv, archive_path)
                        print(
                            f"[WARN] Existing partial CSV had no reusable timestep metadata; "
                            f"moved it to {archive_path}"
                        )

                    if not selected_rows:
                        print(
                            f"[INFO] Existing partial CSV already contains all missing timestep(s); "
                            f"merging into {out_csv}"
                        )
                        merge_timeseries_csv(out_csv, save_target_csv)
                        os.remove(save_target_csv)
                        continue

            selected_vtu_files = []
            selected_work_items = []
            for row in selected_rows:
                vtu_path = resolve_vtu_path(results_dir, int(row["step"]))
                if vtu_path is None:
                    print(f"[WARN] Could not find VTU for step {row['step']} in {results_dir}")
                    continue
                selected_vtu_files.append(vtu_path)
                selected_work_items.append((row, vtu_path))

            if not selected_vtu_files:
                print(f"[WARN] No VTU files resolved in time range {time_range} for {case_name}, skipping")
                continue

            print(
                f"  [INFO] Loading {len(selected_vtu_files)} timestep(s) for {case_name} "
                f"in time range [{min(time_range)}, {max(time_range)}]"
            )
        else:
            print(f"  [INFO] Loading full time series for {case_name}")

        if selected_work_items is not None:
            print(f"[INFO] Saving selected timesteps to {save_target_csv}")
            save_selected_timesteps_with_progress(selected_work_items, save_target_csv)
            if merge_after_save:
                merge_timeseries_csv(out_csv, save_target_csv)
                os.remove(save_target_csv)
        else:
            volume = OpenDataFile(selected_vtu_files)

            anim = GetAnimationScene()
            anim.UpdateAnimationUsingDataTimeSteps()

            print(f"[INFO] Saving ALL timesteps to {save_target_csv}")
            save_resampled_timeseries_csv(save_target_csv, volume)
            delete_paraview_proxy(volume)

    print(f"[OK] {case_name} done")

print("[DONE] all configured cases")
