# export_centerline_conc_batch_verbose.py
import argparse
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

CASE_SPECS = [
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

    out_csv = os.path.join(CONCENTRATION_DIR, f"{case_name}_concentration.csv")

    if SAVE_ONLY_TMAX:
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

        candidate_names = [
            f"result_{step_idx:04d}.vtu",
            f"result_{step_idx:03d}.vtu",
            f"result_{step_idx}.vtu",
        ]

        vtu_path = None
        for name in candidate_names:
            test_path = os.path.join(results_dir, name)
            if os.path.exists(test_path):
                vtu_path = test_path
                break

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
        print(f"  [INFO] Loading full time series for {case_name}")
        volume = OpenDataFile(vtu_files)

        anim = GetAnimationScene()
        anim.UpdateAnimationUsingDataTimeSteps()

        resampled = ResampleWithDataset(
            SourceDataArrays=volume,
            DestinationMesh=centerline,
        )
        resampled.UpdatePipeline()

        print(f"[INFO] Saving ALL timesteps to {out_csv}")
        SaveData(
            out_csv,
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

    print(f"[OK] {case_name} done")

print("[DONE] all configured cases")
