# export_centerline_conc_batch_verbose.py
import os
import sys
from glob import glob

PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"

# Make sure we're under pvpython *before* importing paraview.simple
if "paraview" not in sys.executable.lower():
    print(f"[INFO] Relaunching under pvpython: {PV_PYTHON}")
    cmd = [PV_PYTHON] + sys.argv
    os.execv(PV_PYTHON, cmd)

from paraview.simple import *

root_dir = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance"
centerline_path = os.path.join(root_dir, "centerline_LCA.vtp")

run_geometries = ["80"]
run_suffixes   = [""]
dye_suffixes   = ["",""]

child_folder = "96-procs"

# flag: True → use only the timestep with max inlet conc
SAVE_ONLY_TMAX = True


def find_tmax_from_avgfile(path):
    """
    Parse B_HF_Concentration_average.txt and return:
    - the timestep index (1-based),
    - the physical time value,
    - the max inlet concentration value.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    header_found = False
    inlet_index = None
    max_val = float("-inf")
    best_step = None
    best_time = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        # Detect header (the line that begins with "step")
        if parts[0] == "step":
            header_found = True

            # Find first column whose name CONTAINS "inlet"
            inlet_index = next(
                (i for i, p in enumerate(parts) if "inlet" in p.lower()),
                None
            )

            if inlet_index is None:
                print(f"  [WARN] No inlet column found in header: {parts}")
                return None, None, None

            continue


        # Skip lines until header is found
        if not header_found:
            continue

        # Parse numerical rows: step, time, col1, col2, ...
        try:
            step_idx = int(parts[0])
            time_val = float(parts[1])
            inlet_c = float(parts[inlet_index])
        except Exception:
            continue

        if inlet_c > max_val:
            max_val = inlet_c
            best_step = step_idx
            best_time = time_val

    print(f"  [INFO] Max inlet concentration = {max_val} at step={best_step}, time={best_time}")
    return best_step, best_time, max_val


# load centerline once
centerline = OpenDataFile(centerline_path)

for geom in run_geometries:
    for run_suffix in run_suffixes:
        for dye in dye_suffixes:

            case_name = f"g{geom}_r{run_suffix}_d{dye}"
            case_dir = os.path.join(root_dir, case_name)
            results_dir = os.path.join(case_dir, child_folder)

            print(f"[INFO] Processing {case_name} …")

            vtu_files = sorted(glob(os.path.join(results_dir, "result_*.vtu")))
            if not vtu_files:
                print(f"[WARN] No result_*.vtu in {results_dir}, skipping")
                continue

            concentration_dir = (
                "/Users/tejjolly/Documents/BioSimm/Simulations/"
                "Post_Processing/2025-11-05--multi-tests/concentrations/"
            )
            os.makedirs(concentration_dir, exist_ok=True)

            out_csv = os.path.join(concentration_dir, f"{case_name}_concentration.csv")

            if SAVE_ONLY_TMAX:
                # --- use B_HF_Concentration_average.txt to pick the step ---
                avgfile = os.path.join(results_dir, "B_HF_Concentration_average.txt")
                if not os.path.exists(avgfile):
                    print(f"[WARN] {avgfile} not found, skipping {case_name}")
                    continue

                step_idx, t_max, max_val = find_tmax_from_avgfile(avgfile)
                if step_idx is None:
                    print(f"[WARN] Could not find t_max for {case_name}, skipping")
                    continue

                # expected naming: result_###.vtu (with zero padding to 3 digits)
                # e.g., step 117 → result_117.vtu; step 3 → result_003.vtu
                candidate_names = [
                    f"result_{step_idx:03d}.vtu",  # zero-padded
                    f"result_{step_idx}.vtu",      # plain
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

                # open only this single VTU (no time series)
                volume = OpenDataFile(vtu_path)

                # resample along centerline
                resampled = ResampleWithDataset(
                    SourceDataArrays=volume,
                    DestinationMesh=centerline
                )
                resampled.UpdatePipeline()

                print(f"[INFO] Saving ONLY t_max to {out_csv}")
                SaveData(
                    out_csv,
                    proxy=resampled,
                    WriteTimeSteps=0,  # single static dataset
                    ChooseArraysToWrite=1,
                    AddTime=0,         # no time column needed if you don't want it
                    AddTimeStep=0,
                    AddMetaData=1,
                    PointDataArrays=[
                        "Concentration",
                        "Pressure",
                        "Velocity",
                    ],
                )

            else:
                # --- Original behavior: load all VTUs as time series and save everything ---
                print(f"  [INFO] Loading full time series for {case_name}")
                volume = OpenDataFile(vtu_files)

                anim = GetAnimationScene()
                anim.UpdateAnimationUsingDataTimeSteps()

                resampled = ResampleWithDataset(
                    SourceDataArrays=volume,
                    DestinationMesh=centerline
                )
                resampled.UpdatePipeline()

                print(f"[INFO] Saving ALL timesteps to {out_csv}")
                SaveData(
                    out_csv,
                    proxy=resampled,
                    WriteTimeSteps=1,  # full time series
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

print("[DONE] all geometries, suffixes, and dyes")