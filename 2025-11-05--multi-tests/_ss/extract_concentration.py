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
from paraview.servermanager import Fetch

root_dir = "/Volumes/biosimm-Tej-Jolly/2025-11-16--TAG"
centerline_path = os.path.join(root_dir, "g13_centerline--stenosis.vtp")

run_geometries = ["37"]
run_suffixes   = ["100", "24", "43", "62"]
dye_suffixes   = ["N","B"]

child_folder = "96-procs"

# ---- NEW: flag to control behavior ----
SAVE_ONLY_TMAX = True   # True → only save timestep with max inlet concentration
# ---------------------------------------

def find_tmax_from_avgfile(path):
    """
    Parse B_HF_Concentration_average.txt and return:
    - the timestep index (1-based),
    - the physical time value,
    - the max concentration value.
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
            # Find which column index corresponds to inlet_3.vtp
            inlet_index = parts.index("inlet_3.vtp")
            continue

        # Skip lines until header is found
        if not header_found:
            continue

        # Parse numerical rows: step, time, col1, col2, ...
        try:
            step_idx = int(parts[0])
            time_val = float(parts[1])
            inlet_c = float(parts[inlet_index])
        except:
            continue

        if inlet_c > max_val:
            max_val = inlet_c
            best_step = step_idx
            best_time = time_val

    print(f"[INFO] Max inlet concentration = {max_val} at step={best_step}, time={best_time}")
    return best_step, best_time, max_val


def get_time_values(proxy):
    """Try to get timestep values associated with a proxy or the global TimeKeeper."""
    tv = getattr(proxy, "TimestepValues", None)
    if tv:
        return list(tv)

    anim = GetAnimationScene()
    tk = anim.TimeKeeper
    tv = getattr(tk, "TimestepValues", None)
    if tv:
        return list(tv)

    return []


def find_tmax_external(case_dir):
    """
    Read B_HF_Concentration_average.txt to find the timestep
    with maximum inlet_3.vtp concentration.
    """
    avgfile = os.path.join(case_dir, child_folder, "B_HF_Concentration_average.txt")
    if not os.path.exists(avgfile):
        print(f"[WARN] No B_HF_Concentration_average.txt in {case_dir}")
        return None

    return find_tmax_from_avgfile(avgfile)


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

            # open time series
            volume = OpenDataFile(vtu_files)

            # tell PV about time
            anim = GetAnimationScene()
            anim.UpdateAnimationUsingDataTimeSteps()

            # resample along centerline
            resampled = ResampleWithDataset(
                SourceDataArrays=volume,
                DestinationMesh=centerline
            )
            resampled.UpdatePipeline()

            concentration_dir = (
                "/Users/tejjolly/Documents/BioSimm/Simulations/"
                "Post_Processing/2025-11-05--multi-tests/concentrations/"
            )
            os.makedirs(concentration_dir, exist_ok=True)

            out_csv = os.path.join(concentration_dir, f"{case_name}_concentration.csv")

            if SAVE_ONLY_TMAX:
                step_idx, t_max, max_val = find_tmax_external(case_dir)

                if step_idx is not None:
                    times = get_time_values(volume)
                    if not times:
                        print(f"[WARN] No timestep values found for {case_name}, cannot map step {step_idx}")
                        continue

                    # step_idx from file is 1-based; Python list is 0-based
                    ts_index = step_idx - 1

                    if ts_index < 0 or ts_index >= len(times):
                        print(f"[WARN] step_idx {step_idx} out of range for {case_name} (len(times)={len(times)})")
                        continue

                    target_time = times[ts_index]
                    print(f"[INFO] Mapping step {step_idx} → ParaView time {target_time} (file should be ~result_{step_idx:03d}.vtu)")
                    print(f"[INFO] B_HF file time = {t_max}, max inlet conc = {max_val}")

                    anim.AnimationTime = target_time
                    resampled.UpdatePipeline(time=target_time)

                    SaveData(
                        out_csv,
                        proxy=resampled,
                        WriteTimeSteps=0,
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
                else:
                    print(f"[WARN] No t_max found; skipping save for {case_name}")

            else:
                # --- Original behavior: save ALL timesteps ---
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
