# export_centerline_conc_batch_verbose.py
import os
from glob import glob
from paraview.simple import *
import os, sys, subprocess

PV_PYTHON = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"

if "paraview" not in sys.executable.lower():
    print(f"[INFO] Relaunching under pvpython: {PV_PYTHON}")
    cmd = [PV_PYTHON] + sys.argv
    os.execv(PV_PYTHON, cmd) 

root_dir = "/Volumes/biosimm-Tej-Jolly/2025-11-10--TAG_01"
centerline_path = os.path.join(root_dir, "path_LCA_1.vtp")

run_geometries = ["13", "37"]                
run_suffixes   = ["24", "62", "81", "100"]                

child_folder = "96-procs"

# load centerline once
centerline = OpenDataFile(centerline_path)

for geom in run_geometries:
    for run_suffix in run_suffixes:

        case_name = f"g{geom}_r{run_suffix}"
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

        out_csv = os.path.join(case_dir, "concentration.csv")
        print(f"[INFO] Saving to {out_csv}")

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

print("[DONE] all geometries and suffixes")