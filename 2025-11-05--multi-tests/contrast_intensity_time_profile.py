#!/usr/bin/env python3
"""
Count the number of cells with concentration above a threshold in result_###.vtu
files and create an intensity-vs-time profile.

Edit the SETTINGS block below. No command-line parsing is used.
"""

import os
import re
import sys
from pathlib import Path

# ============================================================
# SETTINGS — EDIT THESE
# ============================================================

RESULTS_DIR = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g80/96-procs"

ARRAY_NAME = "Concentration"   # Name of the concentration array
THRESHOLD = 5.0                # Count cells with concentration > THRESHOLD
DT = 0.01                      # Seconds per result index, e.g. result_346 -> 3.46 s

T_START = 0                  # Start time in seconds
T_END = 6.1                   # End time in seconds; use None for last available file

NORMALIZE_Y = False            # If True, plot y-axis as 0 to 1
STRICT_TIME_WINDOW = False     # If True, only include files with exact implied time in [T_START, T_END]

# Leave as None to auto-name outputs in RESULTS_DIR
OUT_PREFIX = None

# Path to pvpython for auto-relaunch if vtk is unavailable in regular python
PVPYTHON_PATH = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"

# ============================================================
# END SETTINGS
# ============================================================


def maybe_relaunch_under_pvpython(pvpython_path: str):
    exe = sys.executable.lower()
    if "pvpython" in exe:
        return
    try:
        import vtk  # noqa: F401
        return
    except Exception:
        pass

    if os.path.exists(pvpython_path):
        print(f"[INFO] Relaunching under pvpython: {pvpython_path}")
        os.execv(pvpython_path, [pvpython_path] + sys.argv)
    else:
        raise RuntimeError(
            "This script needs VTK/ParaView Python. "
            f"Could not import vtk and pvpython was not found at:\n{pvpython_path}"
        )


maybe_relaunch_under_pvpython(PVPYTHON_PATH)

import math
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

RESULT_RE = re.compile(r"result_(\d+)\.vtu$")


def find_result_files(results_dir: Path):
    files = []
    for path in results_dir.glob("result_*.vtu"):
        m = RESULT_RE.match(path.name)
        if m:
            idx = int(m.group(1))
            files.append((idx, path))
    files.sort(key=lambda x: x[0])
    return files


def read_array_as_cell_array(vtu_path: Path, array_name: str):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    data = reader.GetOutput()

    cell_data = data.GetCellData()
    arr = cell_data.GetArray(array_name)
    if arr is not None:
        return vtk_to_numpy(arr), "cell"

    point_data = data.GetPointData()
    parr = point_data.GetArray(array_name)
    if parr is not None:
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(data)
        p2c.PassPointDataOff()
        p2c.Update()

        converted = p2c.GetOutput()
        carr = converted.GetCellData().GetArray(array_name)
        if carr is None:
            raise RuntimeError(
                f"Array '{array_name}' found in point data of {vtu_path.name}, "
                "but conversion to cell data failed."
            )
        return vtk_to_numpy(carr), "point->cell"

    raise RuntimeError(
        f"Array '{array_name}' not found in either cell data or point data "
        f"for file: {vtu_path}"
    )


def build_output_prefix(results_dir: Path, array_name: str, threshold: float, out_prefix):
    if out_prefix:
        return Path(out_prefix)
    safe_thr = str(threshold).replace(".", "p")
    return results_dir / f"cell_intensity_{array_name}_thr_{safe_thr}"


def main():
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found:\n{results_dir}")

    result_files = find_result_files(results_dir)
    if not result_files:
        raise FileNotFoundError(f"No result_###.vtu files found in:\n{results_dir}")

    all_indices = np.array([idx for idx, _ in result_files], dtype=int)
    all_times = all_indices * DT

    t_start = float(T_START)
    t_end = float(all_times[-1]) if T_END is None else float(T_END)

    if t_end < t_start:
        raise ValueError(f"T_END ({t_end}) must be >= T_START ({t_start})")

    if STRICT_TIME_WINDOW:
        selected = [
            (idx, path) for idx, path in result_files
            if (idx * DT) >= t_start and (idx * DT) <= t_end
        ]
    else:
        start_idx = int(math.ceil(t_start / DT))
        end_idx = int(math.floor(t_end / DT))
        selected = [
            (idx, path) for idx, path in result_files
            if idx >= start_idx and idx <= end_idx
        ]

    if not selected:
        raise RuntimeError(
            "No files selected in the requested time range.\n"
            f"Requested: [{t_start}, {t_end}] s\n"
            f"Available file times span roughly: [{all_times[0]}, {all_times[-1]}] s"
        )

    print(f"[INFO] Results dir   : {results_dir}")
    print(f"[INFO] Array name    : {ARRAY_NAME}")
    print(f"[INFO] Threshold     : {THRESHOLD}")
    print(f"[INFO] dt            : {DT}")
    print(f"[INFO] Time window   : [{t_start}, {t_end}] s")
    print(f"[INFO] Files selected: {len(selected)}")
    print(f"[INFO] First/last    : result_{selected[0][0]}.vtu to result_{selected[-1][0]}.vtu")

    times = []
    counts = []
    data_origin_used = None

    for idx, path in selected:
        values, origin = read_array_as_cell_array(path, ARRAY_NAME)
        if data_origin_used is None:
            data_origin_used = origin

        count_above = int(np.count_nonzero(values > THRESHOLD))
        t = idx * DT

        times.append(t)
        counts.append(count_above)

        print(f"[OK] {path.name}: time={t:.4f} s, cells_above_{THRESHOLD}={count_above}")

    times = np.asarray(times, dtype=float)
    counts = np.asarray(counts, dtype=float)

    max_count = float(np.max(counts)) if len(counts) > 0 else 0.0
    counts_norm = counts / max_count if max_count > 0 else np.zeros_like(counts)

    y = counts_norm if NORMALIZE_Y else counts

    out_prefix = build_output_prefix(results_dir, ARRAY_NAME, THRESHOLD, OUT_PREFIX)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")

    with open(csv_path, "w") as f:
        f.write("time_s,file_index,cells_above_threshold,normalized_cells_above_threshold,threshold,array_name,data_origin\n")
        for t, (idx, _), c, cn in zip(times, selected, counts, counts_norm):
            f.write(f"{t:.8f},{idx},{int(c)},{cn:.8f},{THRESHOLD},{ARRAY_NAME},{data_origin_used}\n")

    plt.figure(figsize=(8, 4.5))
    plt.plot(times, y, marker="o")
    plt.xlabel("Time [s]")
    if NORMALIZE_Y:
        plt.ylabel("Normalized intensity")
        plt.ylim(0.0, 1.0)
        title_y = "normalized cell count"
    else:
        plt.ylabel("Cells above threshold")
        title_y = "cell count"

    plt.title(f"Intensity profile ({title_y})\n{ARRAY_NAME} > {THRESHOLD}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("\n[DONE]")
    print(f"CSV written to: {csv_path}")
    print(f"PNG written to: {png_path}")
    print(f"Data origin used: {data_origin_used}")


if __name__ == "__main__":
    main()
