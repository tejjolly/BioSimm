#!/usr/bin/env python3
"""
Dump one PNG per timestep for a concentration animation from result_###.vtu files.

This version uses MANUAL_CAMERA only and does NOT try to load a .pvcvbc file.
Run with pvpython, or with regular python and let it relaunch itself under pvpython.
"""

import os
import re
import sys
from pathlib import Path

# ============================================================
# SETTINGS — EDIT THESE
# ============================================================

RESULTS_DIR = "/Volumes/maxone/2026-02-03--mass_balance/g87_r43/96-procs"
OUTPUT_FOLDER_NAME = "3d_dye_frames"

START_INDEX = 172
END_INDEX   = 2408

ARRAY_NAME = "Concentration"
COLOR_RANGE_MIN = 0.0
COLOR_RANGE_MAX = 150.0
COLOR_PRESET = "X Ray"
BELOW_RANGE_THRESHOLD = 5.0
BELOW_RANGE_COLOR = [1.0, 1.0, 1.0]

REPRESENTATION = "Volume"

# Camera / view
USE_MANUAL_CAMERA = True
MANUAL_CAMERA = {
    "CameraPosition": [9.933034174424655, -19.901768806769738, -7.284113708312885],
    "CameraFocalPoint": [4.183179915007794, -2.7663879227894252, -15.475958977824508],
    "CameraViewUp": [-0.5710608274257231, 0.19015248209849012, 0.7985809695526385],
    "CameraParallelScale": 5.136031835603023,
    "CameraParallelProjection": 1,
    "CameraViewAngle": 30.0,
}

# Rendering
VIEW_SIZE = [1600, 900]
ORIENTATION_AXES_VISIBLE = False
SHOW_SCALAR_BAR = False
BACKGROUND = [1.0, 1.0, 1.0]   # white

# Auto-relaunch under pvpython if needed
PVPYTHON_PATH = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"

# If True, overwrite existing PNGs. If False, skip frames already present.
OVERWRITE_EXISTING = False

# Optional filename prefix
FRAME_PREFIX = "concentration_"

# ============================================================
# END SETTINGS
# ============================================================


def maybe_relaunch_under_pvpython(pvpython_path: str):
    exe = sys.executable.lower()
    if "pvpython" in exe or "pvbatch" in exe:
        return
    try:
        import paraview  # noqa: F401
        return
    except Exception:
        pass

    if os.path.exists(pvpython_path):
        print(f"[INFO] Relaunching under pvpython: {pvpython_path}")
        os.execv(pvpython_path, [pvpython_path] + sys.argv)
    else:
        raise RuntimeError(
            "This script needs ParaView Python. "
            f"Could not import paraview and pvpython was not found at:\n{pvpython_path}"
        )


maybe_relaunch_under_pvpython(PVPYTHON_PATH)

from paraview.simple import *
import vtk

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


def get_array_association(vtu_path: Path, array_name: str):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    data = reader.GetOutput()

    if data.GetPointData().GetArray(array_name) is not None:
        return "POINTS"
    if data.GetCellData().GetArray(array_name) is not None:
        return "CELLS"

    point_names = [data.GetPointData().GetArrayName(i) for i in range(data.GetPointData().GetNumberOfArrays())]
    cell_names = [data.GetCellData().GetArrayName(i) for i in range(data.GetCellData().GetNumberOfArrays())]

    raise RuntimeError(
        f"Array '{array_name}' not found in {vtu_path.name}\n"
        f"Point arrays: {point_names}\n"
        f"Cell arrays: {cell_names}"
    )


def apply_manual_camera(view):
    view.CameraPosition = MANUAL_CAMERA["CameraPosition"]
    view.CameraFocalPoint = MANUAL_CAMERA["CameraFocalPoint"]
    view.CameraViewUp = MANUAL_CAMERA["CameraViewUp"]
    view.CameraParallelScale = MANUAL_CAMERA["CameraParallelScale"]
    view.CameraParallelProjection = MANUAL_CAMERA["CameraParallelProjection"]
    view.CameraViewAngle = MANUAL_CAMERA["CameraViewAngle"]


def configure_display(source, view, association: str):
    display = Show(source, view)

    try:
        display.Representation = REPRESENTATION
    except Exception:
        display.SetRepresentationType(REPRESENTATION)

    ColorBy(display, (association, ARRAY_NAME))

    lut = GetColorTransferFunction(ARRAY_NAME)
    lut.ApplyPreset(COLOR_PRESET, True)
    lut.RescaleTransferFunction(BELOW_RANGE_THRESHOLD, COLOR_RANGE_MAX)
    try:
        lut.InvertTransferFunction()
    except Exception as e:
        print(f"[WARN] Could not invert color transfer function: {e}")
    try:
        lut.UseBelowRangeColor = 1
        lut.BelowRangeColor = BELOW_RANGE_COLOR
    except Exception as e:
        print(f"[WARN] Could not configure below-range color: {e}")

    try:
        pwf = GetOpacityTransferFunction(ARRAY_NAME)
        pwf.RescaleTransferFunction(COLOR_RANGE_MIN, COLOR_RANGE_MAX)
        pwf.Points = [
            COLOR_RANGE_MIN, 1.0, 0.5, 0.0,
            COLOR_RANGE_MAX, 1.0, 0.5, 0.0,
        ]
    except Exception as e:
        print(f"[WARN] Could not configure opacity transfer function: {e}")

    try:
        display.SetScalarBarVisibility(view, SHOW_SCALAR_BAR)
    except Exception:
        pass

    try:
        HideUnusedScalarBars(view)
    except Exception:
        pass

    return display


def setup_view():
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = VIEW_SIZE
    view.Background = BACKGROUND
    view.OrientationAxesVisibility = 1 if ORIENTATION_AXES_VISIBLE else 0
    return view


def save_frame(view, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    SaveScreenshot(str(out_png), view, ImageResolution=VIEW_SIZE)


def build_output_dir(results_dir: Path):
    return results_dir.parent / OUTPUT_FOLDER_NAME


def main():
    results_dir = Path(RESULTS_DIR)
    output_dir = build_output_dir(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found:\n{results_dir}")

    all_files = find_result_files(results_dir)
    if not all_files:
        raise FileNotFoundError(f"No result_###.vtu files found in:\n{results_dir}")

    selected = [(idx, p) for idx, p in all_files if START_INDEX <= idx <= END_INDEX]
    if not selected:
        raise RuntimeError(
            f"No result files found in requested index range [{START_INDEX}, {END_INDEX}]"
        )

    assoc = get_array_association(selected[0][1], ARRAY_NAME)

    print(f"[INFO] Results dir     : {results_dir}")
    print(f"[INFO] Output dir      : {output_dir}")
    print(f"[INFO] Index range     : [{START_INDEX}, {END_INDEX}]")
    print(f"[INFO] Files selected  : {len(selected)}")
    print(f"[INFO] Array           : {ARRAY_NAME}")
    print(f"[INFO] Association     : {assoc}")
    print(f"[INFO] Representation  : {REPRESENTATION}")
    print(f"[INFO] Color preset    : {COLOR_PRESET}")
    print(f"[INFO] Color range     : [{COLOR_RANGE_MIN}, {COLOR_RANGE_MAX}]")
    print(f"[INFO] Below-range    : < {BELOW_RANGE_THRESHOLD} -> {BELOW_RANGE_COLOR}")
    print(f"[INFO] Scalar bar      : {SHOW_SCALAR_BAR}")
    print(f"[INFO] Orientation axes: {ORIENTATION_AXES_VISIBLE}")
    print(f"[INFO] Overwrite       : {OVERWRITE_EXISTING}")

    view = setup_view()

    first_idx, first_file = selected[0]
    source = XMLUnstructuredGridReader(FileName=[str(first_file)])
    source.UpdatePipeline()

    configure_display(source, view, assoc)
    Render()

    if USE_MANUAL_CAMERA:
        apply_manual_camera(view)
        print("[INFO] Applied manual camera settings.")
    else:
        print("[WARN] USE_MANUAL_CAMERA is False, so default ParaView camera will be used.")

    Render()

    for idx, path in selected:
        out_png = output_dir / f"{FRAME_PREFIX}{idx:06d}.png"

        if out_png.exists() and not OVERWRITE_EXISTING:
            print(f"[SKIP] {out_png.name} already exists")
            continue

        if idx != first_idx:
            try:
                Delete(source)
            except Exception:
                pass

            source = XMLUnstructuredGridReader(FileName=[str(path)])
            source.UpdatePipeline()
            configure_display(source, view, assoc)
            apply_manual_camera(view)
            Render()

        save_frame(view, out_png)
        print(f"[OK] Saved {out_png}")

    print("\n[DONE]")
    print(f"Frames saved in: {output_dir}")


if __name__ == "__main__":
    main()
