#!/usr/bin/env python3
"""
Count the number of cells with concentration above a threshold in result_###.vtu
files and create a fraction-above-threshold-vs-time profile.

This version can exclude cells whose CENTERS fall outside a spatial mask
(e.g. to remove the aorta and keep only coronaries).

It also:
- saves outputs to the local output folder
- saves outputs to the case-level CIP_animation folder
- snapshots any existing non-snapshot contents in that case-level CIP_animation folder
  into _ss-YYYYMMDD-HHMMSS before writing new outputs

Edit the SETTINGS block below. No command-line parsing is used.
"""

import csv
import os
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

# ============================================================
# SETTINGS — EDIT THESE
# ============================================================

RESULTS_DIR = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24/96-procs"

ARRAY_NAME = "Concentration"   # Name of the concentration array
THRESHOLD = 5.0                # Count cells with concentration > THRESHOLD
DT = 0.01                      # Seconds per result index, e.g. result_346 -> 3.46 s

T_START = 4.3                 # Start time in seconds
T_END = 26.66                # End time in seconds; use None for last available file

NORMALIZE_Y = True             # If True, plot y-axis as 0 to 1
STRICT_TIME_WINDOW = False     # If True, only include files with exact implied time in [T_START, T_END]

# Leave as None to auto-name outputs in the configured output directories
OUT_PREFIX = None

LOCAL_OUTPUT_DIR = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2025-11-05--multi-tests/images"
SAVE_ANIMATION = True
WRITE_DEBUG_MASK_VTU = True
DEBUG_MASK_ONLY_FIRST_FILE = True

# Path to pvpython for auto-relaunch if vtk is unavailable in regular python
PVPYTHON_PATH = "/Applications/ParaView-5.13.1.app/Contents/bin/pvpython"

# ------------------------------------------------------------
# SPATIAL EXCLUSION SETTINGS
# ------------------------------------------------------------

USE_CACHE = False  # keep False initially so you do not reuse old non-excluded CSVs

# Plane-based mask settings
EXCLUDE_MODE = "multi_plane"   # allowed: "none", "plane", "multi_plane", "box"
PLANE_TOL = 1.0e-12

PLANES = [
    {
        "origin": (-0.655774, -4.97082, -14.8494),
        "normal": (-0.667198, -0.325845, -0.66983),
        "keep_positive_side": True,
        "name": "plane_1",
    },
    {
        "origin": (2.35393, -1.71064, -13.331),
        "normal": (0.809434, -0.112858, 0.576263),
        "keep_positive_side": True,
        "name": "plane_2",
    },
]

MULTI_PLANE_COMBINE_MODE = "union"   # allowed: "union", "intersection"

# Rotated-box mask settings
EXCLUDE_BOX = True

# Box values you gave
BOX_POSITION = (3.2026, -3.64914, -14.3248)
BOX_ROTATION = (-26.8878, 20.271, 106.829)   # degrees
BOX_LENGTH = (3.23458, 3.3829, 3.4215)

# Rotation order for the box orientation. Start with "XYZ".
# If the box looks misaligned, the first thing to try is changing this.
BOX_ROTATION_ORDER = "XYZ"

# Tiny tolerance for inside-box test
BOX_TOL = 1.0e-12

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


def read_grid_and_array_as_cell_array(vtu_path: Path, array_name: str):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    data = reader.GetOutput()

    cell_data = data.GetCellData()
    arr = cell_data.GetArray(array_name)
    if arr is not None:
        return data, vtk_to_numpy(arr), "cell"

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
        return converted, vtk_to_numpy(carr), "point->cell"

    raise RuntimeError(
        f"Array '{array_name}' not found in either cell data or point data "
        f"for file: {vtu_path}"
    )


def build_geometry_name(results_dir: Path):
    geometry_name = results_dir.parent.name.strip()
    if not geometry_name:
        raise RuntimeError(f"Could not determine geometry name from results dir:\n{results_dir}")
    return geometry_name


def build_output_stem(results_dir: Path, out_prefix):
    if out_prefix:
        return Path(out_prefix).name
    geometry_name = build_geometry_name(results_dir)
    return f"{geometry_name}-intensity"


def get_results_output_root(results_dir: Path):
    return results_dir.parent / "CIP_animation"


def build_output_prefixes(results_dir: Path, out_prefix):
    stem = build_output_stem(results_dir, out_prefix)

    prefixes = [
        Path(LOCAL_OUTPUT_DIR) / stem,
        get_results_output_root(results_dir) / stem,
    ]

    unique_prefixes = []
    seen = set()
    for prefix in prefixes:
        resolved = str(prefix)
        if resolved not in seen:
            seen.add(resolved)
            unique_prefixes.append(prefix)
    return unique_prefixes


def tuple_to_string(x):
    return ",".join(f"{float(v):.12g}" for v in x)


def build_spatial_signature():
    if EXCLUDE_MODE == "none":
        return "exclude_mode=none"
    if EXCLUDE_MODE == "plane":
        if len(PLANES) == 0:
            return "exclude_mode=plane;planes=0"
        plane = PLANES[0]
        return (
            f"exclude_mode=plane;"
            f"origin={tuple_to_string(plane['origin'])};"
            f"normal={tuple_to_string(plane['normal'])};"
            f"keep_positive_side={plane['keep_positive_side']}"
        )
    if EXCLUDE_MODE == "multi_plane":
        parts = [
            "exclude_mode=multi_plane",
            f"combine_mode={MULTI_PLANE_COMBINE_MODE}",
        ]
        for i, plane in enumerate(PLANES):
            prefix = f"plane_{i}"
            parts.extend([
                f"{prefix}_origin={tuple_to_string(plane['origin'])}",
                f"{prefix}_normal={tuple_to_string(plane['normal'])}",
                f"{prefix}_keep_positive_side={plane['keep_positive_side']}",
            ])
        return ";".join(parts)
    if EXCLUDE_MODE == "box":
        return (
            f"exclude_mode=box;"
            f"position={tuple_to_string(BOX_POSITION)};"
            f"rotation={tuple_to_string(BOX_ROTATION)};"
            f"length={tuple_to_string(BOX_LENGTH)};"
            f"rotation_order={BOX_ROTATION_ORDER}"
        )
    raise ValueError(f"Unsupported EXCLUDE_MODE: {EXCLUDE_MODE}")


def load_cached_results(output_prefixes, threshold: float, array_name: str, spatial_signature: str):
    for out_prefix in output_prefixes:
        csv_path = out_prefix.with_suffix(".csv")
        if not csv_path.exists():
            continue

        cached = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row_idx = int(row["file_index"])
                    row_threshold = float(row["threshold"])
                    row_array_name = row["array_name"]
                except (KeyError, TypeError, ValueError):
                    continue

                if not math.isclose(row_threshold, threshold, rel_tol=0.0, abs_tol=1e-12):
                    continue
                if row_array_name != array_name:
                    continue

                row_spatial_signature = row.get("spatial_signature", "")
                if row_spatial_signature != spatial_signature:
                    continue

                try:
                    fraction = float(row["fraction_above_threshold"])
                    count = int(row["cells_above_threshold"])
                    n_cells = int(row["total_cells"])
                    excluded_cells = int(row.get("excluded_cells", 0))
                    original_total_cells = int(row.get("original_total_cells", n_cells + excluded_cells))
                except (KeyError, TypeError, ValueError):
                    continue

                cached[row_idx] = {
                    "fraction": fraction,
                    "count": count,
                    "total_cells": n_cells,
                    "excluded_cells": excluded_cells,
                    "original_total_cells": original_total_cells,
                    "data_origin": row.get("data_origin", "cached"),
                }

        if cached:
            return cached, csv_path

    return {}, None


def write_csv(
    csv_path: Path,
    times,
    selected,
    fractions,
    fractions_norm,
    counts,
    total_cells,
    excluded_cells,
    original_total_cells,
    data_origin,
    spatial_signature,
):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "file_index",
            "fraction_above_threshold",
            "normalized_fraction_above_threshold",
            "cells_above_threshold",
            "total_cells",
            "excluded_cells",
            "original_total_cells",
            "threshold",
            "array_name",
            "data_origin",
            "spatial_signature",
        ])

        for t, (idx, _), frac, frac_norm, count, n_cells, n_excl, n_orig in zip(
            times,
            selected,
            fractions,
            fractions_norm,
            counts,
            total_cells,
            excluded_cells,
            original_total_cells,
        ):
            writer.writerow([
                f"{t:.8f}",
                idx,
                f"{frac:.8f}",
                f"{frac_norm:.8f}",
                int(count),
                int(n_cells),
                int(n_excl),
                int(n_orig),
                THRESHOLD,
                ARRAY_NAME,
                data_origin,
                spatial_signature,
            ])


def get_fraction_y_limits(fractions):
    if len(fractions) == 0:
        return (0.0, 1.0)
    y_max = float(np.max(fractions))
    if y_max <= 0.0:
        y_max = 1.0
    return (0.0, min(1.0, y_max * 1.05))


def get_x_limits(times):
    if len(times) == 0:
        return (0.0, 1.0)
    x_min = 0.0
    x_max = float(times[-1])
    if math.isclose(x_max, x_min):
        x_max = x_min + max(DT, 1.0)
    return (x_min, x_max)


def style_axes(ax, times, y_label, y_limits):
    x_min, x_max = get_x_limits(times)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*y_limits)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_plot(plot_path: Path, times, y, y_label, y_limits):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, y, color="black", linewidth=1.5)
    style_axes(ax, times, y_label, y_limits)
    fig.tight_layout()
    fig.savefig(plot_path, transparent=True)
    plt.close(fig)


def build_png_paths(out_prefix: Path):
    frac_png_path = out_prefix.parent / f"{out_prefix.name}-frac.svg"
    norm_png_path = out_prefix.parent / f"{out_prefix.name}-norm.svg"
    return frac_png_path, norm_png_path


def build_animation_dir(out_prefix: Path, tag: str):
    return out_prefix.parent / f"{out_prefix.name}-{tag}"


def save_animation(animation_dir: Path, times, y, y_label, y_limits):
    animation_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(1, len(times) + 1):
        frame_path = animation_dir / f"frame_{frame_idx:04d}.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times[:frame_idx], y[:frame_idx], color="black", linewidth=1.5)
        style_axes(ax, times, y_label, y_limits)
        fig.tight_layout()
        fig.savefig(frame_path, dpi=200, transparent=True)
        plt.close(fig)


def get_cell_centers_numpy(grid):
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(grid)
    cell_centers.Update()
    pts = cell_centers.GetOutput().GetPoints()
    if pts is None:
        return np.empty((0, 3), dtype=float)
    return vtk_to_numpy(pts.GetData())


def apply_rotation_order(transform, rotation_deg, order):
    rx, ry, rz = rotation_deg
    for axis in order.upper():
        if axis == "X":
            transform.RotateX(rx)
        elif axis == "Y":
            transform.RotateY(ry)
        elif axis == "Z":
            transform.RotateZ(rz)
        else:
            raise ValueError(f"Unsupported rotation axis in order '{order}': {axis}")


def vtk_matrix_to_numpy(vtk_matrix):
    m = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            m[i, j] = vtk_matrix.GetElement(i, j)
    return m


def build_keep_mask_from_exclusion_box(cell_centers):
    n = cell_centers.shape[0]
    if not EXCLUDE_BOX:
        return np.ones(n, dtype=bool)

    if n == 0:
        return np.ones(0, dtype=bool)

    transform = vtk.vtkTransform()
    transform.Identity()

    apply_rotation_order(transform, BOX_ROTATION, BOX_ROTATION_ORDER)
    transform.Translate(*BOX_POSITION)

    inverse = vtk.vtkTransform()
    inverse.DeepCopy(transform)
    inverse.Inverse()

    M = vtk_matrix_to_numpy(inverse.GetMatrix())

    pts_h = np.column_stack([cell_centers, np.ones(n, dtype=float)])
    pts_local_h = pts_h @ M.T
    pts_local = pts_local_h[:, :3]

    half_lengths = 0.5 * np.asarray(BOX_LENGTH, dtype=float)

    inside_box = np.all(
        np.abs(pts_local) <= (half_lengths + BOX_TOL),
        axis=1
    )

    keep_mask = ~inside_box
    return keep_mask


def build_side_mask_from_plane(cell_centers, origin, normal, keep_positive_side, tol):
    n = cell_centers.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)

    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)

    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("Plane normal cannot be zero.")

    normal = normal / norm
    signed_dist = (cell_centers - origin) @ normal

    if keep_positive_side:
        return signed_dist >= -tol
    else:
        return signed_dist <= tol


def build_keep_mask_from_plane(cell_centers):
    if len(PLANES) == 0:
        return np.ones(cell_centers.shape[0], dtype=bool)

    plane = PLANES[0]
    return build_side_mask_from_plane(
        cell_centers,
        plane["origin"],
        plane["normal"],
        plane["keep_positive_side"],
        PLANE_TOL,
    )


def build_keep_mask_from_multi_plane(cell_centers):
    n = cell_centers.shape[0]
    if n == 0:
        return np.ones(0, dtype=bool)

    if len(PLANES) == 0:
        return np.ones(n, dtype=bool)

    masks = []
    for plane in PLANES:
        mask = build_side_mask_from_plane(
            cell_centers,
            plane["origin"],
            plane["normal"],
            plane["keep_positive_side"],
            PLANE_TOL,
        )
        masks.append(mask)

    if MULTI_PLANE_COMBINE_MODE == "union":
        keep_mask = masks[0].copy()
        for mask in masks[1:]:
            keep_mask |= mask
        return keep_mask

    elif MULTI_PLANE_COMBINE_MODE == "intersection":
        keep_mask = masks[0].copy()
        for mask in masks[1:]:
            keep_mask &= mask
        return keep_mask

    else:
        raise ValueError(f"Unsupported MULTI_PLANE_COMBINE_MODE: {MULTI_PLANE_COMBINE_MODE}")


def build_keep_mask(cell_centers):
    n = cell_centers.shape[0]

    if EXCLUDE_MODE == "none":
        return np.ones(n, dtype=bool)
    elif EXCLUDE_MODE == "plane":
        return build_keep_mask_from_plane(cell_centers)
    elif EXCLUDE_MODE == "multi_plane":
        return build_keep_mask_from_multi_plane(cell_centers)
    elif EXCLUDE_MODE == "box":
        return build_keep_mask_from_exclusion_box(cell_centers)
    else:
        raise ValueError(f"Unsupported EXCLUDE_MODE: {EXCLUDE_MODE}")


def write_debug_mask_vtu(debug_vtu_path: Path, grid, keep_mask):
    debug_grid = vtk.vtkUnstructuredGrid()
    debug_grid.DeepCopy(grid)

    included_arr = vtk.vtkUnsignedCharArray()
    included_arr.SetName("IncludedMask")
    included_arr.SetNumberOfTuples(keep_mask.size)

    excluded_arr = vtk.vtkUnsignedCharArray()
    excluded_arr.SetName("ExcludedMask")
    excluded_arr.SetNumberOfTuples(keep_mask.size)

    for i, keep in enumerate(keep_mask):
        included_arr.SetValue(i, 1 if keep else 0)
        excluded_arr.SetValue(i, 0 if keep else 1)

    debug_grid.GetCellData().AddArray(included_arr)
    debug_grid.GetCellData().AddArray(excluded_arr)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(debug_vtu_path))
    writer.SetInputData(debug_grid)
    writer.Write()


def make_unique_snapshot_dir(parent_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_dir = parent_dir / f"_ss-{timestamp}"

    counter = 1
    while snapshot_dir.exists():
        snapshot_dir = parent_dir / f"_ss-{timestamp}-{counter:02d}"
        counter += 1

    return snapshot_dir


def snapshot_existing_results_output(results_output_root: Path):
    results_output_root.mkdir(parents=True, exist_ok=True)

    entries_to_snapshot = [
        entry for entry in results_output_root.iterdir()
        if not entry.name.startswith("_ss-")
    ]
    if not entries_to_snapshot:
        return None

    snapshot_dir = make_unique_snapshot_dir(results_output_root)
    snapshot_dir.mkdir(parents=False, exist_ok=False)

    for entry in entries_to_snapshot:
        shutil.move(str(entry), str(snapshot_dir / entry.name))

    return snapshot_dir


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

    print(f"[INFO] Results dir      : {results_dir}")
    print(f"[INFO] Array name       : {ARRAY_NAME}")
    print(f"[INFO] Threshold        : {THRESHOLD}")
    print(f"[INFO] dt               : {DT}")
    print(f"[INFO] Time window      : [{t_start}, {t_end}] s")
    print(f"[INFO] Files selected   : {len(selected)}")
    print(f"[INFO] First/last       : result_{selected[0][0]}.vtu to result_{selected[-1][0]}.vtu")
    print(f"[INFO] Exclude mode     : {EXCLUDE_MODE}")
    if EXCLUDE_MODE == "plane":
        if len(PLANES) > 0:
            plane = PLANES[0]
            print(f"[INFO] Plane name       : {plane.get('name', 'plane_0')}")
            print(f"[INFO] Plane origin     : {plane['origin']}")
            print(f"[INFO] Plane normal     : {plane['normal']}")
            print(f"[INFO] Keep positive    : {plane['keep_positive_side']}")
        else:
            print("[INFO] Plane count      : 0")
    if EXCLUDE_MODE == "multi_plane":
        print(f"[INFO] Plane combine    : {MULTI_PLANE_COMBINE_MODE}")
        for i, plane in enumerate(PLANES):
            print(f"[INFO] Plane {i} name    : {plane.get('name', f'plane_{i}')}")
            print(f"[INFO] Plane {i} origin  : {plane['origin']}")
            print(f"[INFO] Plane {i} normal  : {plane['normal']}")
            print(f"[INFO] Plane {i} keep +  : {plane['keep_positive_side']}")
    if EXCLUDE_MODE == "box":
        print(f"[INFO] Box position     : {BOX_POSITION}")
        print(f"[INFO] Box rotation     : {BOX_ROTATION}")
        print(f"[INFO] Box length       : {BOX_LENGTH}")
        print(f"[INFO] Rotation order   : {BOX_ROTATION_ORDER}")

    out_prefixes = build_output_prefixes(results_dir, OUT_PREFIX)
    spatial_signature = build_spatial_signature()

    if USE_CACHE:
        cached_results, cache_source = load_cached_results(
            out_prefixes,
            THRESHOLD,
            ARRAY_NAME,
            spatial_signature,
        )
    else:
        cached_results, cache_source = {}, None

    if cache_source is not None:
        print(f"[INFO] Cache source     : {cache_source}")
        print(f"[INFO] Cached rows      : {len(cached_results)}")

    results_output_root = get_results_output_root(results_dir)
    snapshot_dir = snapshot_existing_results_output(results_output_root)
    if snapshot_dir is not None:
        print(f"[INFO] Existing results-side contents moved to: {snapshot_dir}")

    times = []
    counts = []
    fractions = []
    total_cells = []
    excluded_cells = []
    original_total_cells = []
    data_origin_used = None

    for idx, path in selected:
        cached = cached_results.get(idx)

        if cached is not None:
            fraction_above = float(cached["fraction"])
            count_above = int(cached["count"])
            n_cells = int(cached["total_cells"])
            n_excluded = int(cached["excluded_cells"])
            n_original = int(cached["original_total_cells"])
            origin = cached["data_origin"]

            if data_origin_used is None:
                data_origin_used = origin

            print(
                f"[CACHE] {path.name}: time={idx * DT:.4f} s, "
                f"fraction_above_{THRESHOLD}={fraction_above:.8f}, "
                f"cells_above_{THRESHOLD}={count_above}/{n_cells}, "
                f"excluded={n_excluded}, original_total={n_original}"
            )
        else:
            grid, values, origin = read_grid_and_array_as_cell_array(path, ARRAY_NAME)
            if data_origin_used is None:
                data_origin_used = origin

            cell_centers = get_cell_centers_numpy(grid)
            if values.size != cell_centers.shape[0]:
                raise RuntimeError(
                    f"Mismatch in {path.name}: values has {values.size} entries but "
                    f"cell centers has {cell_centers.shape[0]} entries."
                )

            keep_mask = build_keep_mask(cell_centers)
            if WRITE_DEBUG_MASK_VTU:
                should_write_debug = True
                if DEBUG_MASK_ONLY_FIRST_FILE and idx != selected[0][0]:
                    should_write_debug = False

                if should_write_debug:
                    debug_vtu_path = get_results_output_root(results_dir) / f"debug_clip_{path.stem}.vtu"
                    write_debug_mask_vtu(debug_vtu_path, grid, keep_mask)
                    print(f"[DEBUG] Mask VTU written to: {debug_vtu_path}")

            values_kept = values[keep_mask]

            n_original = int(values.size)
            n_included = int(np.count_nonzero(keep_mask))
            n_cells = int(values_kept.size)
            n_excluded = int(n_original - n_included)

            if n_cells != n_included:
                raise RuntimeError(
                    f"Mismatch in {path.name}: n_cells={n_cells} but n_included={n_included}"
                )

            print(
                f"[MASK] {path.name}: included={n_included}, excluded={n_excluded}, original_total={n_original}"
            )

            count_above = int(np.count_nonzero(values_kept > THRESHOLD))
            fraction_above = (count_above / n_cells) if n_cells > 0 else 0.0

            print(
                f"[OK] {path.name}: time={idx * DT:.4f} s, "
                f"fraction_above_{THRESHOLD}={fraction_above:.8f}, "
                f"cells_above_{THRESHOLD}={count_above}/{n_cells}, "
                f"included={n_included}, excluded={n_excluded}, original_total={n_original}"
            )

        t = idx * DT

        times.append(t)
        counts.append(count_above)
        fractions.append(fraction_above)
        total_cells.append(n_cells)
        excluded_cells.append(n_excluded)
        original_total_cells.append(n_original)

    times = np.asarray(times, dtype=float)
    counts = np.asarray(counts, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    total_cells = np.asarray(total_cells, dtype=int)
    excluded_cells = np.asarray(excluded_cells, dtype=int)
    original_total_cells = np.asarray(original_total_cells, dtype=int)

    max_fraction = float(np.max(fractions)) if len(fractions) > 0 else 0.0
    fractions_norm = fractions / max_fraction if max_fraction > 0 else np.zeros_like(fractions)
    fraction_y_limits = get_fraction_y_limits(fractions)
    norm_y_limits = (0.0, 1.0)

    saved_csv_paths = []
    saved_png_paths = []
    saved_animation_dirs = []

    for out_prefix in out_prefixes:
        csv_path = out_prefix.with_suffix(".csv")
        frac_png_path, norm_png_path = build_png_paths(out_prefix)

        try:
            out_prefix.parent.mkdir(parents=True, exist_ok=True)

            write_csv(
                csv_path,
                times,
                selected,
                fractions,
                fractions_norm,
                counts,
                total_cells,
                excluded_cells,
                original_total_cells,
                data_origin_used,
                spatial_signature,
            )

            save_plot(frac_png_path, times, fractions, "Cell Count Fraction", fraction_y_limits)
            saved_csv_paths.append(csv_path)
            saved_png_paths.append(frac_png_path)

            if NORMALIZE_Y:
                save_plot(norm_png_path, times, fractions_norm, "Normalized Cell Count", norm_y_limits)
                saved_png_paths.append(norm_png_path)

            if SAVE_ANIMATION:
                frac_animation_dir = build_animation_dir(out_prefix, "frac")
                save_animation(
                    frac_animation_dir,
                    times,
                    fractions,
                    "Cell Count Fraction",
                    fraction_y_limits,
                )
                saved_animation_dirs.append(frac_animation_dir)

                if NORMALIZE_Y:
                    norm_animation_dir = build_animation_dir(out_prefix, "norm")
                    save_animation(
                        norm_animation_dir,
                        times,
                        fractions_norm,
                        "Normalized Cell Count",
                        norm_y_limits,
                    )
                    saved_animation_dirs.append(norm_animation_dir)

        except Exception as exc:
            print(f"[WARN] Could not save outputs to {out_prefix.parent}: {exc}")

    print("\n[DONE]")
    for csv_path in saved_csv_paths:
        print(f"CSV written to: {csv_path}")
    for png_path in saved_png_paths:
        print(f"Plot written to: {png_path}")
    for animation_dir in saved_animation_dirs:
        print(f"Animation frames written to: {animation_dir}")
    print(f"Data origin used: {data_origin_used}")


if __name__ == "__main__":
    main()
