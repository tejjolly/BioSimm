#!/usr/bin/env python3
"""
Count the number of cells with concentration above a threshold in result_###.vtu
files and create a fraction-above-threshold-vs-time profile.

Edit the SETTINGS block below. No command-line parsing is used.
"""

import csv
import os
import re
import sys
from pathlib import Path

# ============================================================
# SETTINGS — EDIT THESE
# ============================================================

RESULTS_DIR = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24/96-procs"

ARRAY_NAME = "Concentration"   # Name of the concentration array
THRESHOLD = 5.0                # Count cells with concentration > THRESHOLD
DT = 0.01                      # Seconds per result index, e.g. result_346 -> 3.46 s

T_START = 5.16                  # Start time in seconds
T_END = 26.66                  # End time in seconds; use None for last available file

NORMALIZE_Y = True            # If True, plot y-axis as 0 to 1
STRICT_TIME_WINDOW = False     # If True, only include files with exact implied time in [T_START, T_END]

# Leave as None to auto-name outputs in the configured output directories
OUT_PREFIX = None

LOCAL_OUTPUT_DIR = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2025-11-05--multi-tests/images"
SAVE_ANIMATION = True

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


def build_output_prefixes(results_dir: Path, out_prefix):
    stem = build_output_stem(results_dir, out_prefix)
    prefixes = [
        Path(LOCAL_OUTPUT_DIR) / stem,
        results_dir.parent / stem,
    ]

    unique_prefixes = []
    seen = set()
    for prefix in prefixes:
        resolved = str(prefix)
        if resolved not in seen:
            seen.add(resolved)
            unique_prefixes.append(prefix)
    return unique_prefixes


def load_cached_results(output_prefixes, threshold: float, array_name: str):
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

                try:
                    fraction = float(row["fraction_above_threshold"])
                    count = int(row["cells_above_threshold"])
                    n_cells = int(row["total_cells"])
                except (KeyError, TypeError, ValueError):
                    continue

                cached[row_idx] = {
                    "fraction": fraction,
                    "count": count,
                    "total_cells": n_cells,
                    "data_origin": row.get("data_origin", "cached"),
                }

        if cached:
            return cached, csv_path

    return {}, None


def write_csv(csv_path: Path, times, selected, fractions, fractions_norm, counts, total_cells, data_origin):
    with open(csv_path, "w") as f:
        f.write(
            "time_s,file_index,fraction_above_threshold,normalized_fraction_above_threshold,"
            "cells_above_threshold,total_cells,threshold,array_name,data_origin\n"
        )
        for t, (idx, _), frac, frac_norm, count, n_cells in zip(
            times, selected, fractions, fractions_norm, counts, total_cells
        ):
            f.write(
                f"{t:.8f},{idx},{frac:.8f},{frac_norm:.8f},{int(count)},{int(n_cells)},"
                f"{THRESHOLD},{ARRAY_NAME},{data_origin}\n"
            )


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
    return out_prefix.parent / "CIP_animation" / f"{out_prefix.name}-{tag}"


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

    out_prefixes = build_output_prefixes(results_dir, OUT_PREFIX)
    cached_results, cache_source = load_cached_results(out_prefixes, THRESHOLD, ARRAY_NAME)
    if cache_source is not None:
        print(f"[INFO] Cache source  : {cache_source}")
        print(f"[INFO] Cached rows   : {len(cached_results)}")

    times = []
    counts = []
    fractions = []
    total_cells = []
    data_origin_used = None

    for idx, path in selected:
        cached = cached_results.get(idx)
        if cached is not None:
            fraction_above = float(cached["fraction"])
            count_above = int(cached["count"])
            n_cells = int(cached["total_cells"])
            origin = cached["data_origin"]
            if data_origin_used is None:
                data_origin_used = origin
            print(
                f"[CACHE] {path.name}: time={idx * DT:.4f} s, fraction_above_{THRESHOLD}={fraction_above:.8f}, "
                f"cells_above_{THRESHOLD}={count_above}/{n_cells}"
            )
        else:
            values, origin = read_array_as_cell_array(path, ARRAY_NAME)
            if data_origin_used is None:
                data_origin_used = origin

            count_above = int(np.count_nonzero(values > THRESHOLD))
            n_cells = int(values.size)
            fraction_above = (count_above / n_cells) if n_cells > 0 else 0.0

            print(
                f"[OK] {path.name}: time={idx * DT:.4f} s, fraction_above_{THRESHOLD}={fraction_above:.8f}, "
                f"cells_above_{THRESHOLD}={count_above}/{n_cells}"
            )

        t = idx * DT

        times.append(t)
        counts.append(count_above)
        fractions.append(fraction_above)
        total_cells.append(n_cells)

    times = np.asarray(times, dtype=float)
    counts = np.asarray(counts, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    total_cells = np.asarray(total_cells, dtype=int)

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
                data_origin_used,
            )
            save_plot(frac_png_path, times, fractions, "Pixel Count Fraction", fraction_y_limits)
            saved_csv_paths.append(csv_path)
            saved_png_paths.append(frac_png_path)
            if NORMALIZE_Y:
                save_plot(norm_png_path, times, fractions_norm, "Normalized Pixel Count", norm_y_limits)
                saved_png_paths.append(norm_png_path)

            if SAVE_ANIMATION:
                frac_animation_dir = build_animation_dir(out_prefix, "frac")
                save_animation(
                    frac_animation_dir,
                    times,
                    fractions,
                    "Pixel Count Fraction",
                    fraction_y_limits,
                )
                saved_animation_dirs.append(frac_animation_dir)

                if NORMALIZE_Y:
                    norm_animation_dir = build_animation_dir(out_prefix, "norm")
                    save_animation(
                        norm_animation_dir,
                        times,
                        fractions_norm,
                        "Normalized Pixel Count",
                        norm_y_limits,
                    )
                    saved_animation_dirs.append(norm_animation_dir)
        except Exception as exc:
            print(f"[WARN] Could not save outputs to {out_prefix.parent}: {exc}")

    print("\n[DONE]")
    for csv_path in saved_csv_paths:
        print(f"CSV written to: {csv_path}")
    for png_path in saved_png_paths:
        print(f"PNG written to: {png_path}")
    for animation_dir in saved_animation_dirs:
        print(f"Animation frames written to: {animation_dir}")
    print(f"Data origin used: {data_origin_used}")


if __name__ == "__main__":
    main()
