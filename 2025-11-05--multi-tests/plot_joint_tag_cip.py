#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

CASES = [
    {   "folder": "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87",
        "tag_time_s": 17.03},
    {   "folder": "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24",
        "tag_time_s": 17.03},
    {   "folder": "/Volumes/maxone/2026-02-03--mass_balance/g87_r43",
        "tag_time_s": 14.45},
    {   "folder": "/Volumes/maxone/2026-02-03--mass_balance/g87_r62",
        "tag_time_s": 14.45},
]

# If tag_time_s is None for a case, the largest Time in that case's
# concentration_timeseries.csv is used. Set tag_time_s to the per-case maximum
# TAG time once you have those values.
TAG_TIME_TOLERANCE_SEC = 1.0e-6

OUTPUT_DIR = Path("/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/joint_plots")
OUTPUT_FORMATS = ("png", "svg")
DPI = 600
SHOW_FIGURES = False

TAG_RELATIVE_PATH = Path("TAG/concentrations/concentration_timeseries.csv")
TAG_FALLBACK_RELATIVE_PATH = Path("TAG/concentration_timeseries.csv")
CIP_RELATIVE_PATH_TEMPLATE = "CIP_animation/{case_name}-intensity.csv"

TAG_Y_COLUMN = "Concentration"
TAG_TIME_COLUMN = "Time"
REVERSE_TAG_X = True
NORMALIZE_TAG_Y = True
CIP_X_COLUMN = "time_s"
CIP_Y_COLUMN = "normalized_fraction_above_threshold"
SHIFT_CIP_TO_EARLIEST_START = True

FIGSIZE = (8, 5)
FONT_SIZE = 16
LINEWIDTH = 3
BUPU_RANGE = (0.2, 0.9)
FIRST_SERIES_COLOR = "0.5"

# ============================================================
# HELPERS
# ============================================================


def case_folder(case):
    return Path(case["folder"]).expanduser()


def case_name(case):
    return str(case.get("label") or case_folder(case).name)


def resolve_tag_csv(case):
    if case.get("tag_csv"):
        return Path(case["tag_csv"]).expanduser()

    folder = case_folder(case)
    preferred = folder / TAG_RELATIVE_PATH
    if preferred.exists():
        return preferred

    fallback = folder / TAG_FALLBACK_RELATIVE_PATH
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"TAG concentration_timeseries.csv not found for {case_name(case)}.\n"
        f"Tried:\n  {preferred}\n  {fallback}"
    )


def resolve_cip_csv(case):
    if case.get("cip_csv"):
        return Path(case["cip_csv"]).expanduser()

    folder = case_folder(case)
    preferred = folder / CIP_RELATIVE_PATH_TEMPLATE.format(case_name=folder.name)
    if preferred.exists():
        return preferred

    candidates = sorted(
        path for path in (folder / "CIP_animation").glob("*-intensity.csv")
        if not path.name.startswith("._") and "progress" not in path.name
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise RuntimeError(
            f"Multiple CIP intensity CSVs found for {case_name(case)}; set cip_csv explicitly:\n"
            + "\n".join(f"  {path}" for path in candidates)
        )

    raise FileNotFoundError(
        f"CIP intensity CSV not found for {case_name(case)}.\n"
        f"Tried:\n  {preferred}"
    )


def float_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except ValueError:
        return None


def select_rows_at_time(csv_path, requested_time):
    best_time = None
    best_rows = []
    best_distance = math.inf
    current_time = None
    current_rows = []

    def consider_group(group_time, group_rows):
        nonlocal best_time, best_rows, best_distance
        if group_time is None or not group_rows:
            return

        if requested_time is None:
            if best_time is None or group_time > best_time:
                best_time = group_time
                best_rows = list(group_rows)
            return

        distance = abs(group_time - requested_time)
        if distance < best_distance:
            best_distance = distance
            best_time = group_time
            best_rows = list(group_rows)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if TAG_TIME_COLUMN not in reader.fieldnames:
            raise KeyError(f"{csv_path} does not contain a '{TAG_TIME_COLUMN}' column.")

        for row in reader:
            row_time = float_or_none(row.get(TAG_TIME_COLUMN))
            if row_time is None:
                continue

            if current_time is None or math.isclose(row_time, current_time, abs_tol=1.0e-12):
                current_time = row_time
                current_rows.append(row)
            else:
                consider_group(current_time, current_rows)
                current_time = row_time
                current_rows = [row]

    consider_group(current_time, current_rows)

    if not best_rows:
        raise RuntimeError(f"No rows with usable '{TAG_TIME_COLUMN}' values found in {csv_path}")

    if requested_time is not None and abs(best_time - requested_time) > TAG_TIME_TOLERANCE_SEC:
        print(
            f"[WARN] {csv_path}: requested TAG time {requested_time:g} s was not found; "
            f"using nearest saved time {best_time:g} s."
        )

    return best_time, best_rows


def row_value(row, column, csv_path):
    if column not in row:
        raise KeyError(f"{csv_path} does not contain a '{column}' column.")
    return float(row[column])


def point_columns(rows):
    if not rows:
        return None

    columns = rows[0].keys()
    candidates = [
        ("Points:0", "Points:1", "Points:2"),
        ("Points_0", "Points_1", "Points_2"),
    ]
    for candidate in candidates:
        if all(column in columns for column in candidate):
            return candidate
    return None


def cumulative_arclength_from_points(rows, csv_path):
    columns = point_columns(rows)
    if columns is None:
        return np.arange(len(rows), dtype=float), "Centerline point index"

    points = np.array(
        [[row_value(row, column, csv_path) for column in columns] for row in rows],
        dtype=float,
    )
    if len(points) == 0:
        return np.array([], dtype=float), "Centerline length [cm]"

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return arc, "Centerline length [cm]"


def maybe_reverse_tag_profile(x, y):
    if not REVERSE_TAG_X:
        return x, y
    if len(x) == 0:
        return x, y

    x_flipped = float(np.nanmax(x)) - x
    order = np.argsort(x_flipped)
    return x_flipped[order], y[order]


def maybe_normalize_tag_y(y, case_label):
    if not NORMALIZE_TAG_Y:
        return y, None

    finite = y[np.isfinite(y)]
    if finite.size == 0:
        raise RuntimeError(f"No finite TAG concentration values found for {case_label}.")

    scale = float(np.nanmax(finite))
    if scale <= 0:
        raise RuntimeError(
            f"Cannot normalize TAG profile for {case_label}; max concentration is {scale:g}."
        )

    return y / scale, scale


def load_tag_profile(case):
    csv_path = resolve_tag_csv(case)
    requested_time = case.get("tag_time_s")
    selected_time, rows = select_rows_at_time(csv_path, requested_time)

    y = np.array([row_value(row, TAG_Y_COLUMN, csv_path) for row in rows], dtype=float)
    x, x_label = cumulative_arclength_from_points(rows, csv_path)
    x, y = maybe_reverse_tag_profile(x, y)
    y, normalization_scale = maybe_normalize_tag_y(y, case_name(case))

    finite = np.isfinite(x) & np.isfinite(y)
    return {
        "name": case_name(case),
        "csv_path": csv_path,
        "requested_time": requested_time,
        "selected_time": selected_time,
        "x": x[finite],
        "y": y[finite],
        "x_label": x_label,
        "normalization_scale": normalization_scale,
    }


def load_cip_profile(case):
    csv_path = resolve_cip_csv(case)
    x = []
    y = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for column in [CIP_X_COLUMN, CIP_Y_COLUMN]:
            if column not in reader.fieldnames:
                raise KeyError(f"{csv_path} does not contain a '{column}' column.")

        for row in reader:
            x_val = float_or_none(row.get(CIP_X_COLUMN))
            y_val = float_or_none(row.get(CIP_Y_COLUMN))
            if x_val is None or y_val is None:
                continue
            x.append(x_val)
            y.append(y_val)

    if not x:
        raise RuntimeError(f"No usable CIP rows found in {csv_path}")

    order = np.argsort(x)
    return {
        "name": case_name(case),
        "csv_path": csv_path,
        "x": np.asarray(x, dtype=float)[order],
        "y": np.asarray(y, dtype=float)[order],
    }


def maybe_shift_cip_profiles_to_earliest_start(profiles):
    if not SHIFT_CIP_TO_EARLIEST_START:
        return profiles

    starts = [float(profile["x"][0]) for profile in profiles if len(profile["x"]) > 0]
    if not starts:
        return profiles

    earliest_start = min(starts)
    shifted_profiles = []
    for profile in profiles:
        if len(profile["x"]) == 0:
            shifted_profiles.append(profile)
            continue

        original_start = float(profile["x"][0])
        shifted_profile = dict(profile)
        shifted_profile["x"] = profile["x"] - original_start + earliest_start
        shifted_profile["original_start_time"] = original_start
        shifted_profile["shifted_start_time"] = earliest_start
        shifted_profiles.append(shifted_profile)

    return shifted_profiles


def colors_for_cases(n):
    if n <= 0:
        return []

    cmap = plt.get_cmap("BuPu")
    colors = [FIRST_SERIES_COLOR]
    if n > 1:
        colors.extend(cmap(np.linspace(BUPU_RANGE[0], BUPU_RANGE[1], n - 1)))
    return colors


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        out_path = OUTPUT_DIR / f"{stem}.{fmt}"
        fig.savefig(out_path, format=fmt, dpi=DPI, transparent=True)
        print(f"[OK] wrote {out_path}")


def plot_joint_tag(cases, colors):
    profiles = [load_tag_profile(case) for case in cases]
    x_label = profiles[0]["x_label"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for profile, color in zip(profiles, colors):
        ax.plot(profile["x"], profile["y"], color=color, linewidth=LINEWIDTH)
        print(
            f"[TAG] {profile['name']}: t={profile['selected_time']:g} s, "
            f"rows={len(profile['x'])}, "
            f"norm_scale={profile['normalization_scale']}, "
            f"source={profile['csv_path']}"
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Concentration (TAG)")
    ax.set_xlim(left=0)
    if NORMALIZE_TAG_Y:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(bottom=0)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "joint_TAG")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_joint_cip(cases, colors):
    profiles = maybe_shift_cip_profiles_to_earliest_start(
        [load_cip_profile(case) for case in cases]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for profile, color in zip(profiles, colors):
        ax.plot(profile["x"], profile["y"], color=color, linewidth=LINEWIDTH)
        if SHIFT_CIP_TO_EARLIEST_START:
            shift_note = (
                f"start={profile['original_start_time']:g} -> "
                f"{profile['shifted_start_time']:g} s, "
            )
        else:
            shift_note = ""
        print(
            f"[CIP] {profile['name']}: rows={len(profile['x'])}, "
            f"{shift_note}"
            f"source={profile['csv_path']}"
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration (CIP)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, "joint_CIP")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def main():
    cases = list(CASES)
    if not cases:
        raise ValueError("Add at least one case to CASES.")

    colors = colors_for_cases(len(cases))

    with plt.rc_context({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
    }):
        plot_joint_tag(cases, colors)
        plot_joint_cip(cases, colors)


if __name__ == "__main__":
    main()
