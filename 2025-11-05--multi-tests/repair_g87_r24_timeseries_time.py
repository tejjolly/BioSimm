#!/usr/bin/env python3
"""
One-off repair for g87_r24 concentration_timeseries.csv.

The early rows have Time values that were written as timestep-like values.
Rows with TimeStep <= MAX_REPAIR_STEP are repaired using:

    Time = TimeStep * DT

Rows after MAX_REPAIR_STEP are left unchanged.
"""

import csv
from pathlib import Path


CSV_PATH = Path(
    "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/"
    "g87_r24/TAG/concentrations/concentration_timeseries.csv"
)
DT = 0.01
MAX_REPAIR_STEP = 1579


def make_unique_backup_path(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        return backup_path

    counter = 1
    while True:
        candidate = path.with_suffix(path.suffix + f".bak.{counter:03d}")
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    tmp_path = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    backup_path = make_unique_backup_path(CSV_PATH)

    repaired_rows = 0
    unchanged_rows = 0

    with CSV_PATH.open("r", newline="") as f_in, tmp_path.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise RuntimeError(f"No CSV header found in {CSV_PATH}")
        if "TimeStep" not in fieldnames or "Time" not in fieldnames:
            raise RuntimeError(f"Need TimeStep and Time columns; found: {fieldnames}")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            step = int(float(row["TimeStep"]))
            if step <= MAX_REPAIR_STEP:
                row["Time"] = f"{step * DT:.12g}"
                repaired_rows += 1
            else:
                unchanged_rows += 1
            writer.writerow(row)

    CSV_PATH.rename(backup_path)
    tmp_path.rename(CSV_PATH)

    print(f"Repaired rows      : {repaired_rows}")
    print(f"Unchanged rows     : {unchanged_rows}")
    print(f"Backup written to  : {backup_path}")
    print(f"Repaired CSV       : {CSV_PATH}")


if __name__ == "__main__":
    main()
