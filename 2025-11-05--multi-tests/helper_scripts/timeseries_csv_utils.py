import csv
import os


def relabel_saved_timeseries_csv(csv_path, selected_rows):
    if not selected_rows or not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        return

    frame_col = None
    for candidate in ["TimeStep", "Time Step", "time_step", "Time", "time"]:
        if candidate in fieldnames:
            frame_col = candidate
            break
    if frame_col is None:
        if len(selected_rows) != 1:
            print(f"[WARN] Could not find a frame column in {csv_path}; leaving saved times unchanged")
            return

        timestep_col = "TimeStep"
        time_col = "Time"
        if timestep_col not in fieldnames:
            fieldnames.insert(0, timestep_col)
        if time_col not in fieldnames:
            fieldnames.insert(1, time_col)

        step = int(selected_rows[0]["step"])
        time_value = float(selected_rows[0]["time"])
        for row in rows:
            row[timestep_col] = step
            row[time_col] = f"{time_value:.12g}"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[INFO] Relabeled single saved timestep in {csv_path}")
        return

    saved_frame_ids = []
    seen_ids = set()
    for row in rows:
        frame_id = row.get(frame_col, "")
        if frame_id not in seen_ids:
            seen_ids.add(frame_id)
            saved_frame_ids.append(frame_id)

    if len(saved_frame_ids) != len(selected_rows):
        print(
            f"[WARN] Saved frame count ({len(saved_frame_ids)}) does not match selected step count "
            f"({len(selected_rows)}) in {csv_path}; leaving saved times unchanged"
        )
        return

    step_map = {
        saved_id: int(row["step"])
        for saved_id, row in zip(saved_frame_ids, selected_rows)
    }
    time_map = {
        saved_id: float(row["time"])
        for saved_id, row in zip(saved_frame_ids, selected_rows)
    }

    timestep_col = next(
        (candidate for candidate in ["TimeStep", "Time Step", "time_step"] if candidate in fieldnames),
        "TimeStep",
    )
    time_col = next(
        (candidate for candidate in ["Time", "time"] if candidate in fieldnames),
        "Time",
    )

    if timestep_col not in fieldnames:
        fieldnames.insert(0, timestep_col)
    if time_col not in fieldnames:
        insert_at = 1 if timestep_col in fieldnames else 0
        fieldnames.insert(insert_at, time_col)

    for row in rows:
        frame_id = row.get(frame_col, "")
        row[timestep_col] = step_map[frame_id]
        row[time_col] = f"{time_map[frame_id]:.12g}"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Relabeled saved times in {csv_path} using physical times from the average file")


def load_saved_timeseries_metadata(csv_path):
    if not os.path.exists(csv_path):
        return [], []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        return [], []

    timestep_col = next(
        (candidate for candidate in ["TimeStep", "Time Step", "time_step"] if candidate in fieldnames),
        None,
    )
    time_col = next(
        (candidate for candidate in ["Time", "time"] if candidate in fieldnames),
        None,
    )

    saved_steps = []
    saved_times = []
    seen_frames = set()
    frame_key_col = timestep_col or time_col
    if frame_key_col is None:
        return [], []

    for row in rows:
        frame_key = row.get(frame_key_col, "")
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)

        if timestep_col is not None:
            try:
                saved_steps.append(int(float(row[timestep_col])))
            except Exception:
                saved_steps.append(None)
        if time_col is not None:
            try:
                saved_times.append(float(row[time_col]))
            except Exception:
                saved_times.append(None)

    return saved_steps, saved_times


def infer_timeseries_frame_columns(fieldnames):
    timestep_col = next(
        (candidate for candidate in ["TimeStep", "Time Step", "time_step"] if candidate in fieldnames),
        None,
    )
    time_col = next(
        (candidate for candidate in ["Time", "time"] if candidate in fieldnames),
        None,
    )
    return timestep_col, time_col


def missing_selected_rows_from_timeseries_csv(csv_path, selected_rows, atol=1e-9):
    if not selected_rows:
        return []

    saved_steps, saved_times = load_saved_timeseries_metadata(csv_path)
    if saved_steps and all(step is not None for step in saved_steps):
        saved_step_set = set(saved_steps)
        return [row for row in selected_rows if int(row["step"]) not in saved_step_set]

    if saved_times and all(time is not None for time in saved_times):
        missing_rows = []
        for row in selected_rows:
            target_time = float(row["time"])
            if not any(abs(saved_time - target_time) <= atol for saved_time in saved_times):
                missing_rows.append(row)
        return missing_rows

    return list(selected_rows)


def load_timeseries_csv_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def load_csv_fieldnames(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def append_timeseries_csv(target_csv_path, new_csv_path):
    if not os.path.exists(new_csv_path):
        return

    new_fieldnames, new_rows = load_timeseries_csv_rows(new_csv_path)
    if not new_rows:
        return

    if not os.path.exists(target_csv_path):
        with open(target_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
        return

    existing_fieldnames = load_csv_fieldnames(target_csv_path)
    if existing_fieldnames != new_fieldnames:
        raise ValueError(
            f"Cannot append {new_csv_path} to {target_csv_path}: CSV headers differ."
        )

    with open(target_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=existing_fieldnames)
        writer.writerows(new_rows)


def frame_sort_key(frame_key):
    kind, value = frame_key
    return (0, value) if kind == "step" else (1, value)


def group_rows_by_frame(rows, timestep_col, time_col):
    groups = {}
    for row in rows:
        if timestep_col is not None and row.get(timestep_col, "") != "":
            frame_key = ("step", int(float(row[timestep_col])))
        elif time_col is not None and row.get(time_col, "") != "":
            frame_key = ("time", float(row[time_col]))
        else:
            frame_key = ("row", len(groups))
        groups.setdefault(frame_key, []).append(row)
    return groups


def merge_timeseries_csv(existing_csv_path, new_csv_path):
    if not os.path.exists(existing_csv_path):
        os.replace(new_csv_path, existing_csv_path)
        return
    if not os.path.exists(new_csv_path):
        return

    existing_fieldnames, existing_rows = load_timeseries_csv_rows(existing_csv_path)
    new_fieldnames, new_rows = load_timeseries_csv_rows(new_csv_path)

    fieldnames = list(existing_fieldnames)
    for field in new_fieldnames:
        if field not in fieldnames:
            fieldnames.append(field)

    timestep_col, time_col = infer_timeseries_frame_columns(fieldnames)
    merged_groups = group_rows_by_frame(existing_rows, timestep_col, time_col)
    new_groups = group_rows_by_frame(new_rows, timestep_col, time_col)

    for frame_key, rows in new_groups.items():
        merged_groups[frame_key] = rows

    merged_frame_keys = sorted(merged_groups.keys(), key=frame_sort_key)
    merged_rows = []
    for frame_key in merged_frame_keys:
        merged_rows.extend(merged_groups[frame_key])

    with open(existing_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
