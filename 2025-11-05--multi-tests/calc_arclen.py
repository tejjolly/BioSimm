from csv import DictReader, writer
from pathlib import Path
from math import sqrt


INPUT_CSV = Path(__file__).resolve().parent / "centerlines" / "centerline_LCA.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "centerlines" / "centerline_LCA_arclen.csv"

POINT_COLUMN_SETS = [
    ("Points_0", "Points_1", "Points_2"),
    ("Points:0", "Points:1", "Points:2"),
]


def get_point_columns(fieldnames):
    for cols in POINT_COLUMN_SETS:
        if all(col in fieldnames for col in cols):
            return cols
    raise ValueError(f"Could not find point columns in {fieldnames}")


with INPUT_CSV.open(newline="") as f:
    reader = DictReader(f)
    rows = list(reader)
    point_cols = get_point_columns(reader.fieldnames or [])


points = []
for row in rows:
    values = [row[col].strip() for col in point_cols]
    if any(value == "" for value in values):
        continue
    points.append(tuple(float(value) for value in values))

arc = [0.0]
for i in range(1, len(points)):
    x0, y0, z0 = points[i - 1]
    x1, y1, z1 = points[i]
    ds = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    arc.append(arc[-1] + ds)


with OUTPUT_CSV.open("w", newline="") as f:
    csv_writer = writer(f)
    csv_writer.writerow(["ArcLength"])
    for value in arc:
        csv_writer.writerow([value])


print(f"Saved {len(arc)} arc-length values to {OUTPUT_CSV}")
