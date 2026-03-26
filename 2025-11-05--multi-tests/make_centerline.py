#!/usr/bin/env pvpython
import csv
import vtk

# ---------------- user settings ----------------
INPUT_CSV  = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2025-11-05--multi-tests/centerlines/centerline_LCA.csv"
OUTPUT_VTP = "/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing/2025-11-05--multi-tests/centerlines/centerline_LCA.vtp"

HAS_HEADER = True   # set False if there is no header row
X_COL = 0           # column index for x
Y_COL = 1           # column index for y
Z_COL = 2           # column index for z
# ------------------------------------------------


def read_points_from_csv(path, has_header=True, x_col=0, y_col=1, z_col=2):
    coords = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)  # skip header

        for row in reader:
            if not row:
                continue
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                z = float(row[z_col])
            except (ValueError, IndexError):
                # skip any malformed lines
                continue
            coords.append((x, y, z))
    return coords


def build_polyline(coords):
    """
    coords: list of (x, y, z) tuples
    returns vtkPolyData with a single polyline cell
    """
    npts = len(coords)
    if npts < 2:
        raise RuntimeError(f"Need at least 2 points to build a polyline, got {npts}")

    pts = vtk.vtkPoints()
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(npts)

    for i, (x, y, z) in enumerate(coords):
        pid = pts.InsertNextPoint(float(x), float(y), float(z))
        polyline.GetPointIds().SetId(i, pid)

    lines = vtk.vtkCellArray()
    lines.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(lines)

    return polydata


def write_vtp(polydata, out_path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()  # smaller file; optional
    writer.Write()


def main():
    print(f"[INFO] Reading points from {INPUT_CSV}")
    coords = read_points_from_csv(
        INPUT_CSV,
        has_header=HAS_HEADER,
        x_col=X_COL,
        y_col=Y_COL,
        z_col=Z_COL,
    )
    print(f"[INFO] Loaded {len(coords)} points")

    if len(coords) < 2:
        raise SystemExit("[ERROR] Not enough points to build a centerline.")

    print("[INFO] Building polyline...")
    polydata = build_polyline(coords)

    print(f"[INFO] Writing VTP to {OUTPUT_VTP}")
    write_vtp(polydata, OUTPUT_VTP)
    print("[OK] Done.")


if __name__ == "__main__":
    main()