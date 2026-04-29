import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ============================================================
# USER SETTINGS
# ============================================================
# INPUT_DIR = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24/96-procs"
INPUT_DIR = "/Volumes/maxone/2026-02-03--mass_balance/g87_r43/96-procs"

FILE_PREFIX = "result_"
FILE_SUFFIX = ".vtu"
INDEX_PADDING = 0          # use 0 for no zero-padding, or e.g. 4 for result_0001.vtu

START_INDEX = 946
END_INDEX = START_INDEX + 85
INDEX_STEP = 1

OUTPUT_FILENAME = "cycle_average.vtu"

# Safety / behavior
CHECK_POINT_COORDS = False   # verifies all VTUs use the same point coordinates
CHECK_NUM_CELLS = False      # verifies same number of cells
VERBOSE = True
# ============================================================


def make_filename(idx):
    if INDEX_PADDING > 0:
        return f"{FILE_PREFIX}{idx:0{INDEX_PADDING}d}{FILE_SUFFIX}"
    return f"{FILE_PREFIX}{idx}{FILE_SUFFIX}"


def read_vtu(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    grid = reader.GetOutput()

    if grid is None or grid.GetNumberOfPoints() == 0:
        raise ValueError(f"Failed to read VTU or VTU is empty: {path}")

    return grid


def get_array_numpy(vtk_array):
    arr = vtk_to_numpy(vtk_array)
    ncomp = vtk_array.GetNumberOfComponents()

    if ncomp == 1:
        arr = arr.reshape(-1, 1)
    else:
        arr = arr.reshape(-1, ncomp)

    return arr


def copy_active_attributes(src_data, dst_data):
    if src_data.GetScalars() is not None:
        dst_data.SetActiveScalars(src_data.GetScalars().GetName())
    if src_data.GetVectors() is not None:
        dst_data.SetActiveVectors(src_data.GetVectors().GetName())
    if src_data.GetNormals() is not None:
        dst_data.SetActiveNormals(src_data.GetNormals().GetName())
    if src_data.GetTCoords() is not None:
        dst_data.SetActiveTCoords(src_data.GetTCoords().GetName())
    if src_data.GetTensors() is not None:
        dst_data.SetActiveTensors(src_data.GetTensors().GetName())


def initialize_accumulators(data_obj):
    accumulators = {}
    meta = {}

    n_arrays = data_obj.GetNumberOfArrays()
    for i in range(n_arrays):
        arr = data_obj.GetArray(i)
        if arr is None:
            continue

        name = arr.GetName()
        if name is None:
            continue

        arr_np = get_array_numpy(arr).astype(np.float64)
        accumulators[name] = np.zeros_like(arr_np, dtype=np.float64)
        meta[name] = {
            "ncomp": arr.GetNumberOfComponents(),
            "ntuple": arr.GetNumberOfTuples(),
        }

    return accumulators, meta


def add_to_accumulators(data_obj, accumulators, meta, label, fname):
    for name in accumulators:
        arr = data_obj.GetArray(name)
        if arr is None:
            raise ValueError(f"{label} array '{name}' missing in file: {fname}")

        if arr.GetNumberOfComponents() != meta[name]["ncomp"]:
            raise ValueError(
                f"{label} array '{name}' component mismatch in file: {fname}"
            )

        if arr.GetNumberOfTuples() != meta[name]["ntuple"]:
            raise ValueError(
                f"{label} array '{name}' tuple count mismatch in file: {fname}"
            )

        arr_np = get_array_numpy(arr).astype(np.float64)
        accumulators[name] += arr_np


def build_output_data(accumulators, n_files, template_data):
    out_data = template_data.NewInstance()

    for name, summed_arr in accumulators.items():
        avg_arr = summed_arr / n_files

        if avg_arr.shape[1] == 1:
            vtk_arr = numpy_to_vtk(avg_arr[:, 0], deep=True)
        else:
            vtk_arr = numpy_to_vtk(avg_arr, deep=True)

        vtk_arr.SetName(name)
        out_data.AddArray(vtk_arr)

    copy_active_attributes(template_data, out_data)
    return out_data


def main():
    file_paths = []
    for idx in range(START_INDEX, END_INDEX + 1, INDEX_STEP):
        fname = make_filename(idx)
        path = os.path.join(INPUT_DIR, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")
        file_paths.append(path)

    if len(file_paths) == 0:
        raise ValueError("No VTU files found in the requested range.")

    if VERBOSE:
        print("Averaging these files:")
        print(f"  start index : {START_INDEX}")
        print(f"  end index   : {END_INDEX}")
        print(f"  step        : {INDEX_STEP}")
        print(f"  file count  : {len(file_paths)}")
        print()

    # Read first file as template
    first_grid = read_vtu(file_paths[0])
    n_points_ref = first_grid.GetNumberOfPoints()
    n_cells_ref = first_grid.GetNumberOfCells()

    if VERBOSE:
        print(f"Template file: {file_paths[0]}")
        print(f"  points = {n_points_ref}")
        print(f"  cells  = {n_cells_ref}")
        print()

    # Reference point coordinates
    ref_points_np = None
    if CHECK_POINT_COORDS:
        ref_points_np = vtk_to_numpy(first_grid.GetPoints().GetData()).copy()

    # Initialize accumulators from first file
    point_accums, point_meta = initialize_accumulators(first_grid.GetPointData())
    cell_accums, cell_meta = initialize_accumulators(first_grid.GetCellData())

    if VERBOSE:
        print("Point-data arrays found:")
        for name in point_accums:
            print(f"  {name}")
        print()

        print("Cell-data arrays found:")
        for name in cell_accums:
            print(f"  {name}")
        print()

    # Add first file
    add_to_accumulators(first_grid.GetPointData(), point_accums, point_meta, "Point-data", file_paths[0])
    add_to_accumulators(first_grid.GetCellData(), cell_accums, cell_meta, "Cell-data", file_paths[0])

    # Process remaining files
    for i, path in enumerate(file_paths[1:], start=2):
        if VERBOSE:
            print(f"[{i}/{len(file_paths)}] Reading {path}")

        grid = read_vtu(path)

        if grid.GetNumberOfPoints() != n_points_ref:
            raise ValueError(f"Point count mismatch in file: {path}")

        if CHECK_NUM_CELLS and grid.GetNumberOfCells() != n_cells_ref:
            raise ValueError(f"Cell count mismatch in file: {path}")

        if CHECK_POINT_COORDS:
            pts_np = vtk_to_numpy(grid.GetPoints().GetData())
            if pts_np.shape != ref_points_np.shape or not np.allclose(pts_np, ref_points_np):
                raise ValueError(f"Point coordinates differ in file: {path}")

        add_to_accumulators(grid.GetPointData(), point_accums, point_meta, "Point-data", path)
        add_to_accumulators(grid.GetCellData(), cell_accums, cell_meta, "Cell-data", path)

    # Build output grid from template geometry/topology
    output_grid = vtk.vtkUnstructuredGrid()
    output_grid.DeepCopy(first_grid)

    # Replace point-data arrays with averages
    output_grid.GetPointData().Initialize()
    averaged_point_data = build_output_data(point_accums, len(file_paths), first_grid.GetPointData())
    for i in range(averaged_point_data.GetNumberOfArrays()):
        output_grid.GetPointData().AddArray(averaged_point_data.GetArray(i))
    copy_active_attributes(first_grid.GetPointData(), output_grid.GetPointData())

    # Replace cell-data arrays with averages
    output_grid.GetCellData().Initialize()
    averaged_cell_data = build_output_data(cell_accums, len(file_paths), first_grid.GetCellData())
    for i in range(averaged_cell_data.GetNumberOfArrays()):
        output_grid.GetCellData().AddArray(averaged_cell_data.GetArray(i))
    copy_active_attributes(first_grid.GetCellData(), output_grid.GetCellData())

    # Write output
    output_path = os.path.join(INPUT_DIR, OUTPUT_FILENAME)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(output_grid)

    ok = writer.Write()
    if ok != 1:
        raise RuntimeError(f"Failed to write output file: {output_path}")

    print()
    print(f"[OK] Wrote averaged VTU to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()