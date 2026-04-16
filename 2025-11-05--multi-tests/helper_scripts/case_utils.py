import os
import re


def suffix_to_rmicro(suffix: str) -> float:
    if not suffix:
        return float("nan")
    match = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", suffix)
    if match is None:
        return float("nan")
    return float(match.group(1)) / 100.0


def build_case_base(geom: str, run_suffix: str = "") -> str:
    case = f"g{geom}"
    if run_suffix:
        case += f"_r{run_suffix}"
    return case


def build_case_name(geom: str, run_suffix: str = "", dye: str = "") -> str:
    case = build_case_base(geom, run_suffix)
    if dye:
        case += f"_d{dye}"
    return case


def expand_case_specs(
    run_geometries,
    run_suffixes,
    dyes,
    default_child_folder="96-procs",
    case_dir_overrides=None,
    child_folder_overrides=None,
):
    case_dir_overrides = case_dir_overrides or {}
    child_folder_overrides = child_folder_overrides or {}
    case_specs = []

    for geom in run_geometries:
        suffixes = run_suffixes.get(geom)
        if suffixes is None:
            raise KeyError(f"No run suffixes configured for geometry {geom}")

        for suffix in suffixes:
            for dye in dyes:
                case_name = build_case_name(geom, suffix, dye)
                case_base = build_case_base(geom, suffix)
                case_dir = case_dir_overrides.get(
                    case_name,
                    case_dir_overrides.get(case_base, case_name),
                )
                child_folder = child_folder_overrides.get(
                    case_name,
                    child_folder_overrides.get(case_base, default_child_folder),
                )
                case_specs.append({
                    "geom": geom,
                    "run_suffix": suffix,
                    "dye": dye,
                    "case_dir": case_dir,
                    "child_folder": child_folder,
                })

    return case_specs


def resolve_case_dir(root_dir, case_dir):
    if os.path.isabs(case_dir):
        return case_dir
    return os.path.join(root_dir, case_dir)


def results_dir_from_case_spec(root_dir, case_spec):
    case_dir = resolve_case_dir(root_dir, case_spec["case_dir"])
    return os.path.join(case_dir, case_spec.get("child_folder", "96-procs"))


def tag_output_root_from_case_spec(root_dir, case_spec, tag_output_folder="TAG"):
    case_dir = resolve_case_dir(root_dir, case_spec["case_dir"])
    return os.path.join(case_dir, tag_output_folder)


def add_output_paths_to_case_specs(
    case_specs,
    root_dir,
    tag_output_folder="TAG",
    concentration_subdir="concentrations",
):
    enriched_specs = []
    for case_spec in case_specs:
        spec = dict(case_spec)
        tag_dir = tag_output_root_from_case_spec(root_dir, spec, tag_output_folder)
        spec["results_dir"] = results_dir_from_case_spec(root_dir, spec)
        spec["tag_dir"] = tag_dir
        spec["concentration_dir"] = os.path.join(tag_dir, concentration_subdir)
        enriched_specs.append(spec)
    return enriched_specs


def concentration_csv_path(concentration_dir, geom, run_suffix="", dye="", full_series=False):
    filename = "concentration_timeseries.csv" if full_series else "concentration.csv"
    return os.path.join(concentration_dir, filename)


def resolve_existing_path(path_config):
    if isinstance(path_config, (list, tuple)):
        for path in path_config:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find any configured path in {path_config}")
    return path_config
