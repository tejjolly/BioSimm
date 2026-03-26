#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS (edit here)
# =========================

# where you run stuff
BASE_DIR = Path("/Users/tejjolly/Documents/BioSimm/Simulations/2025-11-15--zeroD/")
SUBFOLDER = "g13_r24/"        # <- change this per test
ROOT = BASE_DIR / SUBFOLDER

# choose reflection
REFLECT = False   # True → original on bottom; False → both on +y

RANGES = [
    {"new": (1, 86), "orig": (2580, 3440)},
]

# -------------------------
# Paths under this subfolder
# -------------------------
NEW_DIR = ROOT / "96-procs"
ORIG_DIR = ROOT / "svSolver_results"

NEW_FLOW_FILE   = NEW_DIR / "B_NS_Velocity_flux.txt"
NEW_PRESS_FILE  = NEW_DIR / "B_NS_Pressure_average.txt"
ORIG_FLOW_FILE  = ORIG_DIR / "all_results-flows.txt"
ORIG_PRESS_FILE = ORIG_DIR / "all_results-pressure.txt"

IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# STYLING (BuPu aesthetic)
# =========================
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.dpi': 600,
})
CMAP = plt.get_cmap('BuPu')
COLOR_NEW  = CMAP(0.9)
COLOR_ORIG = CMAP(0.65)
COLOR_DIFF = CMAP(0.50)
LW_MAIN, LW_AUX, MS = 2.0, 1.6, 5.0

# correct Pa → mmHg
PA_TO_MMHG = 1.0 / 133.322

# =========================
# Helpers
# =========================
_num_re = re.compile(r'^[\+\-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][\+\-]?\d+)?$')

def _first_inlet_name(tokens):
    for t in tokens:
        if "inlet" in t.lower():
            return t
    raise ValueError("No inlet-like column found.")

def read_sv_boundary_auto_inlet(path: Path):
    steps, times, vals = [], [], []
    inlet_name = None
    with open(path, "r") as f:
        headers = None; col_idx = time_idx = step_idx = None
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("face areas:"):
                continue
            if headers is None and s.lower().startswith("step"):
                headers = s.split()
                inlet_name = _first_inlet_name(headers)
                col_idx  = headers.index(inlet_name)
                time_idx = headers.index("time")
                step_idx = headers.index("step")
                continue
            if headers is not None:
                toks = [t.replace('D','E').replace('d','E') for t in s.split()]
                if len(toks) <= col_idx:
                    continue
                if not (_num_re.match(toks[step_idx]) and _num_re.match(toks[time_idx]) and _num_re.match(toks[col_idx])):
                    continue
                steps.append(int(float(toks[step_idx])))
                times.append(float(toks[time_idx]))
                vals.append(float(toks[col_idx]))
    if not vals:
        raise ValueError(f"No data parsed from {path} for inlet column.")
    return inlet_name, np.array(steps, int), np.array(times, float), np.array(vals, float)

def read_original_table_auto_inlet(path: Path):
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "step" not in df:
        raise ValueError(f"'step' column not found in {path}. Columns: {list(df.columns)}")
    inlet_cols = [c for c in df.columns if "inlet" in c.lower()]
    if not inlet_cols:
        raise ValueError(f"No inlet-like column found in {path}. Columns: {list(df.columns)}")
    inlet_col = inlet_cols[0]
    return inlet_col, df["step"].to_numpy(int), df[inlet_col].to_numpy(float)

def slice_by_step(steps, values, step_min, step_max):
    m = (steps >= step_min) & (steps <= step_max)
    return values[m]

def _label_series(x, y, text, color, dy=0.0, dx=0.0):
    if len(x) == 0:
        return
    xl = x[-1]
    yl = y[-1]
    plt.text(
        xl + dx,
        yl + dy,
        text,
        color=color,
        fontsize=14,
        va='center',
        ha='left'
    )

def plot_indexed(
    y_new,
    y_orig,
    ylabel,
    png_path: Path,
    svg_path: Path,
    new_label,
    orig_label,
    diff_label,
    reflect: bool = True,
    xlim=None,
    ylim=None,
):
    """
    If reflect=True:
        new → +y (keep sign)
        orig → −abs(orig)   # always bottom
    If reflect=False:
        both → +y using abs()
    """
    if len(y_new) == 0 and len(y_orig) == 0:
        print("[plot] nothing to plot")
        return

    x_new  = np.arange(len(y_new))
    x_orig = np.arange(len(y_orig))

    n_overlap = min(len(y_new), len(y_orig))
    x_diff = np.arange(n_overlap)
    # diff is based on raw values (not the plotted ones)
    y_diff = y_new[:n_overlap] - y_orig[:n_overlap]

    plt.figure(figsize=(9, 4))

    if reflect:
        # new stays as-is
        if len(y_new):
            plt.plot(x_new, y_new, color=COLOR_NEW, lw=LW_MAIN)
        # original forced to bottom
        if len(y_orig):
            y_orig_plot = -np.abs(y_orig)
            plt.plot(x_orig, y_orig_plot, color=COLOR_ORIG, lw=LW_AUX)
        # diff around 0
        if n_overlap > 0:
            plt.plot(x_diff, y_diff, color=COLOR_DIFF, lw=LW_AUX, linestyle="--")

        # labels
        if len(y_new):
            _label_series(x_new, y_new, new_label, COLOR_NEW, dy=0, dx=0)
        if len(y_orig):
            _label_series(x_orig, -np.abs(y_orig), orig_label, COLOR_ORIG, dy=0, dx=0)
        if n_overlap > 0:
            _label_series(x_diff, y_diff, diff_label, COLOR_DIFF, dy=0, dx=0)

    else:
        # NO reflection: both positive
        if len(y_new):
            plt.plot(x_new, np.abs(y_new), color=COLOR_NEW, lw=LW_MAIN)
        if len(y_orig):
            plt.plot(x_orig, np.abs(y_orig), color=COLOR_ORIG, lw=LW_AUX)
        if n_overlap > 0:
            # diff still raw
            plt.plot(x_diff, y_diff, color=COLOR_DIFF, lw=LW_AUX, linestyle="--")

        # labels (place at positive end)
        if len(y_new):
            _label_series(x_new, np.abs(y_new), new_label, COLOR_NEW, dy=0, dx=0)
        if len(y_orig):
            _label_series(x_orig, np.abs(y_orig), orig_label, COLOR_ORIG, dy=0, dx=0)
        if n_overlap > 0:
            _label_series(x_diff, y_diff, diff_label, COLOR_DIFF, dy=0, dx=0)

    plt.axhline(0.0, color='k', lw=0.8)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(False)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(png_path, dpi=600, transparent=True, bbox_inches='tight')
    plt.savefig(svg_path,           transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()

# =========================
# Main
# =========================
def main():
    # Load (auto-detect inlet)
    in_name_flow_new, new_steps_f, new_times_f, new_flow = read_sv_boundary_auto_inlet(NEW_FLOW_FILE)
    in_name_press_new, new_steps_p, new_times_p, new_press = read_sv_boundary_auto_inlet(NEW_PRESS_FILE)
    in_name_flow_orig, orig_steps_f, orig_flow = read_original_table_auto_inlet(ORIG_FLOW_FILE)
    in_name_press_orig, orig_steps_p, orig_press = read_original_table_auto_inlet(ORIG_PRESS_FILE)

    print(f"[auto] NEW inlet (flow): {in_name_flow_new} | NEW inlet (press): {in_name_press_new}")
    print(f"[auto] ORIG inlet (flow): {in_name_flow_orig} | ORIG inlet (press): {in_name_press_orig}")

    for r in RANGES:
        nmin, nmax = r["new"]
        omin, omax = r["orig"]

        # Slice
        y_new_flow  = slice_by_step(new_steps_f,  new_flow,  nmin, nmax)
        y_orig_flow = slice_by_step(orig_steps_f, orig_flow, omin, omax)
        y_new_press  = slice_by_step(new_steps_p,  new_press,  nmin, nmax)
        y_orig_press = slice_by_step(orig_steps_p, orig_press, omin, omax)

        # convert pressure to mmHg
        y_new_press_mmHg  = y_new_press * PA_TO_MMHG
        y_orig_press_mmHg = y_orig_press * PA_TO_MMHG

        # CSV padding
        n_max = max(len(y_new_flow), len(y_orig_flow), len(y_new_press_mmHg), len(y_orig_press_mmHg))
        def pad(v, N):
            out = np.full(N, np.nan, dtype=float)
            out[:len(v)] = v
            return out

        flow_overlap = min(len(y_new_flow), len(y_orig_flow))
        press_overlap = min(len(y_new_press_mmHg), len(y_orig_press_mmHg))

        csv_df = pd.DataFrame({
            "index":            np.arange(n_max),
            "new_flow":         pad(y_new_flow,  n_max),
            "orig_flow":        pad(y_orig_flow, n_max),
            "diff_flow":        pad(y_new_flow[:flow_overlap] - y_orig_flow[:flow_overlap], n_max),
            "new_press_mmHg":   pad(y_new_press_mmHg,  n_max),
            "orig_press_mmHg":  pad(y_orig_press_mmHg, n_max),
            "diff_press_mmHg":  pad(y_new_press_mmHg[:press_overlap] - y_orig_press_mmHg[:press_overlap], n_max),
        })
        out_csv = ROOT / "inlet.csv"
        csv_df.to_csv(out_csv, index=False)

        flow_png  = IMAGES_DIR / "inlet_flow.png"
        flow_svg  = IMAGES_DIR / "inlet_flow.svg"
        press_png = IMAGES_DIR / "inlet_pressure.png"
        press_svg = IMAGES_DIR / "inlet_pressure.svg"

        # flow plot
        plot_indexed(
            y_new_flow,
            y_orig_flow,
            ylabel="Flow rate [cm3/s]",
            png_path=flow_png,
            svg_path=flow_svg,
            new_label="svMultiPhysics",
            orig_label="svSolver",
            diff_label="Difference",
            reflect=REFLECT,
            xlim=None,
            ylim=None,
        )

        # pressure plot
        plot_indexed(
            y_new_press_mmHg,
            y_orig_press_mmHg,
            ylabel="Pressure [mmHg]",
            png_path=press_png,
            svg_path=press_svg,
            new_label="svMultiPhysics",
            orig_label="svSolver",
            diff_label="Difference",
            reflect=REFLECT,
            xlim=None,
            ylim=None,
        )

        print(f"Wrote CSV: {out_csv}")
        print(f"Wrote figs: {flow_png}, {flow_svg}")
        print(f"Wrote figs: {press_png}, {press_svg}")

    print("Done.")

if __name__ == "__main__":
    main()
