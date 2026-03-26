#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS (edit here)
# =========================

BASE_DIR = Path("/Users/tejjolly/Documents/BioSimm/Simulations/2025-11-15--zeroD/")
SUBFOLDER = "g13_r24/"        # <- change this per test
ROOT = BASE_DIR / SUBFOLDER

REFLECT = False   # True → original on bottom; False → both on +y

RANGES = [
    {"new": (1, 87), "orig": (2200, 3420)},
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
# STYLING
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
COLOR_NEW  = CMAP(0.90)
COLOR_ORIG = CMAP(0.65)
GREY = (0.4, 0.4, 0.4)  # for % diff and secondary axis
LW_MAIN, LW_AUX, MS = 2.0, 1.6, 5.0

# Pa → mmHg
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

def _label_series(ax, x, y, text, color, dy=0.0, dx=0.0):
    if len(x) == 0:
        return
    xl = x[-1]; yl = y[-1]
    ax.text(xl + dx, yl + dy, text, color=color, fontsize=14, va='center', ha='left')

def _percent_diff(new_arr, orig_arr):
    n = min(len(new_arr), len(orig_arr))
    if n == 0:
        return np.array([])
    a = np.asarray(new_arr[:n], dtype=float)
    b = np.asarray(orig_arr[:n], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = 100.0 * (a - b) / b
    pct[~np.isfinite(pct)] = np.nan  # handle div-by-zero and inf
    return pct

def plot_indexed(
    y_new,
    y_orig,
    ylabel,
    png_path: Path,
    svg_path: Path,
    new_label,
    orig_label,
    reflect: bool = True,
    xlim=None,
    ylim=None,
):
    """
    Primary axis: new/orig series (reflected or not).
    Secondary axis: percent difference = 100*(new - orig)/orig (grey, dotted).
    """
    if len(y_new) == 0 and len(y_orig) == 0:
        print("[plot] nothing to plot")
        return

    x_new  = np.arange(len(y_new))
    x_orig = np.arange(len(y_orig))

    # Percent difference on overlap
    pct = _percent_diff(y_new, y_orig)
    x_pct = np.arange(len(pct))

    fig, ax = plt.subplots(figsize=(9, 4))

    # --- primary axis (series) ---
    if reflect:
        if len(y_new):
            ax.plot(x_new, y_new, color=COLOR_NEW, lw=LW_MAIN)
        if len(y_orig):
            y_orig_plot = -np.abs(y_orig)
            ax.plot(x_orig, y_orig_plot, color=COLOR_ORIG, lw=LW_AUX)

        if len(y_new):
            _label_series(ax, x_new, y_new, new_label, COLOR_NEW, dy=0, dx=0)
        if len(y_orig):
            _label_series(ax, x_orig, -np.abs(y_orig), orig_label, COLOR_ORIG, dy=0, dx=0)
    else:
        if len(y_new):
            ax.plot(x_new, np.abs(y_new), color=COLOR_NEW, lw=LW_MAIN)
            _label_series(ax, x_new, np.abs(y_new), new_label, COLOR_NEW, dy=0, dx=0)
        if len(y_orig):
            ax.plot(x_orig, np.abs(y_orig), color=COLOR_ORIG, lw=LW_AUX)
            _label_series(ax, x_orig, np.abs(y_orig), orig_label, COLOR_ORIG, dy=0, dx=0)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # --- secondary axis (percent diff) ---
    ax2 = ax.twinx()
    if len(pct):
        ax2.plot(x_pct, pct, linestyle=':', color=GREY, lw=LW_AUX)
    ax2.set_ylabel("Percent diff [%]", color=GREY)
    ax2.tick_params(axis='y', colors=GREY)
    ax2.spines['right'].set_color(GREY)
    ax2.grid(False)
    ax2.axhline(0.0, color='k', linestyle='--', lw=0.8)
    ax2.set_ylim([-10, 10])

    fig.tight_layout()
    fig.savefig(png_path, dpi=600, transparent=True, bbox_inches='tight')
    fig.savefig(svg_path,           transparent=True, bbox_inches='tight')
    plt.show()
    plt.close(fig)

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

        # CSV padding + %diff columns
        n_max = max(len(y_new_flow), len(y_orig_flow), len(y_new_press_mmHg), len(y_orig_press_mmHg))
        def pad(v, N):
            out = np.full(N, np.nan, dtype=float); out[:len(v)] = v; return out

        flow_pct  = _percent_diff(y_new_flow, y_orig_flow)
        press_pct = _percent_diff(y_new_press_mmHg, y_orig_press_mmHg)

        csv_df = pd.DataFrame({
            "index":             np.arange(n_max),
            "new_flow":          pad(y_new_flow,  n_max),
            "orig_flow":         pad(y_orig_flow, n_max),
            "pct_diff_flow_%":   pad(flow_pct,    n_max),
            "new_press_mmHg":    pad(y_new_press_mmHg,  n_max),
            "orig_press_mmHg":   pad(y_orig_press_mmHg, n_max),
            "pct_diff_press_%":  pad(press_pct,   n_max),
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