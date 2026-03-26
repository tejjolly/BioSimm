import os
import pandas as pd
import numpy as np

base = "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests"

# define geometry + runs
run_geometry = "37"   # change here
run_suffixes = ["24", "62", "81", "100"]   # change/add as needed

cases = [f"g{run_geometry}_r{sfx}" for sfx in run_suffixes]

# load arc length once
arc_path = os.path.join(base, "path_LCA_1_arclen.csv")
arc_df = pd.read_csv(arc_path)
arc = arc_df["ArcLength"].to_numpy()

# indices for the linear region
i0, i1 = 25, 90

TAG = {}        # raw slopes (conc per length)
TAG_norm = {}   # slopes normalized by inlet peak
tmax_map = {}
cinlet_map = {}

for case in cases:
    csv_path = os.path.join(base, case, "concentration.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found, skipping {case}")
        continue

    df = pd.read_csv(csv_path)

    # 1) inlet rows (first row per time)
    inlet_df = df.groupby("Time").head(1)

    # 2) time where inlet concentration is max
    idx_max = inlet_df["Concentration"].idxmax()
    t_max = inlet_df.loc[idx_max, "Time"]
    c_inlet = inlet_df.loc[idx_max, "Concentration"]

    tmax_map[case] = t_max
    cinlet_map[case] = c_inlet

    # 3) full line at that time
    prof = df[df["Time"] == t_max].reset_index(drop=True)

    # 4) grab arc length for the same segment
    # (trim in case arc is longer than this profile)
    x = arc[i0:i1 + 1]
    y = prof.loc[i0:i1, "Concentration"].to_numpy()

    # 5) fit concentration = m * arclen + b
    m, b = np.polyfit(x, y, 1)
    TAG[case] = m

    # 6) normalize by inlet peak for that run
    TAG_norm[case] = m / c_inlet if c_inlet != 0 else np.nan

# print results
for case in cases:
    if case not in TAG:
        continue
    print(
        f"{case}: "
        f"t_max={tmax_map[case]:.5f}, "
        f"C_inlet={cinlet_map[case]:.6f}, "
        f"TAG_slope={TAG[case]:.6f}, "
        f"TAG_norm={TAG_norm[case]:.6e}"
    )