import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base = "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests"

# geometry + runs
run_suffixes = ["24", "62", "81", "100"]
run_geometry = "37"

cases = [f"g{run_geometry}_r{sfx}" for sfx in run_suffixes]

# load arc length once
arc_path = os.path.join(base, "path_LCA_1_arclen.csv")
arc_df = pd.read_csv(arc_path)
arc_len_all = arc_df["ArcLength"].to_numpy()

profiles = {}

for case in cases:
    csv_path = os.path.join(base, case, "concentration.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found, skipping")
        continue

    print(f"[INFO] Reading {csv_path}")
    df = pd.read_csv(csv_path)

    # 1) inlet = first row for each time
    inlet_df = df.groupby("Time").head(1)

    # 2) time where inlet concentration is max
    idx_max = inlet_df["Concentration"].idxmax()
    t_max = inlet_df.loc[idx_max, "Time"]
    c_max = inlet_df.loc[idx_max, "Concentration"]
    print(f"[INFO] {case}: max inlet concentration = {c_max} at Time = {t_max}")

    # 3) whole line at that time
    prof = df[df["Time"] == t_max].reset_index(drop=True)
    profiles[case] = prof

# 4) plot
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()       # secondary y (velocity)
ax_top = ax1.twiny()    # secondary x (point index, on top)

# we'll use the arc length from the first profile's length to set top ticks
first_prof = next(iter(profiles.values()))
npts = len(first_prof)
arc_len = arc_len_all[:npts]  # trim to match

for case, prof in profiles.items():
    # bottom x-axis: arc length
    x_arc = arc_len[:len(prof)]
    conc = prof["Concentration"][:-1]

    # plot concentration vs arc length
    line = ax1.plot(x_arc, conc, label=case)[0]
    color = line.get_color()

    # velocity magnitude from 3 components
    vel_mag = np.sqrt(
        prof["Velocity:0"]**2 +
        prof["Velocity:1"]**2 +
        prof["Velocity:2"]**2
    )[0:-1]

    # plot velocity on secondary y-axis, dotted
    ax2.plot(x_arc, vel_mag, linestyle=":", color=color)

# set bottom / y labels
ax1.set_xlabel("Arc length along centerline")
ax1.set_ylabel("Concentration (at time of max inlet)")
ax2.set_ylabel("Velocity magnitude (same time)")
ax1.set_title("Centerline concentration + velocity snapshot")

# now configure the top x-axis to show point index
# we map the bottom axis limits to point indices
ax_top.set_xlim(ax1.get_xlim())

# choose some tick positions in arc length, then label them by index
# simplest: use every 20th point
tick_idx = np.arange(0, npts, 20)
tick_pos = arc_len[tick_idx]
ax_top.set_xticks(tick_pos)
ax_top.set_xticklabels([str(i) for i in tick_idx])
ax_top.set_xlabel("Point index")

ax1.legend(loc="upper right")
fig.tight_layout()
plt.show()
