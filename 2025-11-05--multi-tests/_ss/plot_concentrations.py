import os
import pandas as pd
import matplotlib.pyplot as plt

base = "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests"

cases = [
    "g13_r24",
    "g13_r62",
    "g13_r81",
    "g13_r100_v3",
]

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
plt.figure(figsize=(8, 5))

for case, prof in profiles.items():
    x = prof.index
    y = prof["Concentration"]
    plt.plot(x, y, label=case)

plt.xlabel("Point index along centerline")
plt.ylabel("Concentration (at time of max inlet)")
plt.title("Centerline concentration snapshot")
plt.legend()
plt.tight_layout()
plt.show()
