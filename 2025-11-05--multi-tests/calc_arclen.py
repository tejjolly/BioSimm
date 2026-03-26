import pandas as pd
import numpy as np

csv_path = "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests/g13_r24/concentration.csv"
df = pd.read_csv(csv_path)

# take one time slice to get the spatial order
t0 = df["Time"].min()
line = df[df["Time"] == t0].reset_index(drop=True)

x = line["Points:0"].to_numpy()
y = line["Points:1"].to_numpy()
z = line["Points:2"].to_numpy()

arc = np.zeros(len(line))
for i in range(1, len(line)):
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    dz = z[i] - z[i-1]
    arc[i] = arc[i-1] + np.sqrt(dx*dx + dy*dy + dz*dz)

# add to dataframe
line["ArcLength"] = arc

# DROP the last point (the weird one)
line = line.iloc[:-1]

# save just the arclengths
line[["ArcLength"]].to_csv(
    "/Volumes/biosimm-Tej-Jolly/2025-11-05--multi_tests/path_LCA_1_arclen.csv",
    index=False
)