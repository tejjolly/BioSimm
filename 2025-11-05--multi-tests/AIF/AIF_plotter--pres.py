import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================
# 0. USER OPTIONS
# ============================

SHOW_FOURIER = False        # <- switch: True to plot Fourier approximation, False to hide
FOURIER_TERMS_PLOT = 18    # <- number of Fourier terms for plotting only

FOURIER_NUMBER_HEADER = 18 # <- stays in the file header, independent of plotting

plot_flow = False


# ============================
# 1. RAW DATA (from your tables)
# ============================

# Bae et. al data
bae_x = np.array([
    0.905172414,
    2.887931034,
    5.0,
    8.793103448,
    11.72413793,
    14.52586207,
    18.40517241,
    21.93965517,
    25.43103448,
    28.75,
    32.24137931
])

bae_y = np.array([
    23.4939759,
    29.51807229,
    29.51807229,
    28.01204819,
    65.6626506,
    112.3493976,
    205.7228916,
    276.5060241,
    299.0963855,
    291.5662651,
    288.5542169
])

# Noelia data
noe_x = np.array([
    0.0,
    1.77,
    3.934,
    5.967,
    7.738,
    9.902,
    11.869,
    13.77,
    15.934,
    17.836
])

noe_y = np.array([
    44.521,
    47.945,
    23.288,
    52.055,
    55.479,
    104.11,
    156.164,
    134.932,
    92.466,
    76.712
])

# ===========================================
# 2. PARAMETERS YOU CAN TWEAK (start / end)
# ===========================================
# Bae et. al over 0.86 s
BAE_START = 0.0 + 0.86
BAE_END   = 0.86 + 0.86   # duration 0.86 s

# Noelia over 0.43 s
NOE_START = 0.0 + 0.86
NOE_END   = 0.43 + 0.86   # duration 0.43 s


# =============================================
# 3. HELPER FUNCTIONS: rescale time + normalize
# =============================================
def read_flow_file_for_plot(filename):
    """
    Read a SimV-style flow file:
    <N> <FourierNumber>
    t0 q0
    t1 q1
    ...
    Returns (t, q) as 1D numpy arrays.
    """
    with open(filename, "r") as f:
        header = f.readline().strip().split()
        # nrows = int(header[0])  # not actually needed, but could be used for checks
        data = np.loadtxt(f)

    t = data[:, 0]
    q = data[:, 1]
    return t, q

def tile_flow_for_plot(t, q, t_max, max_cycles=100):
    """
    Repeat the flow waveform in time up to t_max (or up to max_cycles),
    so that it looks like it is cycling indefinitely.

    t, q: 1D arrays from a single-cycle flow file
    t_max: maximum time to keep (e.g. 3.44)
    max_cycles: hard cap on how many repeats we generate
    """
    t = np.asarray(t)
    q = np.asarray(q)

    if t.size < 2:
        return t, q

    t0 = t[0]
    T = t[-1] - t0  # period

    if T <= 0:
        return t, q

    # shift base cycle to start at 0
    t_base = t - t0

    # how many cycles are needed to cover t_max?
    needed_cycles = int(np.ceil(t_max / T)) + 1
    n_cycles = min(max_cycles, max(1, needed_cycles))

    tiled_t = []
    tiled_q = []

    for k in range(n_cycles):
        tiled_t.append(t_base + k * T)
        tiled_q.append(q)

    tiled_t = np.concatenate(tiled_t)
    tiled_q = np.concatenate(tiled_q)

    # keep only up to t_max
    mask = tiled_t <= t_max
    return tiled_t[mask], tiled_q[mask]


def rescale_time(x, t_start, t_end):
    """
    Linearly map the original x-range [min(x), max(x)]
    to the new range [t_start, t_end].
    """
    x_min = x.min()
    x_max = x.max()
    return t_start + (x - x_min) * (t_end - t_start) / (x_max - x_min)


def normalize_area(x, y):
    """
    Normalize y so that the integral (area under the curve) over x is 1.
    Uses the trapezoidal rule.
    """
    area = np.trapezoid(y, x)
    return y / area


def prepend_zero_head(t, c, dt_before_first=0.01):
    """
    If the series does not start at t = 0, prepend zeros at:
      - t = 0.0
      - halfway between 0 and the first non-zero time
      - (t_first - dt_before_first), if > 0 and > midpoint

    All with c = 0.
    """
    t = np.asarray(t)
    c = np.asarray(c)

    if np.isclose(t[0], 0.0):
        return t, c

    t_first = t[0]
    times = [0.0]

    # midpoint between 0 and first non-zero
    t_mid = 0.5 * t_first
    if t_mid > 0:
        times.append(t_mid)

    # 0.01 s before first non-zero
    t_before = t_first - dt_before_first
    if t_before > 0 and t_before > t_mid:
        times.append(t_before)

    times = np.array(sorted(times))
    zeros = np.zeros_like(times)

    t_new = np.concatenate([times, t])
    c_new = np.concatenate([zeros, c])
    return t_new, c_new


def append_zero_tail(t, c, dt_after_last=0.01, t_final=3.44):
    """
    Append zeros at:
      - t_last + dt_after_last
      - halfway between (t_last + dt_after_last) and t_final
      - t_final

    All with c = 0.
    """
    t = np.asarray(t)
    c = np.asarray(c)

    t_last = t[-1]
    t1 = t_last + dt_after_last

    times = []
    if t1 < t_final:
        t_mid = 0.5 * (t1 + t_final)
        times = [t1, t_mid, t_final]
    else:
        # degenerate case: just ensure at least t1 and t_final
        if t1 > t_final:
            times = [t1, t_final]
        else:
            times = [t_final]

    times = np.array(times)
    zeros = np.zeros_like(times)

    t_ext = np.concatenate([t, times])
    c_ext = np.concatenate([c, zeros])
    return t_ext, c_ext


def write_flow_file(filename, t, c, fourier_number=14):
    """
    Write flow file with format:
    <N> <FourierNumber>
    t0 c0
    t1 c1
    ...
    """
    nrows = len(t)
    with open(filename, "w") as f:
        f.write(f"{nrows} {fourier_number}\n")
        for ti, ci in zip(t, c):
            f.write(f"{ti:.2f} {ci:.3f}\n")

    # Also print to terminal
    print(f"\n{filename}:")
    print(f"{nrows} {fourier_number}")
    for ti, ci in zip(t, c):
        print(f"{ti:.2f} {ci:.3f}")


# ======================================
# 4. FOURIER SERIES APPROXIMATION (PLOT)
# ======================================

def fourier_series_fit(t, f, n_terms, n_eval=400):
    """
    Fit a periodic Fourier series with n_terms (cos/sin pairs) on [t[0], t[-1]]
    and return a dense approximation (t_eval, f_eval) for plotting.
    """
    t = np.asarray(t)
    f = np.asarray(f)
    t0 = t[0]
    T = t[-1] - t0

    if T <= 0:
        raise ValueError("Time range T must be positive for Fourier fit.")

    # Map to theta in [0, 2π]
    theta = 2.0 * np.pi * (t - t0) / T

    # a0, an, bn via trapezoidal integration in theta
    a0 = (1 / np.pi) * np.trapezoid(f, theta)
    a = np.zeros(n_terms + 1)
    b = np.zeros(n_terms + 1)
    a[0] = a0

    for n in range(1, n_terms + 1):
        a[n] = (1 / np.pi) * np.trapezoid(f * np.cos(n * theta), theta)
        b[n] = (1 / np.pi) * np.trapezoid(f * np.sin(n * theta), theta)

    # Evaluate series on a dense grid
    theta_eval = np.linspace(0, 2.0 * np.pi, n_eval)
    f_eval = np.full_like(theta_eval, a[0] / 2.0)

    for n in range(1, n_terms + 1):
        f_eval += a[n] * np.cos(n * theta_eval) + b[n] * np.sin(n * theta_eval)

    # Map back to time domain
    t_eval = t0 + (theta_eval / (2.0 * np.pi)) * T
    return t_eval, f_eval


# ===========================
# 5. RESCALE + NORMALIZE DATA
# ===========================

# Rescale x to desired time windows
bae_t = rescale_time(bae_x, BAE_START, BAE_END)
noe_t = rescale_time(noe_x, NOE_START, NOE_END)

# Normalize the areas to 1 (before adding zero heads/tails)
bae_y_norm = normalize_area(bae_t, bae_y)
noe_y_norm = normalize_area(noe_t, noe_y)

# Quick check: areas should both be ~1
print("Area (Bae et. al):", np.trapezoid(bae_y_norm, bae_t))
print("Area (Noelia):    ", np.trapezoid(noe_y_norm, noe_t))

# ===========================
# 6. PREPEND ZERO HEAD + APPEND ZERO TAIL
# ===========================

# Head zeros (0, midpoint, 0.01 before first non-zero)
bae_t_head, bae_y_head = prepend_zero_head(bae_t, bae_y_norm,
                                           dt_before_first=0.01)
noe_t_head, noe_y_head = prepend_zero_head(noe_t, noe_y_norm,
                                           dt_before_first=0.01)

# Tail zeros (0.01 after last non-zero, midpoint to tfinal, tfinal)
bae_t_out, bae_y_out = append_zero_tail(bae_t_head, bae_y_head,
                                        dt_after_last=0.01,
                                        t_final=3.44)
noe_t_out, noe_y_out = append_zero_tail(noe_t_head, noe_y_head,
                                        dt_after_last=0.01,
                                        t_final=3.44)

# ===========================
# 7. WRITE .flw FILES
# ===========================

write_flow_file("dye_B.flw", bae_t_out, bae_y_out,
                fourier_number=FOURIER_NUMBER_HEADER)
write_flow_file("dye_N.flw", noe_t_out, noe_y_out,
                fourier_number=FOURIER_NUMBER_HEADER)

# ===========================
# 8. PLOT BOTH ON SAME FIGURE + INLET FLOW
# ===========================

# fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(3, 2))

# For plotting, exclude the tail zeros (last 2 or 3 points depending on t_final logic).
# Easiest: keep everything up to the last non-zero index.
def last_nonzero_index(arr):
    nz = np.nonzero(arr)[0]
    return nz[-1] if nz.size > 0 else len(arr) - 1

bae_last_nz = last_nonzero_index(bae_y_out)
noe_last_nz = last_nonzero_index(noe_y_out)

cmap = plt.colormaps["BuPu"]
color_Noe = 'black'
color_Bae = 'black'
# color_Noe = 'blue'
# color_Bae = 'lightblue'
# AIF curves (your usual styling)
ax.plot(
    bae_t_out[2:bae_last_nz+1],
    bae_y_out[2:bae_last_nz+1],
    # marker="o",
    color=color_Bae,
    linestyle="-",
    linewidth=2
)
# ax.plot(
#     noe_t_out[2:noe_last_nz+1],
#     noe_y_out[2:noe_last_nz+1],
#     # marker="s",
#     color=color_Noe,
#     linestyle="-",
#     linewidth=2
# )

# Fourier approximations (for plotting only)
if SHOW_FOURIER:
    bae_tf, bae_cf = fourier_series_fit(
        bae_t_out, bae_y_out, n_terms=FOURIER_TERMS_PLOT
    )
    noe_tf, noe_cf = fourier_series_fit(
        noe_t_out, noe_y_out, n_terms=FOURIER_TERMS_PLOT
    )

    ax.plot(bae_tf, bae_cf, color=color_Bae, linestyle=":", linewidth=1.0)
    ax.plot(noe_tf, noe_cf, color=color_Noe, linestyle="-.", linewidth=1.0)

# Main AIF axis labels
ax.set_xlabel("Time [s]")
ax.set_ylabel("Dye input")

# X-limits and ticks at 0, 0.43, 0.86, ...
t_final = 1.72
ax.set_xlim(0.86, t_final)
ax.set_ylim(top=4)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.43))

# Put labels near the peaks of each original curve, no legend
bae_peak_idx = np.argmax(bae_y_out)
noe_peak_idx = np.argmax(noe_y_out)

# ax.text(
#     bae_t_out[bae_peak_idx],
#     bae_y_out[bae_peak_idx],
#     "Flat dye input",
#     va="bottom",
#     color=color_Bae,
# )
# ax.text(
#     noe_t_out[noe_peak_idx+1],
#     noe_y_out[noe_peak_idx+1],
#     "Sharp dye input",
#     va="bottom",
#     color=color_Noe,
# )

# Only relevant spines for main axis (bottom & left)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ==========================================
# Secondary axis: inlet_sv.flow, tiled in time
# ==========================================
if plot_flow:
    flow_t_raw, flow_q_raw = read_flow_file_for_plot("inlet_sv.flow")
    flow_t, flow_q = tile_flow_for_plot(flow_t_raw, flow_q_raw, t_max=t_final, max_cycles=100)
    color_flow="grey"
    ax2 = ax.twinx()
    ax2.plot(flow_t, -flow_q, color=color_flow, linewidth=1.0, linestyle='--')

    ax2.set_ylabel("Inlet flow rate", color=color_flow)
    ax2.tick_params(axis="y", colors=color_flow)
    ax2.spines["right"].set_color(color_flow)
    ax2.spines["top"].set_visible(False)  # keep top spine off
    ax2.set_ylim(top=175)
    # ax2.vlines([0.86, 0.86*2, 0.86*3],0, 200,linestyle=':', color=color_flow)

plt.tight_layout()
plt.savefig("AIF.svg", transparent=True, dpi=600)
plt.show()