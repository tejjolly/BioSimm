### Linetype and marker legend:
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import numpy as np


# ── Marker legend entries for stenosis (triangle≈45, square≈60) ──
circle   = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                         markersize=10, label='0% stenosis')
triangle = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                         markersize=10, label='45% stenosis')
square   = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                         markersize=10, label='60% stenosis')

# ── Line legend entries for lesion length ──
dotted = mlines.Line2D([], [], color='black', linestyle=':',  linewidth=2, label='1.2 cm')
dashed = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='2.5 cm')

# ── Vessel legend entries (LAD/LCX) ──
lad = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='LAD')
lcx = mlines.Line2D([], [], color='gray',  linestyle='-', linewidth=2, label='LCX', alpha=0.6)

# ── Create figure with only the legend ──
fig, ax = plt.subplots(figsize=(3.75, 0.75), dpi=400)
ax.axis('off')

handles = [circle, triangle, square, dotted, dashed, lad, lcx]

# nice 3 columns × 2 rows layout; tweak as you like
legend = ax.legend(handles=handles, loc='center', frameon=False,
                   ncol=3, handlelength=2.5, handletextpad=0.4, columnspacing=1.2, fontsize=11)

fig.savefig("marker_linetype_vessel_legend.png",
            dpi=400, transparent=True, bbox_inches='tight')
fig.savefig("marker_linetype_vessel_legend.svg",
            transparent=True, bbox_inches='tight')
plt.show()


# ================= DISTRIBUTION HISTOGRAM LEGEND =================
# ================= USER-TUNABLE SETTINGS =================
FIG_WIDTH   = 2.0      # inches
FIG_HEIGHT  = 2.2      # inches
FONT_SIZE   = 12
HANDLE_LENGTH = 2.5
HANDLE_TEXTPAD = 0.2
LABEL_SPACING  = 0.8   # vertical spacing between entries

# =========================================================
LEVELS = [
    (0.24, r"Base $R_{A\text{-}m}$"),
    (0.43, r"$R_{A\text{-}m}x2.7$"),
    (0.62, r"$R_{A\text{-}m}x4.0$"),
    (0.81, r"$R_{A\text{-}m}x6.3$")
]
hue_order = [lab for _, lab in (LEVELS)]

# Color palette, same as main plots
vals = np.r_[0.25, np.linspace(0.6, 1.0, len(hue_order) - 1)]
color_map = mpl.cm.BuPu(vals)
palette = {lab: col for lab, col in zip(hue_order, color_map)}

# We want Base R to appear at the **bottom**, so reverse for legend order:
legend_order = hue_order[::-1]

# Build handles
handles = [
    mlines.Line2D([], [], color=palette[label], linestyle='--', linewidth=18, label=label, alpha = 0.75)
    for label in legend_order
]

# Make the figure & no axes
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=400)
ax.axis('off')

legend = ax.legend(
    handles=handles,
    loc='center',
    frameon=False,
    ncol=1,
    fontsize=FONT_SIZE,
    handlelength=HANDLE_LENGTH,
    handletextpad=HANDLE_TEXTPAD,
    borderpad=0.0,
    labelspacing=LABEL_SPACING
)

fig.savefig("images/R_levels_legend.png", dpi=400, transparent=True, bbox_inches='tight')
fig.savefig("images/R_levels_legend.svg", transparent=True, bbox_inches='tight')
plt.show()
