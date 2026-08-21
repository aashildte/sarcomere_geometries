import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

DATA_DIR = "cell_geometries"
cells = list(range(0, 9))
conditions = ["1_nuclei", "2_nuclei"]

# ----------------------------
# storage
# ----------------------------
sarco_means = []
nuc_means = []

cell_lengths = []
cell_widths = []

nuc_lengths = []
nuc_widths = []


# ----------------------------
# geometry
# ----------------------------
def compute_cell_geometry(h5_path, n_bins=40):
    with h5py.File(h5_path, "r") as f:
        coords = f["mesh/coordinates"][:]

    x = coords[:, 0]
    y = coords[:, 1]

    bins = np.linspace(x.min(), x.max(), n_bins + 1)

    widths = []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(mask) < 5:
            continue
        y_slice = y[mask]
        widths.append(np.max(y_slice) - np.min(y_slice))

    length = x.max() - x.min()
    width = np.median(widths) if len(widths) > 0 else np.nan

    return length, width


# ----------------------------
# loop
# ----------------------------

for cell in cells:
    for cond in conditions:

        base = f"cell_{cell}_with_{cond}"

        # ---------------- SARCOMERE ----------------
        sarc_path = os.path.join(DATA_DIR, base + "_angles.npy")
        sarco_means.append(np.rad2deg(np.mean(np.abs(np.load(sarc_path)))))

        # ---------------- NUCLEUS ----------------
        nuc_path = os.path.join(DATA_DIR, base + "_nuclei_angles.npy")
        nuc = np.load(nuc_path, allow_pickle=True).item()

        angles = nuc["angles"] - np.pi
        angles = (angles + np.pi) % (2*np.pi) - np.pi

        nuc_means.append(np.rad2deg(np.mean(abs(angles))))
        nuc_widths.append(nuc["radi"][1])
        nuc_lengths.append(nuc["radi"][0])

        # ---------------- GEOMETRY ----------------
        h5_path = os.path.join(DATA_DIR, base + ".h5")

        with h5py.File(h5_path, "r") as f:
            coords = f["mesh/coordinates"][:]

        x = coords[:, 0]
        y = coords[:, 1]

        length = x.max() - x.min()

        bins = np.linspace(x.min(), x.max(), 40)
        widths = [
            np.max(y[(x >= bins[i]) & (x < bins[i+1])]) -
            np.min(y[(x >= bins[i]) & (x < bins[i+1])])
            for i in range(len(bins)-1)
            if np.sum((x >= bins[i]) & (x < bins[i+1])) > 5
        ]

        cell_lengths.append(length)
        cell_widths.append(np.median(widths))
# ----------------------------
# normalization
# ----------------------------
def norm(x):
    x = np.array(x)
    return x / np.mean(x)


sarco_means_n = norm(sarco_means)
nuc_means_n = norm(nuc_means)

cell_lengths_n = norm(cell_lengths)
cell_widths_n = norm(cell_widths)

nuc_lengths_n = norm(nuc_lengths)
nuc_widths_n = norm(nuc_widths)


# ----------------------------
# mean ± std helper (subtle)
# ----------------------------
def add_mean_std(ax, x, data):
    data = np.array(data)

    ax.errorbar(
        x,
        np.mean(data),
        yerr=np.std(data),
        fmt='o',
        color='0.25',
        ecolor='0.65',
        elinewidth=1,
        capsize=3,
        markersize=4,
        alpha=0.6,
        zorder=10
    )



# ----------------------------
# ----------------------------
# COLORS (pale nucleus + beige cell)
# ----------------------------
COLOR_SARCO = "#0B2A5A"   # deep navy for sarcomeres
COLOR_NUC = "#A7C4B6"
COLOR_CELL  = "#D3D3D3"   # light beige cell body


# ----------------------------
# compact jitter helper
# ----------------------------
def plot_jitter(ax, data_list, labels, colors, step=0.2, jitter=0.015):
    positions = np.arange(len(data_list)) * step

    for i, d in enumerate(data_list):

        x = np.random.normal(positions[i], jitter, len(d))

        ax.scatter(
            x,
            d,
            s=12,
            color=colors[i]
        )

        add_mean_std(ax, positions[i], d)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    return positions


# ----------------------------
# FIGURE
# ----------------------------
fig, (ax0, ax1, ax2) = plt.subplots(
    1, 3,
    figsize=(10, 3.5),
    gridspec_kw={"width_ratios": [0.6, 1.2, 1.2]}
)


# =========================================================
# PANEL 1 — ANGLES
# =========================================================
plot_jitter(
    ax0,
    [sarco_means, nuc_means],
    ["Sarcomeres", "Nuclei"],
    [COLOR_SARCO, COLOR_NUC],
    step=0.1,
)

ax0.set_xlim(-0.07, 0.17)

ax0.set_title("Angle distribution")
ax0.set_ylabel("Angle (degrees)")

# =========================================================
# PANEL 2 — ABSOLUTE MORPHOLOGY
# =========================================================
plot_jitter(
    ax1,
    [cell_lengths, cell_widths, nuc_lengths, nuc_widths],
    ["Cell length", "Cell width", "Nucleus length", "Nucleus width"],
    [COLOR_CELL, COLOR_CELL, COLOR_NUC, COLOR_NUC],
    step=0.1
)

ax1.set_title("Absolute morphology")
ax1.set_ylabel("Dimension (µm)")


# =========================================================
# PANEL 3 — RELATIVE MORPHOLOGY
# =========================================================
plot_jitter(
    ax2,
    [cell_lengths_n, cell_widths_n, nuc_lengths_n, nuc_widths_n],
    ["Cell length", "Cell width", "Nucleus length", "Nucleus width"],
    [COLOR_CELL, COLOR_CELL, COLOR_NUC, COLOR_NUC],
    step=0.1
)

ax2.set_title("Relative morphology")
ax2.set_ylabel("Relative to mean (-)")

plt.tight_layout()
plt.savefig("cell_statistics.pdf", dpi=300)


def print_stats(name, data):
    data = np.array(data)
    print(f"{name:30s}  mean = {np.mean(data):.4f}   std = {np.std(data):.4f}   n = {len(data)}")


print("\n===== ANGLES =====")
print_stats("Sarcomeres", sarco_means)
print_stats("Nucleui", nuc_means)

print("\n===== ABSOLUTE MORPHOLOGY =====")
print_stats("Cell length", cell_lengths)
print_stats("Cell width", cell_widths)
print_stats("Nucleus length", nuc_lengths)
print_stats("Nucleus width", nuc_widths)

print("\n===== RELATIVE MORPHOLOGY =====")
print_stats("Cell length (norm)", cell_lengths_n)
print_stats("Cell width (norm)", cell_widths_n)
print_stats("Nucleus length (norm)", nuc_lengths_n)
print_stats("Nucleus width (norm)", nuc_widths_n)

plt.show()
