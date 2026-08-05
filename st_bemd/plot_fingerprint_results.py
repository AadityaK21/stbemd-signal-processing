"""
plot_fingerprint_results.py
===========================
Self-contained script that reproduces every fingerprint plot in the report.
Run it from inside the emd2d/ folder (so the baseline modules import):

    cd emd2d
    python plot_fingerprint_results.py

It writes three PNGs to ./plots/ :
    1. fp_orientation.png   -- print + ridge orientation + coherence
    2. fp_imf_error.png     -- IMF-1 per method + orientation-error maps
    3. fp_metrics.png       -- bar chart of the standard metrics

The comments explain the matplotlib patterns you will reuse for any other plot
(grayscale images, an HSV orientation field, masked error maps, shared
colorbars, grouped bar charts).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")              # headless backend: render straight to file, no GUI
import matplotlib.pyplot as plt

# project modules (must be run from the emd2d/ directory)
from pseudo_bemd import pseudo_bemd
from serial_emd import serial_emd
from demd import demd
from bemd import bemd
from structure_tensor import orientation_and_coherence
import metrics as Mdir
import emd_metrics_standard as Mstd

PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)


# ---------------------------------------------------------------------------
# 0. Load the fingerprint and run the four baselines
# ---------------------------------------------------------------------------
def load_fingerprint(path="../data/fingerprint.npy"):
    """Load the pre-saved real fingerprint (90x90, float, zero-mean)."""
    return np.load(path)


def run_baselines(fp):
    """Return an ordered dict {name: (imfs, residual)} for the four methods."""
    decomp = {}
    decomp["pseudo-BEMD"]    = pseudo_bemd(fp)[:2]
    decomp["serial-EMD"]     = serial_emd(fp, order="row")[:2]
    imfs, res, _ = demd(fp);  decomp["DEMD"] = (imfs, res)
    decomp["isotropic BEMD"] = bemd(fp)[:2]
    return decomp


# ---------------------------------------------------------------------------
# 1. Orientation field plot:  image | orientation (HSV) | coherence
# ---------------------------------------------------------------------------
def plot_orientation_field(fp):
    # the structure tensor gives a per-pixel angle (radians) and coherence in [0,1]
    theta, coh = orientation_and_coherence(fp)

    # plt.subplots(rows, cols) returns the Figure and an array of Axes
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))

    # (a) grayscale image: cmap="gray"
    ax[0].imshow(fp, cmap="gray")
    ax[0].set_title("fingerprint")

    # (b) orientation as colour: angles are cyclic, so use a CYCLIC colormap
    #     ("hsv"). Convert rad->deg and wrap to [0,180) since orientation is
    #     defined modulo 180 degrees.
    ax[1].imshow(np.rad2deg(theta) % 180, cmap="hsv")
    ax[1].set_title("ridge orientation (deg)")

    # (c) coherence map with a fixed 0..1 scale and a colorbar
    im = ax[2].imshow(coh, cmap="viridis", vmin=0, vmax=1)
    ax[2].set_title("coherence")
    fig.colorbar(im, ax=ax[2], fraction=0.046)   # attach a colorbar to that axis

    for a in ax:                 # strip ticks: images don't need pixel axes
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/fp_orientation.png", dpi=130)
    plt.close(fig)               # close to free memory when scripting many figures


# ---------------------------------------------------------------------------
# 2. IMF-1 per method + orientation-error map (the key comparison figure)
# ---------------------------------------------------------------------------
def local_orientation_error_map(imf, theta_ref, coh_ref, coh_floor=0.05):
    """Per-pixel angular error (deg) of an IMF vs a reference orientation field,
    masked to regions where the reference is reliable (coherence > floor)."""
    theta_i, _ = orientation_and_coherence(imf)
    d = np.abs(theta_i - theta_ref)
    d = np.minimum(d, np.pi - d)              # fold to [0, pi/2]: angles are mod pi
    return np.rad2deg(d) * (coh_ref > coh_floor)


def plot_imf_and_error(fp, decomp):
    theta_ref, coh_ref = orientation_and_coherence(fp)
    methods = list(decomp.keys())

    # a 2 x N grid: row 0 = IMF-1 images, row 1 = error maps
    fig, ax = plt.subplots(2, len(methods), figsize=(13, 6.6))

    im_err = None
    for j, name in enumerate(methods):
        imfs, _ = decomp[name]
        imf1 = imfs[0]

        ax[0, j].imshow(imf1, cmap="gray")
        ax[0, j].set_title(name, fontsize=10)

        emap = local_orientation_error_map(imf1, theta_ref, coh_ref)
        # vmin/vmax FIXED across all panels so colours are comparable between methods
        im_err = ax[1, j].imshow(emap, cmap="magma", vmin=0, vmax=45)

        # report the single-number coherence-weighted error in the title
        err = Mdir.local_orientation_error(imf1, fp)
        ax[1, j].set_title(f"err = {err:.1f}$^\\circ$", fontsize=9)

    ax[0, 0].set_ylabel("IMF-1 (ridge mode)", fontsize=10)
    ax[1, 0].set_ylabel("orientation error", fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])

    # ONE shared colorbar for the whole error row (pass the row of axes)
    fig.colorbar(im_err, ax=ax[1, :], fraction=0.025, pad=0.01, label="deg")
    fig.savefig(f"{PLOTS}/fp_imf_error.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Grouped bar chart of the standard metrics
# ---------------------------------------------------------------------------
def plot_metrics(fp, decomp):
    methods = list(decomp.keys())
    orient, overlap, energy = [], [], []
    for name in methods:
        imfs, res = decomp[name]
        orient.append(Mdir.local_orientation_error(imfs[0], fp))
        overlap.append(Mstd.spectral_overlap(imfs)[1])           # mean adjacent overlap
        energy.append(Mstd.index_of_energy_conservation(fp, imfs, res)["ratio"])

    x = np.arange(len(methods))      # bar group positions: 0,1,2,3
    w = 0.27                         # width of each bar within a group

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # left axis: orientation error (degrees) on its own scale
    ax[0].bar(x, orient, color="#c0392b")
    ax[0].set_xticks(x); ax[0].set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("local orientation error (deg)")
    ax[0].set_title("orientation tracking (lower = better)")

    # right axis: two bounded [0,1]-ish metrics side by side (grouped bars:
    # shift each series by +/- w using x - w and x)
    ax[1].bar(x - w/2, overlap, w, label="spectral overlap")
    ax[1].bar(x + w/2, energy,  w, label="energy ratio (ideal 1)")
    ax[1].axhline(1.0, ls="--", c="k", lw=0.8)          # mark the ideal energy ratio
    ax[1].set_xticks(x); ax[1].set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax[1].set_title("mode-mixing metrics")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(f"{PLOTS}/fp_metrics.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fp = load_fingerprint()
    decomp = run_baselines(fp)
    plot_orientation_field(fp)
    plot_imf_and_error(fp, decomp)
    plot_metrics(fp, decomp)
    print(f"wrote 3 figures to ./{PLOTS}/")
