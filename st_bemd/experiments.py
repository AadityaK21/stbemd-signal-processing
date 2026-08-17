"""
experiments.py
--------------
Driver for the mid-term benchmark. Generates every figure and the results table
used in the progress report. All baselines are the from-scratch implementations
in this package.
"""
import time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import signals as S
from metrics import (reconstruction_error, orthogonality_index,
                     orientation_error, separation_score,
                     local_orientation_error)
from scipy.ndimage import rotate as nd_rotate
from structure_tensor import dominant_orientation_deg, orientation_and_coherence
from pseudo_bemd import pseudo_bemd
from serial_emd import serial_emd
from bemd import bemd, find_extrema_2d, _envelope_surface
from demd import demd

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.titlesize": 9,
    "axes.grid": False, "image.cmap": "RdBu_r",
})
from _paths import FIGURES, RESULTS, ensure_outputs

ensure_outputs()
FIG = str(FIGURES)
N = 80
results = {}


def show(ax, img, title):
    v = np.max(np.abs(img)) + 1e-9
    ax.imshow(img, vmin=-v, vmax=v)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])


# ----------------------------------------------------------------------
# Figure 1 : the synthetic test signals
# ----------------------------------------------------------------------
def fig_signals():
    pw = S.plane_wave(N, 12, 35)[0]
    tw = S.two_orientation(N)[0]
    vo = S.varying_orientation(N)[0]
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.9))
    show(ax[0], pw, "(a) single orientation\n$\\theta=35^\\circ$")
    show(ax[1], tw, "(b) two orientations\n$20^\\circ + 110^\\circ$")
    show(ax[2], vo, "(c) spatially varying\norientation")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_signals.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2 : IMF decomposition of the two-orientation signal, all methods
# ----------------------------------------------------------------------
def fig_decomp():
    f, info = S.two_orientation(N)
    comps = info["components"]
    decomps = {
        "pseudo-BEMD": pseudo_bemd(f)[:2],
        "serial-EMD (row)": serial_emd(f, "row")[:2],
        "DEMD": demd(f)[:2],
        "isotropic BEMD": bemd(f)[:2],
    }
    fig = plt.figure(figsize=(8.6, 7.6))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.1)
    # ground truth row
    show(fig.add_subplot(gs[0, 0]), f, "input")
    show(fig.add_subplot(gs[0, 1]), comps[0], "true comp. 1 ($20^\\circ$)")
    show(fig.add_subplot(gs[0, 2]), comps[1], "true comp. 2 ($110^\\circ$)")
    for r, (name, (imfs, _res)) in enumerate(decomps.items(), start=1):
        show(fig.add_subplot(gs[r, 0]), imfs[0] if imfs else np.zeros((N, N)),
             f"{name}\nIMF 1")
        show(fig.add_subplot(gs[r, 1]),
             imfs[1] if len(imfs) > 1 else np.zeros((N, N)), "IMF 2")
        resid = f - np.sum(imfs, axis=0) if imfs else f
        show(fig.add_subplot(gs[r, 2]), resid, "IMF$_{\\geq3}$+residual")
    fig.savefig(f"{FIG}/fig2_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 : orientation bias vs input angle
# ----------------------------------------------------------------------
def fig_orientation_bias():
    angles = np.arange(0, 95, 10)
    methods = {"pseudo-BEMD": lambda f: pseudo_bemd(f)[0],
               "DEMD": lambda f: demd(f)[0],
               "isotropic BEMD": lambda f: bemd(f)[0]}
    errs = {m: [] for m in methods}
    for a in angles:
        f = S.plane_wave(N, 12, float(a))[0]
        for m, fn in methods.items():
            imfs = fn(f)
            # orientation of the finest oscillatory IMF
            est = dominant_orientation_deg(imfs[0]) if imfs else 0.0
            errs[m].append(orientation_error(a, est))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    marks = {"pseudo-BEMD": "o-", "DEMD": "s-", "isotropic BEMD": "^-"}
    for m in methods:
        ax.plot(angles, errs[m], marks[m], label=m, ms=4)
    ax.set_xlabel("input orientation $\\theta$ (deg)")
    ax.set_ylabel("IMF-1 orientation error (deg)")
    ax.set_title("Orientation error vs. input angle (single plane wave)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_orientation_bias.png",
                                    bbox_inches="tight")
    plt.close(fig)
    results["orientation_bias_mean_err"] = {m: float(np.mean(errs[m])) for m in methods}


# ----------------------------------------------------------------------
# Figure 4 : scan-order dependence of serial-EMD
# ----------------------------------------------------------------------
def fig_scan_order():
    f = S.two_orientation(N)[0]
    orders = ["row", "col", "snake", "diag"]
    imf1 = {}
    for o in orders:
        imfs = serial_emd(f, o)[0]
        imf1[o] = imfs[0] if imfs else np.zeros((N, N))
    fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.5))
    for k, o in enumerate(orders):
        show(ax[k], imf1[o], f"scan = {o}")
    fig.suptitle("Serial-EMD IMF 1 under different scan orders", y=1.05)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_scan_order.png", bbox_inches="tight")
    plt.close(fig)
    # pairwise correlation between scan orders (1 => order-independent)
    keys = orders
    cc = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            a = imf1[keys[i]].ravel(); b = imf1[keys[j]].ravel()
            cc[i, j] = np.corrcoef(a, b)[0, 1]
    off = cc[~np.eye(4, dtype=bool)]
    results["serial_scan_offdiag_corr_mean"] = float(np.mean(off))
    results["serial_scan_offdiag_corr_min"] = float(np.min(off))


# ----------------------------------------------------------------------
# Figure 5 : why isotropic envelopes mix oriented modes -> motivates ST-BEMD
# ----------------------------------------------------------------------
def fig_kernel_motivation():
    from matplotlib.patches import Circle, Ellipse
    f = S.plane_wave(N, 7, 30)[0]
    th, coh = orientation_and_coherence(f)
    mx, _ = find_extrema_2d(f)
    ys, xs = np.nonzero(mx)

    fig, ax = plt.subplots(1, 3, figsize=(8.6, 3.0))
    # (a) isotropic footprint
    show(ax[0], f, "(a) isotropic kernel\n(current BEMD)")
    ax[0].plot(xs, ys, "k.", ms=2)
    cy, cx = N // 2, N // 2
    ax[0].add_patch(Circle((cx, cy), 9, fill=False, ec="lime", lw=2))
    # (b) anisotropic footprint aligned to local structure-tensor orientation
    show(ax[1], f, "(b) anisotropic kernel\n(proposed ST-BEMD)")
    ax[1].plot(xs, ys, "k.", ms=2)
    ang = np.rad2deg(th[cy, cx])           # local gradient (wave-vector) direction
    # ellipse elongated ALONG crests (perpendicular to gradient), compressed across
    ell = Ellipse((cx, cy), width=20, height=6, angle=ang + 90,
                  fill=False, ec="lime", lw=2)
    ax[1].add_patch(ell)
    # (c) structure-tensor coherence: shows the field is strongly directional
    im = ax[2].imshow(coh, cmap="viridis", vmin=0, vmax=1)
    ax[2].set_title("(c) structure-tensor\ncoherence $c$"); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.colorbar(im, ax=ax[2], shrink=0.7)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_kernel_motivation.png",
                                    bbox_inches="tight")
    plt.close(fig)
    # characterise the signal class: mean coherence (should be high => directional)
    results["mean_coherence"] = {
        "plane": float(orientation_and_coherence(S.plane_wave(N, 12, 35)[0])[1].mean()),
        "two_orientation": float(orientation_and_coherence(S.two_orientation(N)[0])[1].mean()),
        "varying": float(orientation_and_coherence(S.varying_orientation(N)[0])[1].mean()),
    }


# ----------------------------------------------------------------------
# Figure 6 : real oriented texture decomposed by isotropic BEMD
# ----------------------------------------------------------------------
def fig_real_texture():
    # scikit-image supplies the sample texture and nothing else in this repo,
    # so it is an optional extra rather than a hard dependency. Skip the figure
    # rather than taking the whole run down with it.
    try:
        from skimage.data import brick
        from skimage.transform import resize
    except ImportError:
        print("  skipping fig6_real_texture — needs scikit-image "
              "(pip install 'st-bemd[figures]')")
        return {}
    img = brick().astype(float)
    img = resize(img, (96, 96), anti_aliasing=True)
    img = (img - img.mean())
    imfs, res = bemd(img, max_imfs=3)
    fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.5))
    show(ax[0], img, "brick texture")
    for k in range(2):
        show(ax[k + 1], imfs[k] if k < len(imfs) else np.zeros_like(img),
             f"BEMD IMF {k+1}")
    show(ax[3], img - np.sum(imfs[:2], axis=0), "coarse residual")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_real_texture.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 7 : varying-orientation regime -- DEMD vs isotropic BEMD
# ----------------------------------------------------------------------
def fig_varying_orientation():
    g = S.varying_orientation(N)[0]
    th_ref, coh_ref = orientation_and_coherence(g)
    demd_i = demd(g)[0]
    bemd_i = bemd(g)[0]
    demd1 = demd_i[0] if demd_i else np.zeros_like(g)
    bemd1 = bemd_i[0] if bemd_i else np.zeros_like(g)
    th_d = orientation_and_coherence(demd1)[0]
    th_b = orientation_and_coherence(bemd1)[0]

    def angerr(a, b):
        d = np.abs(a - b); d = np.minimum(d, np.pi - d)
        return np.rad2deg(d)

    fig, ax = plt.subplots(2, 3, figsize=(8.6, 5.6))
    show(ax[0, 0], g, "varying-orientation input")
    ax[0, 1].imshow(np.rad2deg(th_ref) % 180, cmap="twilight"); ax[0, 1].set_title("true local orientation")
    ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
    ax[0, 2].axis("off")
    show(ax[1, 0], demd1, "DEMD IMF 1")
    im = ax[1, 1].imshow(angerr(th_d, th_ref) * (coh_ref > 0.05), cmap="hot", vmin=0, vmax=45)
    ax[1, 1].set_title("DEMD local-orient. error"); ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
    ax[1, 2].imshow(angerr(th_b, th_ref) * (coh_ref > 0.05), cmap="hot", vmin=0, vmax=45)
    ax[1, 2].set_title("BEMD local-orient. error"); ax[1, 2].set_xticks([]); ax[1, 2].set_yticks([])
    fig.colorbar(im, ax=ax[1, :].tolist(), shrink=0.6, label="deg")
    fig.savefig(f"{FIG}/fig7_varying_orientation.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Benchmark table : metrics + timing on the two-orientation signal
# ----------------------------------------------------------------------
def benchmark_table():
    f, info = S.two_orientation(N)
    comps = info["components"]
    methods = {
        "pseudo-BEMD": lambda f: pseudo_bemd(f),
        "serial-EMD (row)": lambda f: serial_emd(f, "row"),
        "DEMD": lambda f: (lambda r: (r[0], r[1]))(demd(f)),
        "isotropic BEMD": lambda f: bemd(f),
    }
    table = {}
    for name, fn in methods.items():
        t0 = time.time()
        imfs, res = fn(f)
        dt = time.time() - t0
        table[name] = {
            "n_imfs": len(imfs),
            "recon_err": reconstruction_error(f, imfs, res),
            "orthogonality": orthogonality_index(imfs),
            "separation": separation_score(imfs, comps),
            "runtime_s": dt,
        }
    results["benchmark_two_orientation"] = table

    # ---- varying-orientation regime: where the single-direction assumption breaks
    g = S.varying_orientation(N)[0]
    vary = {}
    for name, fn in methods.items():
        imfs, res = fn(g)
        finest = imfs[0] if imfs else np.zeros_like(g)
        vary[name] = {
            "orthogonality": orthogonality_index(imfs),
            "local_orient_err_deg": local_orientation_error(finest, g),
        }
    results["benchmark_varying_orientation"] = vary

    # ---- proper orientation-invariance test (the proposal's criterion):
    # decompose at 0 deg; rotate input by alpha, decompose, rotate IMF1 back,
    # and correlate with the unrotated IMF1. 1.0 => rotation-equivariant.
    inv = {}
    base = S.plane_wave(N, 12, 30.0)[0]
    for name, fn in methods.items():
        ref_list = fn(base)[0]
        ref_imf1 = ref_list[0] if ref_list else np.zeros_like(base)
        cors = []
        for alpha in [15, 30, 45]:
            grot = nd_rotate(base, angle=alpha, reshape=False, order=3, mode="reflect")
            imf_list = fn(grot)[0]
            c1 = imf_list[0] if imf_list else np.zeros_like(base)
            c1_back = nd_rotate(c1, angle=-alpha, reshape=False, order=3, mode="reflect")
            m = (N // 6)  # ignore rotation border band
            a = ref_imf1[m:-m, m:-m].ravel()
            b = c1_back[m:-m, m:-m].ravel()
            cors.append(float(np.corrcoef(a, b)[0, 1]))
        inv[name] = float(np.mean(cors))
    results["orientation_invariance_corr"] = inv


if __name__ == "__main__":
    t0 = time.time()
    fig_signals();            print("fig1 done")
    fig_decomp();             print("fig2 done")
    fig_orientation_bias();   print("fig3 done")
    fig_scan_order();         print("fig4 done")
    fig_kernel_motivation();  print("fig5 done")
    fig_real_texture();       print("fig6 done")
    fig_varying_orientation();print("fig7 done")
    benchmark_table();        print("benchmark done")
    with open(RESULTS / "results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nALL DONE in {time.time()-t0:.1f}s")
    print(json.dumps(results, indent=2))
