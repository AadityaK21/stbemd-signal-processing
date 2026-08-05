"""
experiments_stbemd.py
---------------------
Second-half evaluation of the proposed ST-BEMD against the baselines.

Produces:
  results_stbemd.json
  figS1_fingerprint_stbemd.png   -- headline real-data: iso vs ST IMF-1 + error maps
  figS2_varying_stbemd.png       -- headline synthetic GT: error maps
  figS3_monogenic.png            -- Riesz/monogenic amplitude/phase/orientation of ST IMF-1
  figS4_summary.png              -- orientation-invariance, rho effect, metric bars
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as ndrotate

from signals import varying_orientation, two_orientation, plane_wave
from metrics import (reconstruction_error, separation_score,
                     local_orientation_error, orthogonality_index)
from emd_metrics_standard import (index_of_energy_conservation, spectral_overlap,
                                  imf_mean_coherence)
from structure_tensor import orientation_and_coherence
from riesz import monogenic
from bemd import bemd
from demd import demd
from stbemd import stbemd

FIG = "/home/claude/figures"
RES = "/home/claude/results"


def loc_err_map(imf, ref):
    th_r, coh_r = orientation_and_coherence(ref)
    th_i, _ = orientation_and_coherence(imf)
    d = np.abs(th_i - th_r); d = np.minimum(d, np.pi - d)
    return np.rad2deg(d) * (coh_r > 0.05)


# ---------------------------------------------------------------------------
def fig_fingerprint():
    fp = np.load("/home/claude/data/fingerprint.npy")
    ib, ir = bemd(fp); sb, sr = stbemd(fp)
    fig, ax = plt.subplots(2, 3, figsize=(11, 7))
    th_ref, _ = orientation_and_coherence(fp)
    ax[0, 0].imshow(fp, cmap="gray"); ax[0, 0].set_title("fingerprint")
    ax[0, 1].imshow(ib[0], cmap="gray"); ax[0, 1].set_title("isotropic BEMD  IMF-1")
    ax[0, 2].imshow(sb[0], cmap="gray"); ax[0, 2].set_title("ST-BEMD  IMF-1")
    ax[1, 0].imshow(np.rad2deg(th_ref) % 180, cmap="hsv"); ax[1, 0].set_title("ridge orientation")
    e_i = loc_err_map(ib[0], fp); e_s = loc_err_map(sb[0], fp)
    ax[1, 1].imshow(e_i, cmap="magma", vmin=0, vmax=45)
    ax[1, 1].set_title(f"iso error  {local_orientation_error(ib[0],fp):.2f}$^\\circ$")
    im = ax[1, 2].imshow(e_s, cmap="magma", vmin=0, vmax=45)
    ax[1, 2].set_title(f"ST error  {local_orientation_error(sb[0],fp):.2f}$^\\circ$")
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[1, :], fraction=0.02, pad=0.01, label="deg")
    fig.savefig(f"{FIG}/figS1_fingerprint_stbemd.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_varying():
    fv = varying_orientation(96)[0]
    ib = bemd(fv)[0][0]; sb = stbemd(fv)[0][0]
    th_ref, _ = orientation_and_coherence(fv)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].imshow(fv, cmap="RdBu_r"); ax[0].set_title("varying-orientation input")
    ax[1].imshow(np.rad2deg(th_ref) % 180, cmap="hsv"); ax[1].set_title("true local orientation")
    ax[2].imshow(loc_err_map(ib, fv), cmap="magma", vmin=0, vmax=10)
    ax[2].set_title(f"iso error  {local_orientation_error(ib,fv):.3f}$^\\circ$")
    im = ax[3].imshow(loc_err_map(sb, fv), cmap="magma", vmin=0, vmax=10)
    ax[3].set_title(f"ST error  {local_orientation_error(sb,fv):.3f}$^\\circ$")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=[ax[2], ax[3]], fraction=0.04, label="deg")
    fig.savefig(f"{FIG}/figS2_varying_stbemd.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_monogenic():
    # in-class rotating AM-FM single carrier: show monogenic amplitude/phase/orientation
    fv = varying_orientation(96)[0]
    xs = np.linspace(-1, 1, 96); X, Y = np.meshgrid(xs, xs)
    A = 0.6 + 0.4 * np.cos(np.pi * X) * np.cos(np.pi * Y)
    f = A * fv
    imf1 = stbemd(f)[0][0]
    m = monogenic(imf1)
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].imshow(imf1, cmap="RdBu_r"); ax[0].set_title("ST-BEMD IMF-1")
    ax[1].imshow(m["amplitude"], cmap="viridis"); ax[1].set_title("monogenic amplitude")
    ax[2].imshow(m["phase"], cmap="twilight"); ax[2].set_title("monogenic phase")
    ax[3].imshow(np.rad2deg(m["orientation"]), cmap="hsv"); ax[3].set_title("monogenic orientation")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{FIG}/figS3_monogenic.png", dpi=130); plt.close(fig)
    # amplitude recovery error vs true AM
    return float(np.mean(np.abs(m["amplitude"] - A)) / np.mean(A))


def orientation_invariance(signal, method, angles=(15, 30, 45, 60)):
    """Equivariance: rotate input, decompose, rotate IMF-1 back, correlate with base IMF-1."""
    def imf1(x):
        out = method(x)
        return out[0][0] if out[0] else np.zeros_like(x)
    base = imf1(signal)
    cs = []
    for a in angles:
        rot = ndrotate(signal, a, reshape=False, order=3, mode="reflect")
        ri = imf1(rot)
        back = ndrotate(ri, -a, reshape=False, order=3, mode="reflect")
        m = (np.abs(base) > 1e-9) | (np.abs(back) > 1e-9)
        x = base[m] - base[m].mean(); y = back[m] - back[m].mean()
        cs.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    return float(np.mean(cs))


def fig_summary(results):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # (a) orientation error: iso vs ST on synthetic GT and fingerprint
    cats = ["varying\n(GT)", "fingerprint\n(real)"]
    iso = [results["varying"]["isotropic BEMD"]["loc_orient_err"],
           results["fingerprint"]["isotropic BEMD"]["loc_orient_err"]]
    st = [results["varying"]["ST-BEMD"]["loc_orient_err"],
          results["fingerprint"]["ST-BEMD"]["loc_orient_err"]]
    x = np.arange(2); w = 0.35
    ax[0].bar(x - w/2, iso, w, label="isotropic BEMD")
    ax[0].bar(x + w/2, st, w, label="ST-BEMD")
    ax[0].set_xticks(x); ax[0].set_xticklabels(cats)
    ax[0].set_ylabel("local orientation error (deg)")
    ax[0].set_title("(a) orientation tracking (lower=better)"); ax[0].legend()
    # (b) rho effect on fingerprint
    rhos = results["rho_sweep"]["rho"]; errs = results["rho_sweep"]["fingerprint_err"]
    ax[1].plot(rhos, errs, "o-")
    ax[1].axhline(results["fingerprint"]["isotropic BEMD"]["loc_orient_err"], ls="--", c="gray", label="isotropic BEMD")
    ax[1].set_xlabel("anisotropy ratio $\\rho_{max}$"); ax[1].set_ylabel("fingerprint error (deg)")
    ax[1].set_title("(b) effect of anisotropy"); ax[1].legend()
    # (c) orientation invariance
    names = list(results["invariance"].keys())
    vals = [results["invariance"][n] for n in names]
    ax[2].bar(names, vals, color=["#888", "#c0392b"])
    ax[2].set_ylim(0.9, 1.0); ax[2].set_title("(c) rotation-equivariance corr (higher=better)")
    for i, v in enumerate(vals): ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(f"{FIG}/figS4_summary.png", dpi=130); plt.close(fig)


def bench(signal, ref_for_orient, components=None):
    rows = {}
    for name, fn in [("isotropic BEMD", bemd), ("ST-BEMD", stbemd)]:
        t = time.time(); imfs, res = fn(signal); secs = time.time() - t
        r = {"recon_err": reconstruction_error(signal, imfs, res),
             "loc_orient_err": local_orientation_error(imfs[0], ref_for_orient),
             "energy_ratio": index_of_energy_conservation(signal, imfs, res)["ratio"],
             "spectral_overlap": spectral_overlap(imfs)[1],
             "imf1_coherence": imf_mean_coherence(imfs[0]),
             "orthogonality": orthogonality_index(imfs),
             "n_imfs": len(imfs), "runtime_s": secs}
        if components is not None:
            r["separation"] = separation_score(imfs, components)
        rows[name] = r
    return rows


def main():
    results = {}
    print("benchmarking varying-orientation (ground truth)...")
    fv = varying_orientation(96)[0]
    results["varying"] = bench(fv, fv)
    print("benchmarking fingerprint...")
    fp = np.load("/home/claude/data/fingerprint.npy")
    results["fingerprint"] = bench(fp, fp)
    print("benchmarking two-orientation (out-of-class)...")
    f2, info = two_orientation(80)
    results["two_orientation"] = bench(f2, f2, components=info["components"])

    print("rho sweep on fingerprint...")
    rhos = [2, 3, 4, 5, 7, 10]
    errs = [local_orientation_error(stbemd(fp, rho_max=r)[0][0], fp) for r in rhos]
    results["rho_sweep"] = {"rho": rhos, "fingerprint_err": errs}

    print("orientation invariance...")
    pw = plane_wave(80, 10, 35)[0]
    results["invariance"] = {
        "isotropic BEMD": orientation_invariance(pw, bemd),
        "ST-BEMD": orientation_invariance(pw, stbemd)}

    print("figures...")
    fig_fingerprint(); fig_varying()
    amp_err = fig_monogenic()
    results["monogenic_amp_relerr"] = amp_err
    fig_summary(results)

    with open(f"{RES}/results_stbemd.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # console summary
    def show(title, rows, keys):
        print(f"\n{title}")
        print("  " + "method".ljust(16) + "".join(k[:13].rjust(15) for k in keys))
        for n, r in rows.items():
            print("  " + n.ljust(16) + "".join(f"{r.get(k, float('nan')):>15.4g}" for k in keys))
    show("VARYING-ORIENTATION (ground truth)", results["varying"],
         ["loc_orient_err", "energy_ratio", "spectral_overlap", "n_imfs"])
    show("FINGERPRINT (real)", results["fingerprint"],
         ["loc_orient_err", "energy_ratio", "spectral_overlap", "imf1_coherence", "n_imfs"])
    show("TWO-ORIENTATION (out-of-class)", results["two_orientation"],
         ["separation", "loc_orient_err", "n_imfs"])
    print(f"\ninvariance: iso={results['invariance']['isotropic BEMD']:.4f}  "
          f"ST={results['invariance']['ST-BEMD']:.4f}")
    print(f"monogenic amplitude rel-err (ST IMF-1 on AM signal) = {amp_err:.3f}")
    print("\nsaved results_stbemd.json + 4 figures")


if __name__ == "__main__":
    main()
