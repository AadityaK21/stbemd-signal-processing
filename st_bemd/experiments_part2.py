"""
experiments_part2.py
--------------------
Mid-semester-review experiments:
  (A) a synthetic AM-FM signal with known multidirectional structure, to make
      mode mixing visible and to score it with the standard EMD metrics;
  (B) a real fingerprint (SOCOFing), the panel's suggested high-orientation-
      variance case, to show how the baselines fail near the core/delta;
  (C) the standard literature EMD-quality metrics applied to all four baselines.

Outputs figures to outputs/figures/ and numbers to
outputs/results/results_part2.json
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signals_ext import multiscale_multiorientation
from pseudo_bemd import pseudo_bemd
from serial_emd import serial_emd
from demd import demd
from bemd import bemd
import metrics as Mdir
import emd_metrics_standard as Mstd
from structure_tensor import orientation_and_coherence

from _paths import FIGURES, RESULTS, ensure_outputs, fingerprint

ensure_outputs()
FIG = str(FIGURES)
RES = str(RESULTS)
RD, BU = "RdBu_r", None


def run_all(signal):
    """Run the four baselines; return dict name -> (imfs, residual, seconds)."""
    out = {}
    t = time.time(); imfs, res = pseudo_bemd(signal);              out["pseudo-BEMD"]     = (imfs, res, time.time()-t)
    t = time.time(); imfs, res = serial_emd(signal, order="row");  out["serial-EMD"]      = (imfs, res, time.time()-t)
    t = time.time(); imfs, res, th = demd(signal);                 out["DEMD"]            = (imfs, res, time.time()-t)
    t = time.time(); imfs, res = bemd(signal);                     out["isotropic BEMD"] = (imfs, res, time.time()-t)
    return out


def std_metrics(signal, decomp):
    rows = {}
    for name, (imfs, res, secs) in decomp.items():
        s = Mstd.summarise(signal, imfs, res)
        s["recon_err"] = Mdir.reconstruction_error(signal, np.array(imfs), res)
        s["n_imfs"] = len(imfs)
        s["runtime_s"] = secs
        rows[name] = s
    return rows


# ===========================================================================
# (A) MULTIDIRECTIONAL SYNTHETIC SIGNAL
# ===========================================================================
def experiment_multidirectional():
    N = 80
    f, info = multiscale_multiorientation(N=N)
    comps = info["components"]                     # coarse, mid, fine
    decomp = run_all(f)

    # ---- Figure: signal + ground-truth components ----
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    ax[0].imshow(f, cmap=RD); ax[0].set_title("input (sum of 3)")
    labels = [f"comp {i+1}: {info['freqs'][i]:.0f} cyc, {info['thetas'][i]:.0f}$^\\circ$"
              for i in range(3)]
    for k in range(3):
        ax[k+1].imshow(comps[k], cmap=RD); ax[k+1].set_title(labels[k])
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{FIG}/fig8_multidir_signal.png", dpi=130); plt.close()

    # ---- Figure: qualitative mode mixing -- each method's first 3 IMFs ----
    methods = ["pseudo-BEMD", "serial-EMD", "DEMD", "isotropic BEMD"]
    fig, ax = plt.subplots(len(methods)+1, 3, figsize=(8.2, 11))
    # top row: ground-truth components (fine -> coarse to align with IMF order)
    gt = [comps[2], comps[1], comps[0]]
    gtlab = ["fine comp (IMF-1 should hold)", "mid comp", "coarse comp"]
    for j in range(3):
        ax[0, j].imshow(gt[j], cmap=RD); ax[0, j].set_title(gtlab[j], fontsize=9)
    ax[0,0].set_ylabel("ground truth", fontsize=9)
    for i, name in enumerate(methods):
        imfs, res, _ = decomp[name]
        for j in range(3):
            a = ax[i+1, j]
            if j < len(imfs):
                a.imshow(imfs[j], cmap=RD)
            else:
                a.imshow(res, cmap=RD)
            if i == 0: pass
        ax[i+1, 0].set_ylabel(name, fontsize=9)
        ax[i+1, 0].set_title("IMF 1" if i==0 else "", fontsize=9)
    for j,t in enumerate(["IMF 1","IMF 2","IMF 3"]):
        ax[1,j].set_title(t, fontsize=9)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{FIG}/fig9_multidir_modemixing.png", dpi=130); plt.close()

    # ---- metrics ----
    rows = std_metrics(f, decomp)
    for name, (imfs, res, _) in decomp.items():
        rows[name]["separation"] = Mdir.separation_score(imfs, comps)
        rows[name]["orthogonality_dir"] = Mdir.orthogonality_index(imfs)

    # ---- Figure: standard-metric spectra + bars ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    # (a) radial spectra of isotropic BEMD's IMFs (show band overlap)
    imfs_b = decomp["isotropic BEMD"][0]
    for k, c in enumerate(imfs_b[:3]):
        centers, prof = Mstd.radial_power_spectrum(c)
        ax[0].plot(centers, prof, label=f"BEMD IMF{k+1}")
    ax[0].set_title("(a) radial power spectra of IMFs (isotropic BEMD)")
    ax[0].set_xlabel("radial spatial frequency"); ax[0].set_ylabel("normalised power")
    ax[0].legend(); ax[0].set_xlim(0, 30)
    # (b) bar chart: bounded mode-mixing metrics per method (all in [0,1])
    names = methods
    corr = [rows[n]["imf_mean_abs_corr"] for n in names]
    so = [rows[n]["spectral_overlap_mean"] for n in names]
    x = np.arange(len(names)); w = 0.38
    ax[1].bar(x-w/2, corr, w, label="mean |IMF correlation|")
    ax[1].bar(x+w/2, so, w, label="adjacent spectral overlap")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) standard mode-mixing metrics (lower = cleaner)")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/fig10_standard_metrics.png", dpi=130); plt.close()

    return rows, info


# ===========================================================================
# (B) REAL FINGERPRINT
# ===========================================================================
def experiment_fingerprint():
    fp = fingerprint()
    decomp = run_all(fp)
    th_ref, coh_ref = orientation_and_coherence(fp)

    # ---- Figure: fingerprint + orientation field + coherence ----
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].imshow(fp, cmap="gray"); ax[0].set_title("real fingerprint (SOCOFing)")
    ax[1].imshow(np.rad2deg(th_ref) % 180, cmap="hsv"); ax[1].set_title("ridge orientation (deg)")
    im = ax[2].imshow(coh_ref, cmap="viridis", vmin=0, vmax=1); ax[2].set_title("structure-tensor coherence")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{FIG}/fig11_fingerprint.png", dpi=130); plt.close()

    # ---- Figure: IMF-1 (ridge mode) per method + local-orientation-error map ----
    methods = ["pseudo-BEMD", "serial-EMD", "DEMD", "isotropic BEMD"]
    fig, ax = plt.subplots(2, len(methods), figsize=(13, 6.6))
    errs = {}
    for i, name in enumerate(methods):
        imfs = decomp[name][0]
        imf1 = imfs[0]
        ax[0, i].imshow(imf1, cmap="gray"); ax[0, i].set_title(name, fontsize=10)
        # per-pixel local orientation error of IMF1 vs fingerprint ridge flow
        th_i, _ = orientation_and_coherence(imf1)
        d = np.abs(th_i - th_ref); d = np.minimum(d, np.pi - d)
        emap = np.rad2deg(d) * (coh_ref > 0.05)
        im = ax[1, i].imshow(emap, cmap="magma", vmin=0, vmax=45)
        errs[name] = Mdir.local_orientation_error(imf1, fp)
        ax[1, i].set_title(f"loc. orient. err = {errs[name]:.2f}$^\\circ$", fontsize=9)
    ax[0,0].set_ylabel("IMF-1 (ridge mode)", fontsize=10)
    ax[1,0].set_ylabel("orientation error map", fontsize=10)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=ax[1, :], fraction=0.025, pad=0.01, label="deg")
    plt.savefig(f"{FIG}/fig12_fingerprint_failure.png", dpi=130, bbox_inches="tight"); plt.close()

    rows = std_metrics(fp, decomp)
    for name in methods:
        rows[name]["local_orient_err_deg"] = errs[name]
    return rows


def main():
    print("== (A) multidirectional synthetic ==")
    rows_a, info = experiment_multidirectional()
    print("== (B) real fingerprint ==")
    rows_b = experiment_fingerprint()
    out = {"multidirectional": rows_a, "fingerprint": rows_b,
           "multidirectional_info": {"freqs": info["freqs"], "thetas": info["thetas"]}}
    with open(f"{RES}/results_part2.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    # console summary tables
    def tbl(title, rows, cols):
        print(f"\n{title}")
        hdr = "  {:<16}".format("method") + "".join(f"{c:>22}" for c in cols)
        print(hdr)
        for n, r in rows.items():
            print("  {:<16}".format(n) + "".join(f"{r.get(c, float('nan')):>22.4g}" for c in cols))
    tbl("(A) multidirectional -- standard metrics",
        rows_a, ["index_of_orthogonality", "spectral_overlap_mean",
                 "energy_leakage", "separation"])
    tbl("(B) fingerprint -- standard + directional",
        rows_b, ["index_of_orthogonality", "spectral_overlap_mean",
                 "imf1_coherence", "local_orient_err_deg"])
    print("\nsaved results_part2.json")


if __name__ == "__main__":
    main()
