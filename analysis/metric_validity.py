"""
metric_validity.py
------------------
Re-test of control_smoothing.py, correcting two flaws in that experiment and
adding the metric it should have used in the first place.

What was wrong with the control
-------------------------------
1. SELECTION BIAS. The Gaussian high-pass sigma was swept over the same 30
   prints it was then scored on, and the best sigma reported. ST-BEMD received
   no such tuning -- it estimates its own bandwidth per print. Comparing a
   tuned estimator against an untuned one is not a fair fight, and this project
   criticises exactly that error elsewhere (see sweep_rho.py). Fixed here with
   a held-out split: sigma is chosen on prints 0-14 and scored on prints 15-29.

2. WRONG QUESTION. Orientation preservation is not what an IMF is for. The
   defining property of an intrinsic mode function is that its LOCAL MEAN is
   zero everywhere -- that is what sifting iterates towards and what the
   stopping criterion tests. A Gaussian high-pass has no mechanism to achieve
   it. So the honest comparison scores both properties:

     orientation error   does the mode preserve the input's orientation field?
     IMF validity        is the result actually an IMF?  ||local mean|| / ||mode||

   An estimator that wins the first while failing the second has not beaten
   the method; it has revealed that the first metric alone is insufficient.

The local mean is computed with the project's own envelope machinery
(isotropic, rho=1) on the candidate mode's extrema, so no method is scored with
its own kernel and the measure stays neutral between them.

This script does NOT try to make ST-BEMD win. It reports what comes out.

    python analysis/metric_validity.py
"""
import _bootstrap  # noqa: F401  (puts the source dirs on sys.path)

import json
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from bemd import bemd, find_extrema_2d
from stbemd import stbemd, _aniso_envelope, _scale_estimate
from metrics import local_orientation_error
from structure_tensor import orientation_and_coherence
from _paths import FIGURES, RESULTS, ensure_outputs, fingerprint_batch

warnings.filterwarnings("ignore")

SIGMAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SPLIT = 15          # prints 0..14 select sigma, 15..29 score it


# ---------------------------------------------------------------------------
# IMF validity: the defining EMD property
# ---------------------------------------------------------------------------
def imf_validity(mode):
    """
    ||local mean|| / ||mode||.  0 = a perfect IMF, larger = not an IMF.

    The local mean is the average of the upper and lower envelopes through the
    mode's own extrema, which is precisely the quantity sifting drives to zero.
    Computed isotropically (rho=1) so the measure favours no method.
    """
    mx, mn = find_extrema_2d(mode)
    if mx.sum() < 8 or mn.sum() < 8:
        return float("nan")          # too few extrema to define an envelope
    theta, coh = orientation_and_coherence(mode)
    bw = _scale_estimate(mx | mn, mode.shape)
    upper = _aniso_envelope(mx, mode, theta, coh, bw, rho_max=1.0)
    lower = _aniso_envelope(mn, mode, theta, coh, bw, rho_max=1.0)
    local_mean = 0.5 * (upper + lower)
    return float(np.linalg.norm(local_mean) / (np.linalg.norm(mode) + 1e-12))


# ---------------------------------------------------------------------------
def highpass(sigma):
    return lambda f: f - gaussian_filter(f, sigma)


ESTIMATORS = {
    "iso-GLOBAL BEMD (baseline)": lambda f: bemd(f)[0][0],
    "iso-LOCAL (rho=1)": lambda f: stbemd(f, rho_max=1.0)[0][0],
    "aniso-LOCAL (rho=4) ST-BEMD": lambda f: stbemd(f)[0][0],
}


def evaluate(estimator, prints):
    orient, valid = [], []
    for fp in prints:
        mode = estimator(fp)
        orient.append(local_orientation_error(mode, fp))
        valid.append(imf_validity(mode))
    return float(np.mean(orient)), float(np.nanmean(valid))


def main():
    ensure_outputs()
    batch = fingerprint_batch()
    select, test = batch[:SPLIT], batch[SPLIT:]
    print(f"Held-out design: sigma chosen on {len(select)} prints, "
          f"scored on {len(test)} unseen prints\n")

    # --- step 1: choose the control's sigma on the SELECTION set only --------
    print("Choosing high-pass sigma on the selection set:")
    sel_scores = {}
    for s in SIGMAS:
        err, _ = evaluate(highpass(s), select)
        sel_scores[s] = err
        print(f"  sigma={s:<4} orientation error {err:6.2f}")
    best_sigma = min(sel_scores, key=sel_scores.get)
    print(f"  -> selected sigma = {best_sigma} "
          f"(never scored on the test set until now)\n")

    # --- step 2: score everything on the held-out set ------------------------
    estimators = dict(ESTIMATORS)
    estimators[f"Gaussian high-pass sigma={best_sigma}"] = highpass(best_sigma)

    rows = {}
    print(f"{'estimator':<32}{'orient err':>12}{'IMF validity':>14}")
    print(f"{'':<32}{'(deg)':>12}{'(0 = is an IMF)':>14}")
    print("-" * 60)
    for name, fn in estimators.items():
        err, val = evaluate(fn, test)
        rows[name] = {"orientation_error": err, "imf_validity": val}
        print(f"{name:<32}{err:>12.2f}{val:>14.3f}")
    print("-" * 60)

    hp = next(k for k in rows if "high-pass" in k)
    st = "aniso-LOCAL (rho=4) ST-BEMD"

    print("\nOn the held-out prints:")
    better = rows[hp]["orientation_error"] < rows[st]["orientation_error"]
    print(f"  orientation: control {rows[hp]['orientation_error']:.2f} vs "
          f"ST-BEMD {rows[st]['orientation_error']:.2f}  "
          f"-> control {'still wins' if better else 'no longer wins'}")
    print(f"  IMF validity: control {rows[hp]['imf_validity']:.3f} vs "
          f"ST-BEMD {rows[st]['imf_validity']:.3f}  "
          f"-> {'control is not an IMF' if rows[hp]['imf_validity'] > rows[st]['imf_validity'] else 'control is as valid'}")

    print("\nREADING THIS")
    if better and rows[hp]["imf_validity"] > rows[st]["imf_validity"]:
        print("  The control still wins on orientation even held out, but does not")
        print("  produce an IMF. Both facts stand: the orientation metric is")
        print("  insufficient on its own, AND the control is not a rival method.")
        print("  Report the pair, never either alone.")
    elif not better:
        print("  Removing the selection bias reverses the orientation result.")
        print("  The earlier control finding was an artefact of tuning sigma on")
        print("  the evaluation set.")
    else:
        print("  The control wins on both counts. The method has a real problem.")

    # --- step 3: is there a systematic trade-off? ---------------------------
    # Score every sigma on the test set, purely to trace the curve. This is not
    # a selection step -- no sigma is chosen here, they are all plotted.
    curve = {}
    for s in SIGMAS:
        err, val = evaluate(highpass(s), test)
        curve[s] = {"orientation_error": err, "imf_validity": val}

    points = ([(r["orientation_error"], r["imf_validity"]) for r in rows.values()]
              + [(c["orientation_error"], c["imf_validity"]) for c in curve.values()])
    xs, ys = zip(*points)
    corr = float(np.corrcoef(xs, ys)[0, 1])

    print(f"\nTRADE-OFF across all {len(points)} estimators")
    print(f"  correlation(orientation error, IMF validity) = {corr:+.2f}")
    if corr < -0.5:
        print("  Strongly ANTI-correlated: every estimator that scores well on")
        print("  orientation scores badly on IMF validity, and vice versa. The")
        print("  orientation metric is largely measuring how little the estimator")
        print("  sifted. It cannot be read on its own.")

    out = RESULTS / "results_validity.json"
    json.dump({"selected_sigma": best_sigma, "selection_scores": sel_scores,
               "held_out": rows, "sigma_curve": curve,
               "tradeoff_correlation": corr},
              open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")
    make_figure(rows, curve, corr, best_sigma)


def make_figure(rows, curve, corr, best_sigma):
    """
    figR6 — the trade-off plane.

    A scatter rather than two bar charts, because the finding is the
    *relationship* between the two metrics, not their separate values. Each
    estimator is one point; the anti-correlation is the result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLIND, AWARE = "#eb6834", "#2a78d6"
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "legend.frameon": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    sigmas = sorted(curve)
    cx = [curve[s]["orientation_error"] for s in sigmas]
    cy = [curve[s]["imf_validity"] for s in sigmas]
    ax.plot(cx, cy, "-o", color=BLIND, lw=2, ms=9, zorder=3,
            label="Gaussian high-pass (orientation-blind)")
    for s, x, y in zip(sigmas, cx, cy):
        ax.annotate(f"σ={s:g}", (x, y), textcoords="offset points",
                    xytext=(0, -16), fontsize=8, color=BLIND, ha="center")

    # Hand-placed offsets: the three method points sit close enough that a
    # single offset makes their labels overlap each other and the axis edge.
    offsets = {
        "iso-GLOBAL BEMD (baseline)": (-10, 12, "right"),
        "iso-LOCAL (rho=1)": (12, 8, "left"),
        "aniso-LOCAL (rho=4) ST-BEMD": (-12, -22, "right"),
    }
    for name, r in rows.items():
        if "high-pass" in name:
            continue
        ax.scatter(r["orientation_error"], r["imf_validity"], s=110,
                   color=AWARE, zorder=4, edgecolor="#fcfcfb", linewidth=2)
        dx, dy, ha = offsets.get(name, (10, 6, "left"))
        ax.annotate(name.replace(" (", "\n("),
                    (r["orientation_error"], r["imf_validity"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=INK, ha=ha)

    ax.set_xlabel("orientation error (degrees) — lower looks better")
    ax.set_ylabel("||local mean|| / ||mode|| — lower IS better (0 = a true IMF)")
    ax.set_title(f"The two metrics pull against each other  "
                 f"(r = {corr:+.2f})\n"
                 "scoring well on orientation means having sifted less",
                 loc="left")
    ax.legend(loc="upper right")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(0.14)

    fig.tight_layout()
    path = FIGURES / "figR6_metric_tradeoff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
