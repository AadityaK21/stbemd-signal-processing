"""
control_smoothing.py
--------------------
Does the fingerprint metric measure ORIENTATION RECOVERY, or just SMOOTHNESS?

The concern
-----------
`local_orientation_error(imf, f)` compares the structure-tensor orientation of an
IMF against that of the input, weighted by the input's coherence. It is
self-referential: it never consults a true orientation, only the input's own
estimate. Two things follow immediately, and the second is the worry:

  1. Its global optimum is the IDENTITY. local_orientation_error(f, f) == 0.
     A "method" that returns the input unchanged scores perfectly. So the metric
     cannot reward extraction; it can only penalise disturbance.
  2. Therefore any operation that disturbs the orientation field gently will
     score well, whether or not it knows anything about orientation.

If that is what is happening, then ST-BEMD's 13.71 -> 8.25 improvement says its
envelope is gentler than a global RBF interpolant, not that it recovers
orientation better. Those are very different claims.

The control
-----------
A Gaussian high-pass, `imf = f - G_sigma * f`. It has no extrema detection, no
envelope, no sifting, no structure tensor -- no orientation input of any kind.
Reconstruction is exact by construction. If this scores as well as ST-BEMD on
the fingerprint metric, the metric is measuring smoothness.

The decisive part is running BOTH metrics on the same estimators:

  fingerprint metric  -- self-referential, orientation of input as reference
  analytic metric     -- closed-form theta, a real ground truth

A method that genuinely recovers orientation should do well on both. A method
that merely band-passes gently should do well on the first and badly on the
second. The gap between the two columns is the answer.

    python analysis/control_smoothing.py
"""
import _bootstrap  # noqa: F401  (puts the source dirs on sys.path)

import json
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from bemd import bemd
from stbemd import stbemd
from metrics import local_orientation_error
from structure_tensor import orientation_and_coherence
import analytic_signals as A
from _paths import RESULTS, ensure_outputs, fingerprint_batch

warnings.filterwarnings("ignore")

SIGMAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N_PRINTS = 30


# ---------------------------------------------------------------------------
# the estimators
# ---------------------------------------------------------------------------
def gaussian_highpass(f, sigma):
    """
    The control. imf = f - (f smoothed). Exact reconstruction, zero orientation
    awareness. This is deliberately not an EMD -- that is the entire point.
    """
    low = gaussian_filter(f, sigma)
    return [f - low], low


def identity(f):
    """
    The reductio: return the input as 'IMF-1'. Scores a perfect 0.00 on the
    fingerprint metric while extracting nothing at all. Included so the metric's
    degenerate optimum is visible as a number rather than an argument.
    """
    return [f.copy()], np.zeros_like(f)


# ---------------------------------------------------------------------------
# the two metrics
# ---------------------------------------------------------------------------
def fold(a):
    return np.minimum(np.abs(a), np.pi - np.abs(a))


def wmean(x, w):
    w = np.where(w > 0.05, w, 0.0)
    return float(np.sum(w * x) / (np.sum(w) + 1e-12))


def analytic_error(imf, theta_true, weight):
    """Orientation error against a CLOSED-FORM theta. No self-reference."""
    th_imf, _ = orientation_and_coherence(imf)
    return wmean(np.rad2deg(fold(th_imf - theta_true)), weight)


def score_fingerprints(estimator, batch):
    """The self-referential metric, over the print batch."""
    errs = [local_orientation_error(estimator(fp)[0][0], fp) for fp in batch]
    return float(np.mean(errs)), float(np.std(errs))


def score_analytic(estimator):
    """The closed-form metric, averaged over the analytic suite."""
    errs = []
    for _name, gen in A.SUITE.items():
        d = gen(96)
        f, theta, amp = d["f"], d["theta"], d["amp"]
        _, coh = orientation_and_coherence(f)
        weight = coh * (amp > 0.3 * amp.max())
        errs.append(analytic_error(estimator(f)[0][0], theta, weight))
    return float(np.mean(errs)), float(np.std(errs))


# ---------------------------------------------------------------------------
def main():
    ensure_outputs()
    batch = fingerprint_batch()[:N_PRINTS]
    print(f"Control experiment — {len(batch)} prints, {len(A.SUITE)} analytic signals\n")

    estimators = [
        ("identity (returns input)", identity),
        *[(f"Gaussian high-pass sigma={s}",
           (lambda s: lambda f: gaussian_highpass(f, s))(s)) for s in SIGMAS],
        ("iso-GLOBAL BEMD (baseline)", bemd),
        ("iso-LOCAL  (rho=1)", lambda f: stbemd(f, rho_max=1.0)),
        ("aniso-LOCAL(rho=4)  ST-BEMD", stbemd),
    ]

    print(f"{'estimator':<30}{'fingerprint':>14}{'analytic GT':>14}   verdict")
    print(f"{'':<30}{'(self-ref)':>14}{'(closed form)':>14}")
    print("-" * 78)

    rows = {}
    for name, fn in estimators:
        fp_mean, fp_std = score_fingerprints(fn, batch)
        an_mean, an_std = score_analytic(fn)
        rows[name] = {"fingerprint_mean": fp_mean, "fingerprint_std": fp_std,
                      "analytic_mean": an_mean, "analytic_std": an_std}
        print(f"{name:<30}{fp_mean:>10.2f}    {an_mean:>10.2f}    ", end="")
        print("orientation-blind" if "high-pass" in name or "identity" in name
              else "orientation-aware")

    print("-" * 78)

    st = rows["aniso-LOCAL(rho=4)  ST-BEMD"]
    iso = rows["iso-GLOBAL BEMD (baseline)"]
    best_hp_name = min((n for n in rows if "high-pass" in n),
                       key=lambda n: rows[n]["fingerprint_mean"])
    hp = rows[best_hp_name]

    print(f"\nBest orientation-BLIND control: {best_hp_name}")
    print(f"  fingerprint metric : {hp['fingerprint_mean']:.2f} deg   "
          f"(ST-BEMD {st['fingerprint_mean']:.2f}, baseline {iso['fingerprint_mean']:.2f})")
    print(f"  analytic metric    : {hp['analytic_mean']:.2f} deg   "
          f"(ST-BEMD {st['analytic_mean']:.2f}, baseline {iso['analytic_mean']:.2f})")

    print("\nVERDICT")
    if hp["fingerprint_mean"] <= st["fingerprint_mean"]:
        print("  A method with NO orientation input matches or beats ST-BEMD on the")
        print("  fingerprint metric. That metric cannot be read as evidence of")
        print("  orientation recovery — it rewards gentle band-passing.")
    elif hp["fingerprint_mean"] < iso["fingerprint_mean"]:
        print("  The orientation-blind control beats the isotropic BEMD baseline but")
        print("  not ST-BEMD. So part of the reported gain is explained by smoothness")
        print(f"  alone: the control closes "
              f"{(iso['fingerprint_mean'] - hp['fingerprint_mean']) / (iso['fingerprint_mean'] - st['fingerprint_mean']) * 100:.0f}% "
              f"of the baseline->ST gap with no orientation input.")
    else:
        print("  The orientation-blind control does NOT beat the baseline. The")
        print("  fingerprint metric survives this challenge.")

    out = RESULTS / "results_control.json"
    json.dump(rows, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")

    make_figure(rows)


def make_figure(rows):
    """
    figR5 — the two metrics side by side.

    Two panels, not one chart with two y-axes: the scales differ by a factor of
    thirty, and overlaying them would manufacture a visual comparison the data
    does not support. Colour encodes whether the estimator can see orientation
    at all, which is the variable under test.
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
        "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })

    names = list(rows)
    blind = ["high-pass" in n or "identity" in n for n in names]
    colours = [BLIND if b else AWARE for b in blind]
    y = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)

    for ax, key, title, note in (
        (ax1, "fingerprint_mean",
         "Fingerprint metric (self-referential)",
         "orientation-blind controls WIN"),
        (ax2, "analytic_mean",
         "Analytic ground truth (closed form)",
         "everything ties"),
    ):
        values = [rows[n][key] for n in names]
        bars = ax.barh(y, values, color=colours, zorder=2, height=0.68)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color=INK)
        ax.set_xlabel("orientation error (degrees)")
        ax.set_title(f"{title}\n{note}", loc="left")
        ax.grid(True, axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.16)

    ax1.set_yticks(y, names)
    ax1.invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1, color=AWARE),
               plt.Rectangle((0, 0), 1, 1, color=BLIND)]
    # Upper right: the identity row is 0.00, so that corner is empty. Lower
    # right would sit on top of the ST-BEMD bar's value label.
    ax1.legend(handles, ["sees orientation", "orientation-blind"],
               loc="upper right")

    fig.suptitle("A method with no orientation input beats ST-BEMD on the "
                 "fingerprint metric, and ties on real ground truth",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    from _paths import FIGURES
    path = FIGURES / "figR5_metric_control.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
