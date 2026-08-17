"""
independent_eval.py
-------------------
The non-circular real-data evaluation. This is the experiment the project was
missing.

The problem with every earlier real-data number
-----------------------------------------------
`local_orientation_error(imf, f)` scores a mode against the orientation of the
same image the mode came from. Two defects follow:

  (a) SHARED MACHINERY. The reference is a structure tensor, and ST-BEMD steers
      its kernel with a structure tensor. The method is being scored by its own
      instrument.
  (b) DEGENERATE OPTIMUM. The identity map scores exactly 0.00. Doing nothing
      wins, so the metric rewards not-sifting rather than orientation recovery
      (see control_smoothing.py and metric_validity.py: r = -0.94 against IMF
      validity).

Swapping the structure tensor for a Gabor bank fixes (a) and NOT (b) — the
identity would still score perfectly under any self-referential comparison. The
instrument was never the whole problem; the protocol was.

The protocol
------------
Score against the orientation of a DIFFERENT IMAGE than the one the method saw:

    reference  theta*  =  orientation of the CLEAN print
    method input       =  the print plus noise
    score              =  angular error of IMF-1 against theta*

Now doing nothing loses: the identity returns the noisy field, whose
orientation differs from the clean reference. Winning requires actually
recovering the underlying ridge orientation through the noise, which is the
capability the project claims. The identity is kept in the line-up precisely as
the check that the protocol bites.

Both defects are addressed at once by taking theta* from `gabor_orientation`, a
matched-filter estimator sharing no machinery with the structure tensor.

Caveat, stated up front: the Gabor reference is itself an estimate, and on the
analytic suite its own error is 0.3-3.2 deg (mean ~1.4) against the structure
tensor's 0.3-0.5. It is the more INDEPENDENT instrument, not the more accurate
one. Differences below ~1.5 deg here mean nothing. The results are reported
under both instruments so that instrument-specific bias is visible.

    python analysis/independent_eval.py
"""
import _bootstrap  # noqa: F401  (puts the source dirs on sys.path)

import json
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

import gabor_orientation as gabor
from bemd import bemd
from stbemd import stbemd
from structure_tensor import orientation_and_coherence
from _paths import FIGURES, RESULTS, ensure_outputs, fingerprint_batch

warnings.filterwarnings("ignore")

N_PRINTS = 12
SNR_DB = [20, 10, 5]
HP_SIGMA = 4.0          # the sigma chosen on the held-out half in metric_validity.py

# Prints are upsampled before anything runs. SOCOFing ridges are ~3.2 px at
# native 90x90, which is close enough to the sampling limit that a matched
# filter cannot be angularly selective: at 1x the Gabor bank and the structure
# tensor disagree by 7.6 deg on the SAME clean image, which swamps the 1-5 deg
# differences between methods. At 2x (ridges ~6.5 px) that disagreement halves
# to 4.0 deg. Upsampling does not change the orientation field, only how well
# either instrument can resolve it.
UPSAMPLE = 2

# Each estimator returns a LIST of candidate modes; select_ridge_mode picks one.
#
# Why a selection step at all. EMD's first IMF is the highest-frequency
# component, and on a noisy input that is the NOISE, not the ridges. Scoring
# IMF-1 against clean ridge orientation therefore measures nothing: an earlier
# version of this script did exactly that and every real method landed at
# 27-36 deg, close to the 45 deg random floor, while the identity "won" purely
# because a noisy print still contains its ridges. The question worth asking is
# whether a method can ISOLATE the ridge layer out of the noise, so the mode
# carrying ridge-scale content is the one to score.
ESTIMATORS = {
    "identity (does nothing)": lambda f: [f.copy()],
    f"Gaussian high-pass s={HP_SIGMA:g}": lambda f: [f - gaussian_filter(f, HP_SIGMA)],
    "iso-GLOBAL BEMD": lambda f: bemd(f)[0],
    "iso-LOCAL (rho=1)": lambda f: stbemd(f, rho_max=1.0)[0],
    "ST-BEMD (rho=4)": lambda f: stbemd(f)[0],
}


def select_ridge_mode(modes, target_wavelength):
    """
    Pick the mode whose dominant wavelength is closest to the input's own ridge
    wavelength, compared in log-scale so a factor of two costs the same either
    way.

    This uses only the NOISY input's spectrum — never the clean reference — so
    it is not oracle selection. Every estimator gets the same rule, and
    single-mode estimators (identity, high-pass) trivially return their one
    mode, so nobody is advantaged by the step itself.
    """
    if len(modes) == 1:
        return modes[0]
    target = np.log(max(target_wavelength, 1e-6))
    costs = [abs(np.log(max(gabor.dominant_wavelength(m), 1e-6)) - target)
             for m in modes]
    return modes[int(np.argmin(costs))]


def fold(a):
    return np.minimum(np.abs(a), np.pi - np.abs(a))


def wmean(x, w):
    w = np.where(w > 0.05, w, 0.0)
    return float(np.sum(w * x) / (np.sum(w) + 1e-12))


def add_noise(f, snr_db, rng):
    power = f.var() / (10 ** (snr_db / 10.0))
    return f + rng.normal(0.0, np.sqrt(power), f.shape)


def run():
    """
    Decompose once, measure with both instruments.

    'gabor' is the independent one; 'tensor' is the project's usual instrument,
    reported as a robustness check. Neither is self-referential here, because
    both take their reference from the CLEAN print while the methods see noise.

    The Gabor bank is pinned to the wavelengths of the clean reference and those
    same filters measure every mode. Letting it retune per image would compare
    the two sides of the subtraction with different instruments — an earlier
    version did that and inflated every error by roughly 8 deg.
    """
    batch = fingerprint_batch()[:N_PRINTS]
    rng = np.random.default_rng(0)
    scores = {inst: {name: {snr: [] for snr in SNR_DB} for name in ESTIMATORS}
              for inst in ("gabor", "tensor")}

    for raw in batch:
        clean = zoom(raw, UPSAMPLE, order=3) if UPSAMPLE > 1 else raw
        lam = gabor.dominant_wavelength(clean)
        lams = [lam * 0.75, lam, lam * 1.33]

        gabor_ref, gabor_w = gabor.orientation(clean, wavelengths=lams)
        tensor_ref, tensor_w = orientation_and_coherence(clean)

        for snr in SNR_DB:
            noisy = add_noise(clean, snr, rng)
            target = gabor.dominant_wavelength(noisy)
            for name, fn in ESTIMATORS.items():
                modes = fn(noisy)
                if not modes:
                    continue
                mode = select_ridge_mode(modes, target)

                th, _ = gabor.orientation(mode, wavelengths=lams)
                scores["gabor"][name][snr].append(
                    np.rad2deg(wmean(fold(th - gabor_ref), gabor_w)))

                th, _ = orientation_and_coherence(mode)
                scores["tensor"][name][snr].append(
                    np.rad2deg(wmean(fold(th - tensor_ref), tensor_w)))

    return {inst: {name: {snr: float(np.mean(v)) for snr, v in per.items()}
                   for name, per in tab.items()}
            for inst, tab in scores.items()}


def report(title, table):
    print(f"\n{title}")
    print(f"{'estimator':<28}" + "".join(f"{f'{s} dB':>10}" for s in SNR_DB)
          + f"{'mean':>10}")
    print("-" * (28 + 10 * (len(SNR_DB) + 1)))
    order = sorted(table, key=lambda n: np.mean(list(table[n].values())))
    for name in order:
        row = table[name]
        cells = "".join(f"{row[s]:>10.2f}" for s in SNR_DB)
        print(f"{name:<28}{cells}{np.mean(list(row.values())):>10.2f}")
    return order


def main():
    ensure_outputs()
    print(f"Non-circular evaluation — {N_PRINTS} prints, reference from the CLEAN "
          f"print,\nmethods see the NOISY print. Identity included as the check "
          f"that the protocol bites.")

    tables = run()
    gabor_table, tensor_table = tables["gabor"], tables["tensor"]
    order_g = report("Measured with the GABOR bank (independent of the method)",
                     gabor_table)
    order_t = report("Measured with the STRUCTURE TENSOR (robustness check)",
                     tensor_table)

    print("\n" + "=" * 78)
    print("READING THIS")
    print("=" * 78)

    ident = "identity (does nothing)"
    hp = next(n for n in ESTIMATORS if "high-pass" in n)
    st = "ST-BEMD (rho=4)"
    iso = "iso-GLOBAL BEMD"

    g_mean = {n: np.mean(list(gabor_table[n].values())) for n in gabor_table}

    print(f"\n1. Does the protocol bite? identity now scores "
          f"{g_mean[ident]:.2f} deg (it scored 0.00 under the old metric).")
    print("   " + ("Yes — doing nothing no longer wins."
                   if g_mean[ident] > g_mean[st] else
                   "NO — doing nothing still wins; the protocol has not fixed it."))

    print(f"\n2. Does the orientation-blind control still beat the method? "
          f"control {g_mean[hp]:.2f} vs ST-BEMD {g_mean[st]:.2f}")
    print("   " + ("The control still wins. The method has no real-data claim."
                   if g_mean[hp] < g_mean[st] else
                   "No — under a non-circular protocol ST-BEMD beats it. The "
                   "earlier\n   control result was an artefact of the metric, not "
                   "a property of the method."))

    print("\n3. Does ST-BEMD beat its own baseline, and does it depend on noise?")
    print(f"   {'SNR':>6}{'iso-GLOBAL':>13}{'ST-BEMD':>10}{'gap':>8}   "
          f"{'(tensor gap)':>14}")
    for snr in SNR_DB:
        gap_g = gabor_table[iso][snr] - gabor_table[st][snr]
        gap_t = tensor_table[iso][snr] - tensor_table[st][snr]
        flag = "" if max(gap_g, gap_t) > 1.5 else "  (under noise floor)"
        print(f"   {snr:>4} dB{gabor_table[iso][snr]:>13.2f}"
              f"{gabor_table[st][snr]:>10.2f}{gap_g:>8.2f}   {gap_t:>13.2f}{flag}")
    hard = SNR_DB[-1]
    gap_hard = gabor_table[iso][hard] - gabor_table[st][hard]
    if gap_hard > 1.5:
        print(f"\n   The advantage GROWS with noise and clears the ~1.5 deg")
        print(f"   instrument floor at {hard} dB ({gap_hard:.2f} deg, and "
              f"{tensor_table[iso][hard] - tensor_table[st][hard]:.2f} under the")
        print("   other instrument). This is the project's first non-circular")
        print("   real-data result that favours the method.")
    else:
        print(f"\n   The gap stays under the instrument floor at every SNR. "
              "No claim available.")

    print(f"\n4. Do the two instruments agree on the ranking? "
          f"{'yes' if order_g == order_t else 'NO — instrument-dependent'}")

    out = RESULTS / "results_independent.json"
    json.dump({"gabor": gabor_table, "tensor": tensor_table,
               "n_prints": N_PRINTS, "snr_db": SNR_DB, "upsample": UPSAMPLE},
              open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")
    make_figure(gabor_table)


def make_figure(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PALETTE = {"identity (does nothing)": "#898781",
               f"Gaussian high-pass s={HP_SIGMA:g}": "#eb6834",
               "iso-GLOBAL BEMD": "#1baf7a",
               "iso-LOCAL (rho=1)": "#eda100",
               "ST-BEMD (rho=4)": "#2a78d6"}
    INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "legend.frameon": False,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for name, row in table.items():
        values = [row[s] for s in SNR_DB]
        ax.plot(SNR_DB, values, "-o", lw=2, ms=8, color=PALETTE[name],
                label=name, zorder=3)
        ax.annotate(f"{values[-1]:.1f}", (SNR_DB[-1], values[-1]),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=PALETTE[name])

    ax.invert_xaxis()
    ax.set_xticks(SNR_DB, [f"{s} dB" for s in SNR_DB])
    ax.set_xlabel("input SNR — noisier to the right")
    ax.set_ylabel("orientation error vs the CLEAN print (degrees)")
    ax.set_title("Non-circular protocol: reference from the clean print,\n"
                 "methods see the noisy one", loc="left")
    ax.legend(loc="upper left")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)

    fig.tight_layout()
    path = FIGURES / "figR7_independent_eval.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
