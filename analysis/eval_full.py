"""
eval_full.py
------------
Expanded evaluation:
  (1) MANY fingerprints with mean+-std and significance test
  (2) orientation error binned by local orientation curvature kappa + improvement-vs-kappa
  (3) analytic-ground-truth synthetic suite (no structure-tensor self-reference)
  (4) robustness to additive noise
  (5) runtime vs image size
Writes outputs/results/results_full.json and four figures:
  figR1_fingerprints.png  per-print paired comparison + curvature bins
  figR2_synthetic.png     analytic-ground-truth suite
  figR3_noise.png         robustness to additive noise
  figR4_runtime.png       runtime vs image size (log-log)

Needs data/fp_batch.npy -- run scripts/fetch_data.py first.

Key honest findings (30 prints, regenerated 2026-08-17):
  iso 13.71 +/- 6.16 -> ST 8.25 +/- 3.14, ST better on 100%, Wilcoxon p=1.9e-9
  BUT the proper ablation (iso-LOCAL rho=1 vs aniso-LOCAL rho=4, see ablation.py)
  shows most of the gain is the local-vs-global ENVELOPE (~4.29 deg); anisotropy
  itself contributes ~1.18 deg, and its benefit SHRINKS as curvature rises
  (+1.42 low, +1.13 medium, +0.20 high) -- the opposite of what the proposed
  mechanism predicts.
"""
import _bootstrap  # noqa: F401  (puts the source dirs on sys.path)
import json, time, warnings
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from structure_tensor import orientation_and_coherence
from bemd import bemd
from stbemd import stbemd
import analytic_signals as A
from _paths import FIGURES, RESULTS, fingerprint_batch

warnings.filterwarnings("ignore")
FIG = str(FIGURES); RES = str(RESULTS)

# --- figure styling --------------------------------------------------------
# Two series throughout (isotropic baseline vs the proposed method), drawn in
# the first two categorical slots. The pair clears CVD and normal-vision
# separation on a light surface, so the comparison survives colour-blind
# readers and greyscale printing; the legend and direct labels mean colour is
# never the only channel carrying identity.
ISO_C, ST_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)

def fold(a): return np.minimum(np.abs(a), np.pi - np.abs(a))
def orient_err_map(imf, ref_theta):
    th_i, _ = orientation_and_coherence(imf)
    return np.rad2deg(fold(th_i - ref_theta))
def orientation_curvature(theta):
    c2, s2 = np.cos(2*theta), np.sin(2*theta)
    cy, cx = np.gradient(c2); sy, sx = np.gradient(s2)
    return 0.5*np.sqrt(cx**2 + cy**2 + sx**2 + sy**2)
def wmean(x, w):
    w = np.where(w > 0.05, w, 0.0)
    return float(np.sum(w*x)/(np.sum(w)+1e-12))


def eval_fingerprints():
    batch = fingerprint_batch()
    per = {"iso": [], "st": []}
    kap_all, ei_all, es_all, w_all = [], [], [], []
    nimf = {"iso": [], "st": []}
    for fp in batch:
        th_ref, coh = orientation_and_coherence(fp)
        kap = orientation_curvature(th_ref)
        ib, _ = bemd(fp); sb, _ = stbemd(fp)
        ei = orient_err_map(ib[0], th_ref); es = orient_err_map(sb[0], th_ref)
        per["iso"].append(wmean(ei, coh)); per["st"].append(wmean(es, coh))
        nimf["iso"].append(len(ib)); nimf["st"].append(len(sb))
        m = coh > 0.05
        kap_all.append(kap[m]); ei_all.append(ei[m]); es_all.append(es[m]); w_all.append(coh[m])
    kap_all = np.concatenate(kap_all); ei = np.concatenate(ei_all)
    es = np.concatenate(es_all); w = np.concatenate(w_all)
    a = np.array(per["iso"]); b = np.array(per["st"])
    summary = {"n_prints": len(batch), "iso_mean": float(a.mean()), "iso_std": float(a.std()),
               "st_mean": float(b.mean()), "st_std": float(b.std()),
               "pct_st_better": float(np.mean(b < a)*100),
               "paired_t_p": float(stats.ttest_rel(a, b).pvalue),
               "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
               "iso_nimf": float(np.mean(nimf["iso"])), "st_nimf": float(np.mean(nimf["st"])),
               "per_print_iso": a.tolist(), "per_print_st": b.tolist()}
    q1, q2 = np.quantile(kap_all, [1/3, 2/3])
    curv = {}
    for name, mask in [("low", kap_all <= q1), ("medium", (kap_all > q1) & (kap_all <= q2)),
                       ("high", kap_all > q2)]:
        curv[name] = {"iso": wmean(ei[mask], w[mask]), "st": wmean(es[mask], w[mask])}
    return summary, curv


def eval_synthetic():
    rows = {}
    for name, gen in A.SUITE.items():
        d = gen(96); f, carrier, bg = d["f"], d["carrier"], d["bg"]
        th, amp = d["theta"], d["amp"]
        _, coh = orientation_and_coherence(f)
        wgt = coh*(amp > 0.3*amp.max())
        r = {}
        for tag, fn in [("iso", bemd), ("st", stbemd)]:
            i1 = fn(f)[0][0]
            th_i, _ = orientation_and_coherence(i1)
            r[tag+"_orient_err"] = wmean(np.rad2deg(fold(th_i - th)), wgt)
            r[tag+"_imf_recov"] = float(np.linalg.norm(i1-carrier)/(np.linalg.norm(carrier)+1e-12))
            if bg is not None:
                bb = bg-bg.mean(); ii = i1-i1.mean()
                r[tag+"_modemix"] = float(abs(bb.ravel()@ii.ravel())/(np.linalg.norm(bb)*np.linalg.norm(ii)+1e-12))
        rows[name] = r
    return rows


def eval_noise():
    base = A.concentric_core(96, freq=9.0, am=False, background=False)
    f0, th = base["f"], base["theta"]; rng = np.random.default_rng(0)
    out = {"snr_db": [], "iso": [], "st": []}
    for snr in [40, 25, 20, 15, 10, 5]:
        noise_p = f0.var()/(10**(snr/10))
        f = f0 + rng.normal(0, np.sqrt(noise_p), f0.shape)
        _, coh = orientation_and_coherence(f)
        out["snr_db"].append(snr)
        for tag, fn in [("iso", bemd), ("st", stbemd)]:
            i1 = fn(f)[0][0]; th_i, _ = orientation_and_coherence(i1)
            out[tag].append(wmean(np.rad2deg(fold(th_i-th)), coh))
    return out


def eval_runtime():
    out = {"size": [], "iso_s": [], "st_s": []}
    for N in [64, 96, 128, 192, 256]:
        f = A.concentric_core(N, freq=N/12, am=True, background=True)["f"]
        t = time.time(); bemd(f); ti = time.time()-t
        t = time.time(); stbemd(f); ts = time.time()-t
        out["size"].append(N); out["iso_s"].append(ti); out["st_s"].append(ts)
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_fingerprints(summary, curv):
    """
    figR1 — the real-data result, both per-print and binned by curvature.

    Left: one row per print, sorted by baseline error, with a segment joining
    the two methods. A paired design is the honest picture here because the
    same print is scored twice; two bare means would hide that the improvement
    is near-universal rather than driven by a few easy prints.

    Right: the same comparison split by local orientation curvature, which is
    where the proposed mechanism predicts the largest gain.
    """
    iso = np.array(summary["per_print_iso"]); st = np.array(summary["per_print_st"])
    order = np.argsort(iso)[::-1]
    iso, st = iso[order], st[order]
    y = np.arange(len(iso))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})

    ax1.hlines(y, st, iso, color=GRID, lw=2, zorder=1)
    ax1.scatter(iso, y, s=34, color=ISO_C, zorder=3, label="isotropic BEMD")
    ax1.scatter(st, y, s=34, color=ST_C, zorder=3, label="ST-BEMD")
    ax1.set_yticks([]); ax1.invert_yaxis()
    ax1.set_xlabel("orientation error (degrees)")
    ax1.set_ylabel(f"{len(iso)} prints, worst baseline first")
    ax1.set_title(f"Every print improves  "
                  f"({summary['pct_st_better']:.0f}% better, "
                  f"Wilcoxon p = {summary['wilcoxon_p']:.0e})", loc="left")
    ax1.legend(loc="lower right")
    _grid(ax1, axis="x")

    for value, colour, label in ((summary["iso_mean"], ISO_C, "iso"),
                                 (summary["st_mean"], ST_C, "ST")):
        ax1.axvline(value, color=colour, ls=":", lw=1.2, zorder=2)
        ax1.annotate(f"{label} mean {value:.1f}", (value, len(iso) - 0.5),
                     textcoords="offset points", xytext=(3, 4),
                     color=colour, fontsize=8)

    bins = ["low", "medium", "high"]
    x = np.arange(len(bins)); w = 0.36
    b1 = ax2.bar(x - w/2 - 0.01, [curv[b]["iso"] for b in bins], w,
                 color=ISO_C, label="isotropic BEMD", zorder=2)
    b2 = ax2.bar(x + w/2 + 0.01, [curv[b]["st"] for b in bins], w,
                 color=ST_C, label="ST-BEMD", zorder=2)
    for bars in (b1, b2):
        ax2.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK)
    ax2.set_xticks(x, [f"{b}\ncurvature" for b in bins])
    ax2.set_ylabel("orientation error (degrees)")
    ax2.set_title("Both methods degrade as ridges curve", loc="left")
    ax2.legend(loc="upper left")
    _grid(ax2)

    fig.tight_layout()
    fig.savefig(f"{FIG}/figR1_fingerprints.png", bbox_inches="tight")
    plt.close(fig)


def fig_synthetic(syn):
    """
    figR2 — the same comparison against ANALYTIC ground truth.

    The fingerprint metric is coherence-weighted self-reference: it asks whether
    the IMF kept the input's orientation, not whether that orientation is right.
    These signals carry a closed-form theta, so the error here is absolute.
    """
    names = list(syn)
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 3.2, 4.6))

    b1 = ax.bar(x - w/2 - 0.01, [syn[n]["iso_orient_err"] for n in names], w,
                color=ISO_C, label="isotropic BEMD", zorder=2)
    b2 = ax.bar(x + w/2 + 0.01, [syn[n]["st_orient_err"] for n in names], w,
                color=ST_C, label="ST-BEMD", zorder=2)
    for bars in (b1, b2):
        # Two decimals: the differences here are hundredths of a degree, and
        # one decimal would print every pair as identical — flattering to the
        # proposed method by hiding that it does not win on this test.
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=INK)

    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylabel("orientation error vs analytic truth (degrees)")
    ax.set_title("Against closed-form ground truth the two methods tie",
                 loc="left")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/figR2_synthetic.png", bbox_inches="tight")
    plt.close(fig)


def fig_noise(noise):
    """figR3 — how far the gap survives additive noise."""
    snr = noise["snr_db"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for values, colour, label, dy in ((noise["iso"], ISO_C, "isotropic BEMD", 9),
                                      (noise["st"], ST_C, "ST-BEMD", -16)):
        ax.plot(snr, values, "-o", color=colour, lw=2, ms=8, label=label, zorder=3)
        # Label the noisiest point only. Two decimals and opposite vertical
        # offsets — at one decimal the two series both print "0.8" on top of
        # each other, which reads as a tie that the numbers do not support.
        ax.annotate(f"{values[-1]:.2f}", (snr[-1], values[-1]),
                    textcoords="offset points", xytext=(-2, dy),
                    color=colour, fontsize=8, ha="center")

    ax.invert_xaxis()                       # noisier to the right
    ax.margins(x=0.08)
    ax.set_xlabel("SNR (dB) — noisier to the right")
    ax.set_ylabel("orientation error (degrees)")
    ax.set_title("Robustness to additive noise", loc="left")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG}/figR3_noise.png", bbox_inches="tight")
    plt.close(fig)


def fig_runtime(runtime):
    """
    figR4 — the cost difference on SMOOTH synthetic signals, on a log axis.

    Worth reading carefully, because it does not show the headline speedup.
    These are smooth concentric signals with few extrema, and there the local
    envelope is the *slower* option at small sizes — it pays a fixed per-pixel
    cost that the global RBF solve avoids while its system stays small. The
    curves cross somewhere above 128², and only past that does the complexity
    difference start to pay.

    The large speedup quoted in the README comes from fingerprint-density
    input, where the extremum count is several thousand and the cubic term
    dominates. Both numbers are real; they measure different regimes, and this
    figure is the conservative one.

    Log-scaled because the point is the differing growth rate: a global RBF
    solve over n extrema is roughly cubic in n, the local envelope roughly
    linear. On a linear axis the small sizes collapse onto zero.
    """
    sizes = runtime["size"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for values, colour, label in ((runtime["iso_s"], ISO_C, "isotropic BEMD (global RBF)"),
                                  (runtime["st_s"], ST_C, "ST-BEMD (local envelope)")):
        ax.plot(sizes, values, "-o", color=colour, lw=2, ms=8, label=label, zorder=3)

    speedup = runtime["iso_s"][-1] / max(runtime["st_s"][-1], 1e-12)
    ax.annotate(f"{speedup:.1f}x here\nat {sizes[-1]}²",
                (sizes[-1], runtime["st_s"][-1]),
                textcoords="offset points", xytext=(0, -30),
                color=INK, fontsize=9, ha="center", va="top")

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(sizes, [f"{s}²" for s in sizes])
    ax.set_xlabel("image side length (pixels)")
    ax.set_ylabel("seconds per decomposition (log scale)")
    ax.set_title("On smooth signals the local envelope only wins above ~128²",
                 loc="left")
    ax.margins(x=0.1)
    ax.legend()
    _grid(ax, axis="both")
    fig.tight_layout()
    fig.savefig(f"{FIG}/figR4_runtime.png", bbox_inches="tight")
    plt.close(fig)


def main():
    print("1/4 fingerprints..."); fp_sum, curv = eval_fingerprints()
    print("2/4 synthetic..."); syn = eval_synthetic()
    print("3/4 noise..."); noise = eval_noise()
    print("4/4 runtime..."); runtime = eval_runtime()
    json.dump({"fingerprints": fp_sum, "curvature_bins": curv, "synthetic": syn,
               "noise": noise, "runtime": runtime},
              open(f"{RES}/results_full.json", "w"), indent=2, default=float)
    print(f"\n30 fp: iso {fp_sum['iso_mean']:.2f}+/-{fp_sum['iso_std']:.2f} -> "
          f"ST {fp_sum['st_mean']:.2f}+/-{fp_sum['st_std']:.2f}, "
          f"better {fp_sum['pct_st_better']:.0f}%, Wilcoxon p={fp_sum['wilcoxon_p']:.1e}")
    for n in ["low", "medium", "high"]:
        print(f"  curv {n:7}: iso {curv[n]['iso']:.2f}  ST {curv[n]['st']:.2f}")

    fig_fingerprints(fp_sum, curv)
    fig_synthetic(syn)
    fig_noise(noise)
    fig_runtime(runtime)
    print(f"saved results_full.json + figR1..figR4 -> {FIG}")


if __name__ == "__main__":
    main()
