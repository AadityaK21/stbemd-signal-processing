"""
eval_full.py
------------
Expanded evaluation:
  (1) MANY fingerprints with mean+-std and significance test
  (2) orientation error binned by local orientation curvature kappa + improvement-vs-kappa
  (3) analytic-ground-truth synthetic suite (no structure-tensor self-reference)
  (4) robustness to additive noise
  (5) runtime vs image size
Outputs results_full.json + figures figR1..figR4. Requires data/fp_batch.npy.

Key honest findings:
  30 fp: iso 11.84 +/- 4.19 -> ST 7.62 +/- 2.22, ST better on 97%, Wilcoxon p=3.7e-9
  BUT proper ablation (iso-LOCAL rho=1 vs aniso-LOCAL rho=4) shows most of the gain
  is the local-vs-global ENVELOPE (~3.57 deg), anisotropy only ~0.65 deg (p~0.045),
  and does NOT grow with curvature -- contradicting the proposed mechanism.
"""
import json, time, warnings
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from structure_tensor import orientation_and_coherence
from bemd import bemd
from stbemd import stbemd
import analytic_signals as A

warnings.filterwarnings("ignore")
FIG = "."; RES = "."

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
    batch = np.load("data/fp_batch.npy")
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
    print("saved results_full.json")


if __name__ == "__main__":
    main()
