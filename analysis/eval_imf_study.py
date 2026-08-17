"""
eval_imf_study.py
-----------------
100-fingerprint study addressing the extra-IMF concern, reporting ORTHOGONALITY
(direction-fair mode mixing) rather than the orientation-blind spectral overlap.

Reports per method: #IMFs, orientation error, energy ratio, mean pairwise
|IMF correlation| (orthogonality; lower=cleaner), and per-IMF energy fractions.

Headline findings (100 SOCOFing prints, regenerated 2026-08-17):
  #IMFs           iso 3.02 +/- 0.14   ST 4.00 +/- 0.00  (ST +1 on 98% of prints)
  orient_err      iso 12.87 +/- 4.96  ST 8.27 +/- 2.76  (35.7% reduction)
  energy_ratio    iso 0.971           ST 0.946  (tied)
  orthogonality   iso 0.119           ST 0.084  (ST cleaner -- less mode mixing)
  extra ST IMF-4 carries ~0.6% of signal energy; IMF-1 (ridge) unchanged.

NOTE: the Huang Index of Orthogonality = 1 - energy_ratio under exact
reconstruction, so it is redundant with the energy-ratio column; the
non-redundant, orientation-fair measure is mean |IMF correlation| below.
Needs data/fp_batch100.npy -- run scripts/fetch_data.py first.
"""
import _bootstrap  # noqa: F401  (puts the source dirs on sys.path)
import json, time, numpy as np
from metrics import local_orientation_error, orthogonality_index
from emd_metrics_standard import index_of_energy_conservation
from bemd import bemd
from stbemd import stbemd
from _paths import RESULTS, fingerprint_batch

batch = fingerprint_batch(large=True)

def metrics(fp, imfs, res):
    return {
        "n_imfs": len(imfs),
        "orient_err": float(local_orientation_error(imfs[0], fp)),
        "energy_ratio": float(index_of_energy_conservation(fp, imfs, res)["ratio"]),
        "orth_meancorr": float(orthogonality_index(imfs)),
        "imf_energy_frac": [float(np.sum(c**2)/(np.sum(fp**2)+1e-12)) for c in imfs],
        "residual_energy_frac": float(np.sum(res**2)/(np.sum(fp**2)+1e-12)),
    }

out = {"iso": [], "st": []}
t0 = time.time()
for i, fp in enumerate(batch):
    ib, ir = bemd(fp); sb, sr = stbemd(fp)
    out["iso"].append(metrics(fp, ib, ir)); out["st"].append(metrics(fp, sb, sr))
    if (i+1) % 25 == 0: print(f"  {i+1}/100 ({time.time()-t0:.0f}s)")

def agg(k):
    a = np.array([r[k] for r in out["iso"]]); b = np.array([r[k] for r in out["st"]])
    return a, b

summary = {}
for k in ["n_imfs", "orient_err", "energy_ratio", "orth_meancorr"]:
    a, b = agg(k)
    summary[k] = {"iso_mean": float(a.mean()), "iso_std": float(a.std()),
                  "st_mean": float(b.mean()), "st_std": float(b.std())}
ai, bi = agg("n_imfs")
summary["nimf_distribution"] = {v: {"iso": int((ai == v).sum()), "st": int((bi == v).sum())}
                                for v in range(2, 7)}
summary["pct_st_more_imfs"] = float((bi > ai).mean()*100)
summary["prints_iso_4imf"] = [int(i) for i in np.where(ai == 4)[0]]

json.dump({"summary": summary, "per_print": out},
          open(RESULTS / "results_imf.json", "w"), indent=2, default=float)

for k in ["n_imfs", "orient_err", "energy_ratio", "orth_meancorr"]:
    m = summary[k]
    print(f"  {k:16} iso {m['iso_mean']:.3f}+/-{m['iso_std']:.3f}   "
          f"ST {m['st_mean']:.3f}+/-{m['st_std']:.3f}")
print("  prints where iso gave 4 IMFs:", summary["prints_iso_4imf"])
print("  saved results_imf.json")
