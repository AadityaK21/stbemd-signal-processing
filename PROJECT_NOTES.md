# ST-BEMD — project notes

Direction-Adaptive Empirical Mode Decomposition for 2D signals: Structure-Tensor-
Guided Sifting and Riesz Spectral Analysis.

Aaditya Kumawat.

These are the working notes: what each module does, what the numbers actually say,
and which claims not to make. The README is the public-facing version; this file is
the one to read before writing anything up.

## Layout

### `st_bemd/` — the main codebase
(Earlier drafts of these notes called this folder `core/`. It was renamed when the
suite was made importable as a package; every reference below uses the real name.)

- `emd1d.py` — 1D EMD core.
- `signals.py`, `signals_ext.py` — synthetic test signals.
- `structure_tensor.py` — Gaussian-windowed gradient outer product; closed-form
  2x2 eigendecomposition -> orientation theta and coherence c. Also
  `dominant_orientation_deg` (global orientation, used by DEMD).
- `bemd.py` — isotropic BEMD (Nunes 2003): 2D sifting, global thin-plate-spline
  RBF envelope. `find_extrema_2d`, `_corner_anchor` live here.
- `stbemd.py` — **the proposed ST-BEMD** as evaluated: anisotropic, structure-
  tensor-oriented, LOCAL kernel-weighted-average (Nadaraya-Watson) envelope,
  elongated along the ridge by a coherence-driven factor 1+(rho-1)c.
- `pseudo_bemd.py`, `demd.py`, `serial_emd.py` — the other three baselines.
- `riesz.py` — Riesz transform + monogenic amplitude/phase/orientation
  (+ instantaneous_frequency).
- `metrics.py` — `local_orientation_error` (coherence-weighted, angle-folded),
  `orthogonality_index` (mean pairwise |IMF correlation|), reconstruction error.
- `emd_metrics_standard.py` — `index_of_energy_conservation` (energy ratio),
  `index_of_orthogonality` (Huang IO = 1 - energy_ratio), `spectral_overlap`
  (radial Bhattacharyya; orientation-blind — see note below), `imf_mean_coherence`.
- `experiments*.py`, `plot_fingerprint_results.py` — driver scripts / plotting.
- `_paths.py` — every script's ROOT/DATA/OUTPUTS live here, plus the sys.path
  bootstrap. Nothing in the repo hardcodes an absolute path any more.

### `analysis/` — extension scripts built during the study
- `analytic_signals.py` — synthetic signals with ANALYTIC ground-truth
  orientation/frequency/amplitude (removes the fingerprint self-reference).
- `bemd_tuned.py` — isotropic BEMD with tunable local-RBF `neighbors` + `smoothing`
  (shows tuning does NOT rescue standard BEMD; ~12-15 deg regardless).
- `stbemd_rbf.py` — the FAITHFUL proposal: anisotropic RBF *interpolation* with
  per-extremum S_i kernel. Performs far worse (~39 deg) — see findings.
- `ablation.py` — the decisive iso-global vs iso-local vs aniso-local ablation.
- `sweep_rho.py` — anisotropy-ratio sweep (shape, not argmin — selection bias).
- `eval_full.py` — 30-print study + curvature bins + analytic synthetic + noise +
  runtime, and figures figR1..figR4.
- `eval_imf_study.py` — 100-print IMF-count + orthogonality study.
- `_bootstrap.py` — import it first in any analysis script; puts the source dirs on
  sys.path so the scripts run from any working directory.

## Data
The SOCOFing fingerprints are not bundled (licensing/size). Rebuild them with:

```bash
python scripts/fetch_data.py
```

That script is seeded (`default_rng(42)`, 300 candidates, contrast floor std >= 0.12,
first 100 kept) and produces byte-identical arrays on any machine — the 100-print
numbers below reproduce to three decimals after a clean rebuild. It writes
`data/fingerprint.npy`, `data/fp_batch.npy` (30 prints) and `data/fp_batch100.npy`.

No PYTHONPATH is needed any more. Run anything from anywhere:

```bash
python analysis/ablation.py
python run_all.py     # regenerates every result and figure
```

## Honest findings (keep these straight)

All numbers below were regenerated 2026-08-17 from a clean data rebuild. Where they
differ from earlier drafts of this file, these are the ones that reproduce.

**0. Two headline numbers, two datasets.** Earlier drafts quoted "40.6%" and "35.6%"
in different places as though one were a correction of the other. They are not:
40.6% is the *synthetic* suite (3.08 -> 1.83 deg), 35.7% is *100 real prints*
(12.87 -> 8.27 deg). Always say which.

1. Over 100 real prints: orientation error 12.87 +/- 4.96 -> 8.27 +/- 2.76 deg;
   IMF orthogonality cleaner for ST (mean |corr| 0.084 vs 0.119); energy ratio
   effectively tied (0.946 vs 0.971); reconstruction exact (~1e-16); ST yields one
   extra sub-1%-energy IMF (n_imfs 3.02 -> 4.00). On the 30-print subset,
   13.71 -> 8.25 deg, ST better on 100% of prints, Wilcoxon p = 1.9e-9.

2. **Attribution:** the proper ablation (`ablation.py`) shows most of the accuracy
   gain comes from the LOCAL-AVERAGING envelope, not from the anisotropy:

   ```
   iso-GLOBAL (RBF)      13.71     --
   iso-LOCAL  (rho=1)     9.43     envelope:    -4.29 deg  (~78% of the total)
   aniso-LOCAL(rho=4)     8.25     anisotropy:  -1.18 deg  (~22%)
   ```

   Interpolation -> averaging is a FABEMD-class idea (cite Bhuiyan et al.), NOT
   novel here. Anisotropy's own effect is real (better on 70% of prints, Wilcoxon
   p = 0.001) but small, and it *shrinks* with curvature (low +1.42, medium +1.13,
   high +0.20) — contradicting the proposed mechanism, which predicts the gain
   should grow where orientation varies fastest. This is unexplained; do not paper
   over it.

3. The proposal's LITERAL method (anisotropic RBF interpolation, `stbemd_rbf.py`)
   actively fails: 39.3 deg for the faithful `Sinv` metric, 48.6 for `S`, against
   8.3 for the corrected local form. Elongated basis functions make the
   interpolation ill-conditioned and the envelope overshoots. Averaging is bounded
   and cannot overshoot, which is why anisotropy is safe there.

4. **Runtime depends entirely on extremum density, and the two regimes differ by an
   order of magnitude.** On fingerprint-density input the global RBF solve is ~O(n^3)
   in the extremum count and the local envelope ~O(n): 1.1x at 64^2 rising to 31.4x
   at 256^2. On *smooth* synthetic signals with few extrema the local envelope is
   *slower* below about 128^2 and reaches only 2.6x at 256^2 (figR4). Quote whichever
   applies and say which. Either way that speed is the envelope choice, not the
   anisotropy.

5. **Against analytic ground truth the two methods tie** (0.44/0.44, 0.44/0.40,
   0.51/0.51, 0.44/0.44, 0.24/0.24 across the five families). The fingerprint metric
   is coherence-weighted self-reference — it measures whether the IMF preserved the
   input's orientation, not whether that orientation was right. Part of the
   fingerprint gain may be the local envelope agreeing with the structure tensor
   rather than recovering true orientation. This is the single biggest open
   weakness in the evaluation.

6. The rho sweep is flat from 0.5 to 6 and degrades past rho ~ 12; the lowest error
   on the 10-print set is at rho = 0.5 — a slightly *compressed* kernel. Reporting an
   argmin here would be selection bias, and the flatness is itself evidence that
   anisotropy is not doing much work.

7. `spectral_overlap` is radial (orientation-blind) and penalises a directional
   method; use `orthogonality_index` (mean |IMF correlation|) for a fair
   mode-mixing measure. On orthogonality ST is cleaner, not worse.

**Do NOT** attribute the accuracy or speed headline to anisotropy (both are the
envelope), do NOT quote a speedup or a percentage without naming the dataset it came
from, and do NOT call this state-of-the-art — the baseline is the 2003 method and the
fast local envelope is FABEMD-class prior work.

## Where the outputs live

`outputs/results/` holds the JSON for every study; `outputs/figures/` holds all 24
figures. Both are committed, so every claim in the README is checkable without
re-running anything. `outputs/logs/` holds the last `run_all.py` transcript and is
git-ignored.
