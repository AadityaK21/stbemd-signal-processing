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
- `gabor_orientation.py` — orientation from a Gabor quadrature bank; an instrument
  INDEPENDENT of the structure tensor, used to break the evaluation's circularity.
- `independent_eval.py` — **the non-circular real-data test**: reference from the
  clean print, methods see the noisy one, scored under both instruments.
- `metric_validity.py` — held-out re-test of the control + IMF validity + trade-off.
- `control_smoothing.py` — **scores orientation-BLIND estimators (Gaussian high-pass,
  identity) on both metrics.** Shows the fingerprint metric rewards gentleness, not
  orientation recovery. Read finding 6 before quoting any fingerprint number.
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
   in the extremum count and the local envelope ~O(n): 1.8x at 64^2 rising to 83.5x
   at 256^2 on the machine that produced the committed benchmark.json. On *smooth*
   synthetic signals with few extrema the local envelope is *slower* below about
   128^2 and reaches only 3.1x at 256^2 (figR4). Quote whichever applies and say
   which. Either way that speed is the envelope choice, not the anisotropy.

   The multiplier is HARDWARE, not method: the same code and input measured 31x on
   one machine and 83x on another, because the baseline's dense solve is dominated
   by whichever BLAS/LAPACK NumPy was built against (44s vs 108s), while ST-BEMD
   was ~1.3s on both. Never quote a bare speedup number without saying on what.

5. **Against analytic ground truth the two methods tie** (0.44/0.44, 0.44/0.40,
   0.51/0.51, 0.44/0.44, 0.24/0.24 across the five families).

6. **THE FINGERPRINT METRIC DOES NOT MEASURE ORIENTATION RECOVERY.** This is the
   most important finding in the project and it invalidates the headline framing.
   `control_smoothing.py` scores estimators with NO orientation input:

   ```
                                  fingerprint    analytic GT
   identity (returns input)             0.00           0.43
   Gaussian high-pass sigma=4.0         4.23           0.42
   iso-GLOBAL BEMD                     13.71           0.42
   iso-LOCAL (rho=1)                    9.43           0.43
   aniso-LOCAL (rho=4) ST-BEMD          8.25           0.41
   ```

   A Gaussian blur subtracted from the input (`f - G_sigma * f`) scores 4.23 --
   about twice as good as ST-BEMD -- with no extrema, no envelope, no sifting and
   no structure tensor. The identity map scores a perfect 0.00, because
   `local_orientation_error(f, f) == 0` by construction. The metric's optimum is
   to do nothing; it ranks estimators by how GENTLY they disturb the orientation
   field, not by how well they recover it.

   So 13.71 -> 8.25 is real as a measurement but supports only the claim "the
   local envelope disturbs orientation less than a global RBF interpolant". It
   does NOT support "ST-BEMD estimates orientation better". Do not write the
   second sentence anywhere.

   Caveat in our own favour: the high-pass is a probe, not a rival method. It
   produces nothing that is an IMF -- no local-mean-zero property, no multi-scale
   decomposition. That is the argument, though: an estimator that cannot do the
   task still wins the metric.

   Re-tested in `metric_validity.py` with the selection bias removed (sigma
   chosen on prints 0-14, scored on 15-29) and with IMF VALIDITY added
   (||local mean|| / ||mode||, the property sifting actually optimises):

   ```
                                orient err   IMF validity
   iso-GLOBAL BEMD                   15.91          0.131
   iso-LOCAL (rho=1)                 10.08          0.210
   aniso-LOCAL (rho=4) ST-BEMD        9.34          0.185
   Gaussian high-pass sigma=4          4.86          0.383
   ```

   Selection bias was NOT the explanation -- the control still wins on
   orientation held out. But across all 10 estimators the two metrics are
   anti-correlated at r = -0.94. The orientation metric is measuring HOW LITTLE
   THE ESTIMATOR SIFTED. That is the mechanism; state it that way.

   Two things follow, in opposite directions:
   - Never claim "ST-BEMD recovers orientation better". Claim only "its envelope
     disturbs the orientation field less than a global RBF interpolant".
   - The high-pass is not a rival method: it buys its score by not doing the job,
     and on IMF validity the ordering reverses. iso-GLOBAL wins that axis (0.131)
     because it interpolates the extrema exactly. Our local-averaging envelope
     does not, so its modes are further from strict IMFs -- the cost was in the
     Limitations from the start, and now it has a number.
   - Anisotropy earns one clean point: ST-BEMD beats iso-LOCAL on BOTH axes
     (9.34 vs 10.08, 0.185 vs 0.210), same machinery, only kernel shape differs.
     Small, but the metric cannot manufacture it.

   Addressed in finding 7.

7. **THE NON-CIRCULAR TEST, AND THE ONE RESULT THAT SURVIVES IT**
   (`independent_eval.py`, instrument in `gabor_orientation.py`).

   Circularity was removed two ways at once, because changing only the
   instrument does nothing -- the identity still scores 0.00 against any
   self-referential reference:
     - different instrument: Gabor quadrature bank (matched filter), shares no
       code with the structure tensor; agrees to 0.02 deg on known plane waves.
     - different protocol: reference from the CLEAN print, methods see the
       NOISY print. Preserving the input no longer wins by default.

   Mean error vs the clean reference, 12 prints, both instruments:

   ```
                          Gabor: 20/10/5 dB      tensor: 20/10/5 dB
   identity                2.72  4.79   7.55      4.53  7.26   9.56
   Gaussian high-pass s=4  9.58  9.91  12.01      4.92  7.64   9.98
   ST-BEMD (rho=4)        10.97 12.57  15.06      6.85 10.56  13.76
   iso-LOCAL (rho=1)      10.69 11.92  16.65      7.07 10.17  15.58
   iso-GLOBAL BEMD        10.04 12.87  18.33      6.67 10.79  17.33
   ```

   THE CLAIM WE CAN MAKE: ST-BEMD degrades more gracefully under noise than
   isotropic BEMD. Gap vs iso-GLOBAL is -0.93 at 20 dB, +0.30 at 10 dB, +3.27
   at 5 dB (+3.56 under the tensor) -- above the ~1.5 deg instrument floor, and
   both instruments agree on the whole ranking. Phrase it as "anisotropic local
   envelopes are more noise-robust than global RBF interpolation". NOT as
   "estimates orientation better" -- at 20 dB it is slightly worse.

   WHAT STILL FAILS: the identity is still the lowest-error entry (5.02). The
   protocol shrank its edge (it was 0.00) but did not remove it, because
   orientation is intrinsically noise-robust -- window-averaged estimators barely
   move at these SNRs. The Gaussian high-pass still beats every EMD method on
   orientation; only IMF validity (finding 6) separates them. Necessary but not
   sufficient. Say so.

   Three implementation traps, all of which produced wrong numbers first:
     - IMF-1 of a NOISY image is the noise. Scoring it gave 27-36 deg (random is
       45). Score the mode whose wavelength matches the input's ridge scale.
     - The Gabor bank must be PINNED to the reference's wavelengths. Letting it
       retune per image measures the two sides of the subtraction with different
       filters and inflates every error by ~8 deg.
     - `dominant_wavelength` needs r**2 detrending or the low-frequency envelope
       outvotes the ridge bump: without it, 7 of 10 prints returned 18.0 px
       instead of ~3.4 px, silently detuning the entire bank.

   Still open: a hand-marked orientation ground truth, or a higher-resolution
   dataset. At 90x90 the ridges are ~3.2 px and no matched filter is angularly
   sharp there; we upsample 2x to halve instrument disagreement (7.6 -> 4.0 deg).

8. The rho sweep is flat from 0.5 to 6 and degrades past rho ~ 12; the lowest error
   on the 10-print set is at rho = 0.5 — a slightly *compressed* kernel. Reporting an
   argmin here would be selection bias, and the flatness is itself evidence that
   anisotropy is not doing much work.

9. `spectral_overlap` is radial (orientation-blind) and penalises a directional
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
