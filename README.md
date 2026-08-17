# ST-BEMD — Direction-Adaptive EMD for 2-D Signals

Structure-tensor-guided Empirical Mode Decomposition for non-linear, non-stationary
2-D signals, benchmarked against four EMD baselines on synthetic signals and real
fingerprints.

**Aaditya Kumawat** · Summer 2026 · MIT licence

```bash
pip install -r requirements.txt

python benchmark.py                       # orientation error, all 5 methods
python benchmark.py --mode runtime        # speedup vs extremum density
python benchmark.py --mode all --save     # everything, writes outputs/
pytest tests/                             # 21 tests
```

The synthetic benchmarks need no download. For the fingerprint evaluation:

```bash
python scripts/fetch_data.py              # rebuilds data/ (~35 MB download)
python analysis/eval_full.py              # 30-print study + figures figR1..figR4
```

Every number quoted below is written to `outputs/` by one of those commands and
committed to the repo, so you can check any claim without re-running anything.

---

## How the method works

Standard 2-D EMD builds upper and lower envelopes by interpolating through extrema
**isotropically** — the same in every direction. On a signal with oriented structure
(a fingerprint ridge, a wave front) that is wrong: the envelope averages extrema from
*neighbouring* ridges, so orientation bleeds across crests.

ST-BEMD makes the envelope kernel **anisotropic and locally oriented**:

1. Compute the structure tensor — Gaussian-windowed outer product of gradients.
2. Closed-form 2×2 eigendecomposition gives local orientation θ and coherence c.
3. Build an elliptical Gaussian kernel at each pixel, elongated **along the ridge**
   with σ_long = σ · (1 + (ρ−1)·c) and σ_short = σ.
4. Envelope = normalised kernel-weighted average of nearby extrema.

Where coherence is zero the kernel becomes circular, so the method **degrades
gracefully to ordinary BEMD** exactly where orientation is undefined.

### A correction to the original formulation

The initial formulation used k(xᵢ,xⱼ) = φ(√((xᵢ−xⱼ)ᵀ S⁻¹ (xᵢ−xⱼ))) with S the raw
structure tensor. Taken literally, this elongates the kernel along the tensor's
**major** eigenvector — the *gradient* direction, i.e. across the ridge. That makes
orientation bleeding worse, not better.

The geometry actually required is elongation along the **minor** eigenvector, along
the ridge. `analysis/stbemd_rbf.py` implements the literal version for comparison:

```
                          orientation error on a real print
literal anisotropic RBF                        39.3°
corrected (this repo)                           8.3°
```

The distinction is the difference between the method working and not working.

---

## Results

Two evaluations run on two different kinds of input, and they give two different
headline numbers. Both are real; they are not interchangeable, and older drafts of
these notes quoted them without saying which was which.

### 1. Synthetic signals — 96×96, no download needed

```
method                     plane wave  two orientations  curved   mean
ST-BEMD (proposed)               0.01              5.14    0.35   1.83
BEMD (global RBF)                0.17              8.74    0.33   3.08
Pseudo-BEMD                      0.02             10.99    1.16   4.06
Serial-EMD                       0.31             25.12    1.07   8.83
DEMD (global rotate)             0.46             34.89    6.73  14.03
```

Mean orientation error in degrees, lower is better. The **"two orientations"** column
is the discriminating case: DEMD rotates the whole image to a single global angle, so
on a signal containing two orientations it fails badly (34.89°). ST-BEMD adapts
per-pixel and reaches 5.14°. Against the isotropic BEMD baseline the mean falls
3.08° → 1.83°, a **40.6% reduction**.

→ `outputs/benchmark.json`

### 2. Real fingerprints — 100 SOCOFing prints

| | isotropic BEMD | ST-BEMD |
|---|---|---|
| orientation error | 12.87° ± 4.96 | **8.27° ± 2.76** |
| IMFs produced | 3.02 ± 0.14 | 4.00 ± 0.00 |
| energy ratio | 0.971 | 0.946 |
| orthogonality (mean pairwise \|IMF corr\|) | 0.119 | **0.084** |

A **35.7% reduction** in orientation error — smaller than the synthetic figure, and
the one to quote when talking about real data. On a 30-print subset with per-print
pairing: 13.71° → 8.25°, better on **100%** of prints, Wilcoxon p = 1.9 × 10⁻⁹.

ST-BEMD produces one extra IMF, carrying under 1% of signal energy; IMF-1, the ridge
layer, is unaffected. Orthogonality is *cleaner*, not worse, so the extra mode is not
mode mixing.

Both methods get worse as ridges curve, and the gap holds across the range:

```
curvature bin      isotropic      ST-BEMD
low                     8.38         5.08
medium                 13.10         7.91
high                   19.91        12.93
```

→ `outputs/results/results_imf.json`, `outputs/results/results_full.json`,
`outputs/figures/figR1_fingerprints.png`

### 3. Runtime — the speedup grows with extremum count

Measured on fingerprint-density input:

| Size | Extrema | BEMD (global RBF) | ST-BEMD (local) | Speedup |
|---|---|---|---|---|
| 64² | 873 | 0.08 s | 0.04 s | 1.8× |
| 128² | 1,956 | 0.53 s | 0.15 s | 3.5× |
| 192² | 4,051 | 2.25 s | 0.29 s | 7.8× |
| 256² | 8,313 | 107.90 s | 1.29 s | **83.5×** |

This is a complexity difference, not a constant factor. Global thin-plate-spline RBF
solves a dense n×n system, roughly **O(n³)** in the number of extrema n; the local
anisotropic envelope sums over a ±3σ window, roughly **O(n)**.

**Treat the headline ratio as hardware-dependent, not as a property of the method.**
The same code and the same input measured **31×** on one machine and **83.5×** on
another. ST-BEMD took about the same wall time on both (1.4 s vs 1.3 s); what moved
was the baseline's dense solve, 44 s against 108 s, which depends on the BLAS/LAPACK
build behind NumPy far more than on anything in this repo. The complexity gap is
real; any single multiplier is a property of the machine that produced it.

**And it is the best case, not the typical one.** On *smooth* signals with few
extrema the local envelope is actually the slower option below about 128², because it
pays a fixed per-pixel cost while the global solve's system stays small.
`figR4_runtime.png` shows that regime honestly: the curves cross above 128² and reach
only 3.1× at 256². Quote the large number for fingerprint-density input and 3.1× for
smooth input; quoting the first without both qualifiers overstates the result.

→ `outputs/benchmark.json`, `outputs/figures/figR4_runtime.png`

---

## What the gain is actually attributable to

This is the part a reader should not skip, and the reason the ablation script exists.

ST-BEMD changes **two** things at once relative to the 2003 baseline: the envelope
goes from global RBF *interpolation* to local kernel-weighted *averaging*, and the
kernel goes from circular to elliptical. The fair control isolates them — iso-LOCAL
is this method with ρ = 1, i.e. the same local machinery with a circular kernel.

```
                          orientation error (30 prints)
iso-GLOBAL (RBF)                    13.71 ± 6.16
iso-LOCAL  (rho=1)                   9.43 ± 3.85     envelope change:  -4.29°
aniso-LOCAL(rho=4)                   8.25 ± 3.14     anisotropy:       -1.18°
```

So roughly **78% of the accuracy gain comes from the envelope change**, not from the
anisotropy that the method is named after. Local averaging is a FABEMD-class idea
(Bhuiyan et al.), not novel here. Anisotropy's own contribution is real and
statistically significant — better on 70% of prints, Wilcoxon p = 0.001 — but small.

Worse for the proposed story, the anisotropy gain *shrinks* as ridges curve:

```
curvature bin      anisotropy gain
low                     +1.42°
medium                  +1.13°
high                    +0.20°
```

The stated mechanism predicts the opposite — orientation-adaptivity should matter
most where orientation varies fastest. It does not. Something else is going on, and
the honest position is that this is unexplained.

Two further results point the same way. The ρ sweep finds essentially flat
performance between ρ = 0.5 and ρ = 6 and degradation past ρ ≈ 12, with the lowest
error on this set at ρ = 0.5 — i.e. a *slightly compressed* kernel, not an elongated
one. And against **analytic** ground truth, where the reference orientation is a
closed-form gradient rather than a structure-tensor estimate, the two methods tie:

```
signal family                 iso     ST
concentric core (high curv)   0.44   0.45
whorl (2 singular points)     0.44   0.40
curved chirp (FM)             0.51   0.51
rotating AM-FM                0.44   0.44
gentle rotation (low curv)    0.24   0.24
```

The fingerprint metric is coherence-weighted **self-reference**: it measures whether
the IMF preserved the input's local orientation, not whether that orientation was
correct.

→ `outputs/figures/figR2_synthetic.png`, `outputs/figures/rho_sweep.png`

### The fingerprint metric does not measure orientation recovery

`analysis/control_smoothing.py` tests the metric itself, by scoring estimators that
have **no orientation input at all** — a plain Gaussian high-pass, `imf = f − G_σ ∗ f`.
No extrema, no envelope, no sifting, no structure tensor.

```
estimator                       fingerprint      analytic GT
                                 (self-ref)     (closed form)
identity (returns input)              0.00             0.43
Gaussian high-pass σ=4.0              4.23             0.42
Gaussian high-pass σ=1.0              7.26             0.27
iso-GLOBAL BEMD (baseline)           13.71             0.42
iso-LOCAL (ρ=1)                       9.43             0.43
aniso-LOCAL (ρ=4)  ST-BEMD            8.25             0.41
```

**A Gaussian blur subtracted from the input scores 4.23°, roughly twice as good as
ST-BEMD's 8.25°.** And the identity map — returning the input untouched, extracting
nothing — scores a perfect 0.00, because `local_orientation_error(f, f)` is zero by
construction. The metric's global optimum is to do nothing, so it cannot reward
extraction; it can only penalise disturbance. It ranks estimators by how gently they
alter the orientation field.

On the closed-form column every estimator lands between 0.27° and 0.43°, and the best
score belongs to the σ=1 high-pass. There is no signal there either way.

**What this means for the headline.** The 13.71° → 8.25° improvement is real as a
measurement, but it is evidence that ST-BEMD's local envelope disturbs the
orientation field less than a global RBF interpolant — not that it recovers
orientation better. Those are different claims, and only the first is supported.

**What it does not mean.** The Gaussian high-pass is not a better method; it is a
probe. It produces no IMF in any meaningful sense — no local-mean-zero property, no
multi-scale decomposition, nothing to iterate. That is precisely the point: an
estimator that fails at the actual task still wins the metric, so the metric is not
measuring the task.

→ `outputs/results/results_control.json`, `outputs/figures/figR5_metric_control.png`

### Re-test: the orientation metric is measuring how little you sifted

`analysis/metric_validity.py` repeats the control with two corrections. First, the
high-pass σ is now chosen on prints 0–14 and scored on prints 15–29, since sweeping σ
on the same prints it was scored on was selection bias — the error this project
objects to elsewhere. Second, it adds the metric the comparison should have used all
along: **IMF validity**, ‖local mean‖ / ‖mode‖, where the local mean is the average of
the envelopes through the mode's own extrema. Driving that to zero is what sifting
*is*; a Gaussian high-pass has no mechanism to achieve it.

Held out, on 15 unseen prints:

```
estimator                      orient err    IMF validity
                                    (deg)   (0 = is an IMF)
iso-GLOBAL BEMD (baseline)          15.91            0.131
iso-LOCAL (ρ=1)                     10.08            0.210
aniso-LOCAL (ρ=4)  ST-BEMD           9.34            0.185
Gaussian high-pass σ=4.0             4.86            0.383
```

The selection bias was **not** the explanation — the control still wins on
orientation, held out. But it is roughly twice as far from being an IMF as any real
method, and across all 10 estimators the two metrics are anti-correlated at
**r = −0.94**. Every estimator that scores well on orientation scores badly on
validity, in a near-straight line.

That is the mechanism, stated precisely: **the orientation metric is largely
measuring how little the estimator sifted.** Do nothing and score 0.00. Sift gently
and score well. Sift properly — which is the job — and score worse. It is not a
measure of orientation recovery at any point on that line.

Two consequences, and they cut in opposite directions:

- The 13.71° → 8.25° headline cannot be read as "ST-BEMD recovers orientation
  better". It supports only "ST-BEMD's envelope disturbs the orientation field less
  than a global RBF interpolant".
- The Gaussian high-pass is not a rival method, and the trade-off plot is why. It
  buys its orientation score by not doing the task. On IMF validity the ordering
  reverses completely and the baseline wins outright (0.131), which is the honest
  cost of this repo's local-averaging envelope — it does not interpolate the extrema,
  so its modes are further from strict IMFs. That cost was noted in Limitations from
  the start; here it finally has a number.

Anisotropy earns a small, clean point here: ST-BEMD is better than iso-LOCAL on
*both* axes (9.34 vs 10.08, and 0.185 vs 0.210). Same envelope machinery, only the
kernel shape differs. It is a modest result, but it is one the metric cannot
manufacture.

→ `outputs/results/results_validity.json`, `outputs/figures/figR6_metric_tradeoff.png`

### A non-circular real-data test — and the one result that survives it

`analysis/independent_eval.py` removes the circularity two ways at once.

**A different instrument.** `analysis/gabor_orientation.py` estimates orientation with
a bank of oriented Gabor quadrature pairs — a matched-filter principle sharing no
machinery with the structure tensor that ST-BEMD steers by. It agrees with the tensor
to 0.02° on plane waves of known angle.

**A different protocol.** Changing the instrument alone fixes nothing, because the
identity map still scores 0.00 against any self-referential reference. So the
reference orientation is taken from the **clean** print while the methods are given a
**noisy** one. Now preserving the input no longer wins by default; recovering the
underlying ridge orientation through noise is the only way to score well.

Reference and modes are measured with filters pinned to the clean print's ridge
wavelength, prints are upsampled 2× (at native 3.2 px ridges the two instruments
disagree by 7.6° on the *same* image, which would swamp everything), and the scored
mode is whichever IMF matches the input's own ridge wavelength — because EMD's IMF-1
on a noisy image is the *noise*, not the ridges.

Mean orientation error against the clean reference, 12 prints:

```
                          ── Gabor bank ──      ── structure tensor ──
                        20 dB  10 dB   5 dB    20 dB  10 dB   5 dB
identity (does nothing)  2.72   4.79   7.55     4.53   7.26   9.56
Gaussian high-pass σ=4   9.58   9.91  12.01     4.92   7.64   9.98
ST-BEMD (ρ=4)           10.97  12.57  15.06     6.85  10.56  13.76
iso-LOCAL (ρ=1)         10.69  11.92  16.65     7.07  10.17  15.58
iso-GLOBAL BEMD         10.04  12.87  18.33     6.67  10.79  17.33
```

**ST-BEMD degrades more gracefully under noise than the isotropic baseline, and this
is the project's only non-circular real-data result that favours the method.** The
advantage is absent in clean conditions and grows as noise rises:

```
SNR      iso-GLOBAL   ST-BEMD     gap    (tensor gap)
20 dB        10.04     10.97    −0.93          −0.18
10 dB        12.87     12.57    +0.30          +0.23
 5 dB        18.33     15.06    +3.27          +3.56
```

At 20 dB ST-BEMD is slightly *worse*. At 5 dB it is ahead by 3.3°, and 3.6° under the
second instrument — clear of the ~1.5° instrument floor, with both instruments
agreeing on the full ranking. That is a defensible claim: **anisotropic local
envelopes are more noise-robust than global RBF interpolation**, not that they
estimate orientation better in general.

**What still does not work.** The identity map remains the lowest-error entry (5.02°).
The protocol reduced its advantage — it scored 0.00 before — but did not remove it,
because orientation is an intrinsically noise-robust property: window-averaged
estimators barely move at these SNRs, so preserving the image still preserves the
field. And the Gaussian high-pass still beats every EMD method here, exactly as it did
before; only the IMF-validity axis separates them. This protocol is **necessary but
not sufficient**, and no orientation number in this project should be quoted without
that caveat.

→ `outputs/results/results_independent.json`, `outputs/figures/figR7_independent_eval.png`

---

## Layout

```
├── benchmark.py            CLI — orientation, runtime, sanity modes
├── run_all.py              regenerate every result and figure in outputs/
├── scripts/fetch_data.py   rebuild data/ from the public mirror
├── st_bemd/                the library
│   ├── stbemd.py           ★ proposed method
│   ├── bemd.py             baseline 1 — isotropic, global RBF
│   ├── pseudo_bemd.py      baseline 2 — 1-D EMD on rows/cols
│   ├── demd.py             baseline 3 — global rotation to dominant angle
│   ├── serial_emd.py       baseline 4 — serialised 1-D EMD
│   ├── structure_tensor.py orientation + coherence
│   ├── riesz.py            Riesz transform, monogenic signal
│   ├── metrics.py          orientation error, orthogonality
│   ├── _paths.py           shared project paths (see note below)
│   ├── signals*.py         synthetic test signals
│   └── experiments*.py     figure drivers for the report
├── analysis/               study scripts (ablation, rho sweep, full eval)
├── outputs/                committed results — figures/ and results/
└── tests/test_stbemd.py    21 tests
```

**Note on the package wrapper.** The research modules import each other by bare name
(`from bemd import find_extrema_2d`), which only worked when the working directory was
the source folder. Rather than rewrite working numerical code, `__init__.py` puts its
own directory on `sys.path` first — the original imports resolve unchanged, and the
suite is also importable as `import st_bemd`. Scripts under `analysis/` do the same
via `import _bootstrap`, so every script runs from any directory with no PYTHONPATH.

All paths resolve through `st_bemd/_paths.py`, so results always land in `outputs/`
regardless of where you launch from.

---

## Reproducing everything

```bash
python scripts/fetch_data.py     # once — rebuilds data/, seeded and deterministic
python run_all.py                # ~4 min — every analysis script, logs to outputs/logs/
python scripts/check_claims.py   # asserts this README matches outputs/
```

`check_claims.py` re-derives all 19 numbers quoted above from the committed JSON and
fails if any has drifted. CI runs it on every push, so the documentation cannot go
stale without the build going red.

`run_all.py` regenerates all 24 figures and 5 result files (`--quick` skips the two
slowest). Individual scripts:

| Script | What it answers |
|---|---|
| `analysis/ablation.py` | envelope vs anisotropy — the decisive experiment |
| `analysis/eval_full.py` | 30-print study, curvature bins, analytic GT, noise, runtime |
| `analysis/eval_imf_study.py` | 100-print IMF-count and orthogonality study |
| `analysis/control_smoothing.py` | **is the fingerprint metric measuring anything?** |
| `analysis/metric_validity.py` | held-out re-test + IMF validity; the trade-off plot |
| `analysis/independent_eval.py` | **the non-circular real-data test** |
| `analysis/gabor_orientation.py` | independent orientation instrument (library + self-test) |
| `analysis/sweep_rho.py` | anisotropy-ratio sweep (shape, not argmin) |
| `analysis/stbemd_rbf.py` | the literal formulation, for contrast |
| `analysis/bemd_tuned.py` | does tuning rescue the baseline? (no) |
| `st_bemd/experiments*.py` | figure drivers for the written report |

The fingerprint data is **not** in the repo — SOCOFing is not redistributable.
`scripts/fetch_data.py` rebuilds it from a public mirror with a fixed seed, so the
arrays are byte-identical on any machine.

---

## Limitations

- **The headline is the envelope, not the anisotropy.** See the attribution section.
  Do not describe this as a 40% win for direction-adaptive kernels.
- **Not state of the art.** The baseline is the 2003 method; the fast local envelope
  is FABEMD-class prior work. This is a careful comparison, not a new frontier.
- **The real-data metric is self-referential.** Coherence-weighted orientation error
  measures preservation, not correctness. Against analytic ground truth the methods
  tie — `analysis/analytic_signals.py` is the honest control.
- **ρ was swept, not learned.** `analysis/sweep_rho.py` reports the shape of the curve
  rather than an argmin, deliberately — picking the best ρ on the evaluation set
  would be selection bias.
- **No comparison against non-EMD methods.** Gabor filter banks and monogenic
  wavelets solve adjacent problems and are not benchmarked here.
- **The envelope is a kernel-weighted average, not an interpolant.** It does not pass
  exactly through the extrema, making it smoother than classical BEMD — a deliberate
  trade for speed and stability, and a deviation from strict EMD.
- **`spectral_overlap` is radial and therefore orientation-blind**, which penalises a
  directional method. Use `orthogonality_index` (mean pairwise |IMF correlation|) for
  a fair mode-mixing comparison.

---

## References

- Nunes et al. (2003) — bidimensional EMD, the isotropic baseline.
- Bhuiyan, Adhami & Khan (2008) — FABEMD; the fast order-statistics envelope this
  method's local averaging belongs to.
- Felsberg & Sommer (2001) — the monogenic signal, used in `riesz.py`.
- Huang et al. (1998) — the original EMD and the index of orthogonality.
