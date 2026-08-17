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
| 64² | 873 | 0.06 s | 0.05 s | 1.1× |
| 128² | 1,956 | 0.48 s | 0.19 s | 2.5× |
| 192² | 4,051 | 2.04 s | 0.35 s | 5.9× |
| 256² | 8,313 | 44.18 s | 1.41 s | **31.4×** |

This is a complexity difference, not a constant factor. Global thin-plate-spline RBF
solves a dense n×n system, roughly **O(n³)** in the number of extrema n; the local
anisotropic envelope sums over a ±3σ window, roughly **O(n)**.

**The 31× figure is the best case, not the typical one.** On *smooth* signals with
few extrema the local envelope is actually the slower option below about 128²,
because it pays a fixed per-pixel cost while the global solve's system stays small.
`figR4_runtime.png` shows that regime honestly: the curves cross above 128² and reach
only 2.6× at 256². Quote 31× for fingerprint-density input and 2.6× for smooth input;
quoting the first without the qualifier overstates the result.

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
correct. The synthetic tie suggests part of the fingerprint gain may be the local
envelope agreeing with the structure tensor rather than recovering true orientation.

→ `outputs/figures/figR2_synthetic.png`, `outputs/figures/rho_sweep.png`

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
