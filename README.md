# ST-BEMD — Direction-Adaptive EMD for 2-D Signals

Structure-tensor-guided Empirical Mode Decomposition for non-linear, non-stationary
2-D signals, benchmarked against four EMD baselines.

**Aaditya Kumawat** · Summer 2026

```bash
pip install -r requirements.txt

python benchmark.py                       # orientation error, all 5 methods
python benchmark.py --mode runtime        # speedup vs extremum density
python benchmark.py --mode all --save     # everything, writes outputs/
pytest tests/                             # 21 tests
```

Runs with no data download — synthetic signals are built in.

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
   with `σ_long = σ · (1 + (ρ−1)·c)` and `σ_short = σ`.
4. Envelope = normalised kernel-weighted average of nearby extrema.

Where coherence is zero the kernel becomes circular, so the method **degrades
gracefully to ordinary BEMD** exactly where orientation is undefined.

### A correction to the original formulation

The initial formulation used `k(xᵢ,xⱼ) = φ(√((xᵢ−xⱼ)ᵀ S⁻¹ (xᵢ−xⱼ)))` with S the raw
structure tensor. Taken literally, this elongates the kernel along the tensor's
**major** eigenvector — the *gradient* direction, i.e. across the ridge. That makes
orientation bleeding worse, not better.

The geometry actually required is elongation along the **minor** eigenvector, along
the ridge. `analysis/stbemd_rbf.py` implements the literal version for comparison: it
scores ~42° mean orientation error versus ~7° for the corrected form. The distinction
is the difference between the method working and not working.

---

## Results (synthetic, 96×96, reproducible)

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
per-pixel and reaches 5.14°. Overall ST-BEMD reduces mean error 40.6% against the
isotropic BEMD baseline (3.08° → 1.83°).

### Runtime — the speedup grows with extremum count

Measured on fingerprint-density input:

| Size | Extrema | BEMD (global RBF) | ST-BEMD (local) | Speedup |
|---|---|---|---|---|
| 64² | 873 | 0.05 s | 0.04 s | 1.2× |
| 128² | 1,956 | 0.48 s | 0.16 s | 3.0× |
| 192² | 4,051 | 1.98 s | 0.33 s | **6.0×** |
| 256² | 8,313 | 39.58 s | 1.21 s | **32.8×** |

This is a complexity difference, not a constant factor. Global thin-plate-spline RBF
solves a dense n×n system, roughly **O(n³)** in the number of extrema; the local
anisotropic envelope sums over a ±3σ window, roughly **O(n)**. The harder the image,
the larger the gain.

---

## Layout

```
├── benchmark.py            CLI — orientation, runtime, sanity modes
├── st_bemd/                the library
│   ├── stbemd.py           ★ proposed method
│   ├── bemd.py             baseline 1 — isotropic, global RBF
│   ├── pseudo_bemd.py      baseline 2 — 1-D EMD on rows/cols
│   ├── demd.py             baseline 3 — global rotation to dominant angle
│   ├── serial_emd.py       baseline 4 — serialised 1-D EMD
│   ├── structure_tensor.py orientation + coherence
│   ├── riesz.py            Riesz transform, monogenic signal
│   ├── metrics.py          orientation error, orthogonality
│   └── signals*.py         synthetic test signals
├── analysis/               study scripts (ablation, rho sweep, full eval)
└── tests/test_stbemd.py    21 tests
```

**Note on the package wrapper.** The research modules import each other by bare name
(`from bemd import find_extrema_2d`), which only worked when the working directory was
the source folder. Rather than rewrite working numerical code, `__init__.py` puts its
own directory on `sys.path` first — the original imports resolve unchanged, and the
suite is also importable as `import st_bemd`.

---

## Reproducing the fingerprint evaluation

The SOCOFing fingerprint data is not bundled (licensing and size). `PROJECT_NOTES.md`
contains the script to build `data/fp_batch.npy` from a local copy of the dataset, then:

```bash
python benchmark.py --data data/fp_batch.npy --limit 30
```

---

## Limitations

- **Not evaluated against analytic ground truth on real data.** The fingerprint metric
  is coherence-weighted self-reference — it measures whether the IMF preserves the
  input's local orientation, not whether that orientation is correct.
  `analysis/analytic_signals.py` addresses this for synthetic signals.
- **ρ (anisotropy ratio) was swept, not learned.** `analysis/sweep_rho.py` reports the
  shape of the curve rather than an argmin, deliberately — selecting the best ρ on the
  evaluation set would be selection bias.
- **No comparison against non-EMD methods.** Gabor filter banks and monogenic wavelets
  solve adjacent problems and are not benchmarked here.
- **The envelope is a kernel-weighted average, not an interpolant.** It does not pass
  exactly through the extrema, making it a smoother envelope than classical BEMD — a
  deliberate trade for speed and stability, and a deviation from strict EMD.
