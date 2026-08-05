# ST-BEMD — project notes

Direction-Adaptive Empirical Mode Decomposition for 2D signals: Structure-Tensor-
Guided Sifting and Riesz Spectral Analysis.

Aaditya Kumawat.

## Layout

### `core/` — the main codebase (the uploaded originals)
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

### `analysis/` — extension scripts built during the study
- `analytic_signals.py` — synthetic signals with ANALYTIC ground-truth
  orientation/frequency/amplitude (removes the fingerprint self-reference).
- `bemd_tuned.py` — isotropic BEMD with tunable local-RBF `neighbors` + `smoothing`
  (shows tuning does NOT rescue standard BEMD; ~12-15 deg regardless).
- `stbemd_rbf.py` — the FAITHFUL proposal: anisotropic RBF *interpolation* with
  per-extremum S_i kernel. Performs far worse (~42 deg) — see findings.
- `ablation.py` — the decisive iso-global vs iso-local vs aniso-local ablation.
- `sweep_rho.py` — anisotropy-ratio sweep (shape, not argmin — selection bias).
- `eval_full.py` — 30-print study + curvature bins + analytic synthetic + noise + runtime.
- `eval_imf_study.py` — 100-print IMF-count + orthogonality study.

## Data
The SOCOFing fingerprints are not bundled (licensing/size). Regenerate with:

```python
import urllib.request, tarfile, io, numpy as np
url='https://codeload.github.com/kairess/fingerprint_recognition/tar.gz/master'
tar=tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()))
m=[x for x in tar.getmembers() if x.name.endswith('x_real.npz')][0]
open('data/x_real.npz','wb').write(tar.extractfile(m).read())
arr=np.load('data/x_real.npz')['data'][...,0].astype(float)
rng=np.random.default_rng(42); idx=rng.choice(len(arr),300,replace=False)
b=[]
for i in idx:
    im=arr[i]; im=(im-im.min())/(im.max()-im.min()+1e-9)
    if im.std()<0.12: continue
    b.append(im-im.mean())
    if len(b)>=100: break
np.save('data/fp_batch100.npy', np.array(b))
np.save('data/fp_batch.npy', np.array(b[:30]))
im0=arr[0]; im0=(im0-im0.min())/(im0.max()-im0.min()+1e-9); np.save('data/fingerprint.npy', im0-im0.mean())
```
Run analysis scripts from the project root so `data/...` paths resolve, with
`core/` and `analysis/` on the path (e.g. `PYTHONPATH=core:analysis`).

## Honest findings (keep these straight)
1. Over 100 real prints: orientation error 12.87 -> 8.27 deg; IMF orthogonality
   cleaner for ST (mean |corr| 0.084 vs 0.119); energy ratio tied (~0.95 vs 0.97);
   reconstruction exact (~1e-16); ST yields one extra sub-1%-energy IMF.
2. **Attribution:** the proper ablation (`ablation.py`) shows most of the accuracy
   gain (~3.57 of ~4.2 deg) comes from the LOCAL-AVERAGING envelope
   (interpolation -> averaging, a FABEMD-class idea — cite Bhuiyan et al.), NOT
   from the anisotropy. Anisotropy's own effect is small (~0.65 deg, p~0.045) and
   does NOT grow with curvature, contradicting the proposed mechanism.
3. The proposal's LITERAL method (anisotropic RBF interpolation, `stbemd_rbf.py`)
   actively fails (~42 deg): elongated basis functions make the interpolation
   ill-conditioned and the envelope overshoots. Averaging is bounded and cannot
   overshoot, which is why anisotropy is safe there.
4. Runtime: replacing the global O(N^6) dense-RBF solve with a local O(N^2)
   envelope gives ~3.2x speedup at 256^2 — but that speed is the envelope choice,
   not the anisotropy.
5. `spectral_overlap` is radial (orientation-blind) and penalises a directional
   method; use `orthogonality_index` (mean |IMF correlation|) for a fair
   mode-mixing measure. On orthogonality ST is cleaner, not worse.

Do NOT attribute the 35.6%/3.2x headline to anisotropy (it is the envelope), and
do NOT call this state-of-the-art (baseline is the 2003 method; the fast local
envelope is FABEMD-class).
