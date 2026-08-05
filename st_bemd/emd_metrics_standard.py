"""
emd_metrics_standard.py
-----------------------
The *standard* quantitative measures used in the EMD / BEMD literature to judge
the quality of a decomposition, implemented for 2D (sums run over all pixels
instead of over time). These complement the proposal-specific directional
metrics in metrics.py.

Implemented here
----------------
1. Index of Orthogonality (IO)            -- Huang et al., 1998
2. Index of Energy Conservation (IEC)     -- Huang et al., 1998 (energy leakage)
3. IMF cross-correlation matrix           -- waveform-similarity mode mixing
4. Radially-averaged power spectrum        -- per-IMF spectral footprint
5. Spectral overlap (Bhattacharyya)        -- scale-leakage mode mixing
6. Per-IMF mean structure-tensor coherence -- directional purity of a mode

Why these
---------
EMD/BEMD is supposed to behave like a dyadic filter bank producing a (quasi-)
orthogonal set of IMFs. Two complementary failures are checked:
  * IO and the cross-correlation matrix detect the "same waveform appearing in
    different IMFs" form of mode mixing (loss of orthogonality);
  * the spectral overlap detects the "disparate scales inside one IMF" form,
    because genuine IMFs should occupy *separate* radial-frequency bands.
IEC checks that energy is conserved (no leakage between modes), which holds iff
the cross terms vanish.
"""
import numpy as np


def _stack(imfs, residual=None):
    """Stack IMFs (+ optional residual) into an (M, H, W) array of components."""
    comps = [np.asarray(c, dtype=float) for c in imfs]
    if residual is not None:
        comps = comps + [np.asarray(residual, dtype=float)]
    return np.stack(comps, axis=0)


# ---------------------------------------------------------------------------
# 1. Index of Orthogonality  (Huang et al. 1998)
# ---------------------------------------------------------------------------
def index_of_orthogonality(imfs, residual=None):
    """
    Global Index of Orthogonality, summed over all pixels and normalised by the
    total signal energy:

        IO = sum_pixels sum_{j!=k} C_j C_k  /  sum_pixels ( sum_j C_j )^2

    IO = 0 for a perfectly orthogonal decomposition; IO -> 1 in the worst case.
    For well-behaved EMD it is typically 1e-2 .. 1e-3. The components C_j are the
    IMFs together with the residual.
    """
    C = _stack(imfs, residual)
    total = np.sum(C, axis=0)                 # = reconstructed signal
    X2 = np.sum(total ** 2)                    # sum of X^2 over pixels
    sum_sq = np.sum(C ** 2)                    # sum_j sum_pixels C_j^2
    cross = X2 - sum_sq                        # = sum_{j!=k} C_j C_k over pixels
    return float(cross / (X2 + 1e-300))


def pairwise_orthogonality(imfs, residual=None):
    """
    Pairwise normalised cross-correlation matrix between components,
    IO_jk = <C_j, C_k> / (||C_j|| ||C_k||). Diagonal is 1; large off-diagonal
    magnitudes localise *which* modes overlap.
    """
    C = _stack(imfs, residual)
    M = C.shape[0]
    flat = C.reshape(M, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1) + 1e-12
    G = (flat @ flat.T) / np.outer(norms, norms)
    return G


# ---------------------------------------------------------------------------
# 2. Index of Energy Conservation  (Huang et al. 1998)
# ---------------------------------------------------------------------------
def index_of_energy_conservation(signal, imfs, residual=None):
    """
    Energy-leakage check. With E_x the signal energy and sum_j E_j the summed
    IMF (+residual) energies, returns a dict with:
        ratio   = sum_j E_j / E_x      (ideal 1.0; cross terms push it away)
        leakage = (sum_j E_j - E_x) / E_x   (signed relative leakage; ideal 0)
    Energy is conserved exactly iff the modes are mutually orthogonal.
    """
    signal = np.asarray(signal, dtype=float)
    C = _stack(imfs, residual)
    E_x = float(np.sum(signal ** 2))
    E_sum = float(np.sum(C ** 2))
    return {"ratio": E_sum / (E_x + 1e-300),
            "leakage": (E_sum - E_x) / (E_x + 1e-300)}


# ---------------------------------------------------------------------------
# 3. IMF cross-correlation (mode-mixing, waveform form)
# ---------------------------------------------------------------------------
def imf_correlation_matrix(imfs):
    """Pearson cross-correlation matrix between IMFs only (no residual)."""
    return pairwise_orthogonality(imfs, residual=None)


def mean_abs_offdiag_correlation(imfs):
    """Mean |correlation| over distinct IMF pairs (0 = orthogonal modes)."""
    G = imf_correlation_matrix(imfs)
    M = G.shape[0]
    if M < 2:
        return 0.0
    iu = np.triu_indices(M, k=1)
    return float(np.mean(np.abs(G[iu])))


# ---------------------------------------------------------------------------
# 4 & 5. Radial power spectrum + spectral overlap (mode-mixing, scale form)
# ---------------------------------------------------------------------------
def radial_power_spectrum(img, nbins=40):
    """
    Radially-averaged power spectrum of a 2D field, returned as a normalised
    distribution over radial spatial-frequency bins (sums to 1). This is the
    isotropic spectral footprint of an IMF.
    """
    img = np.asarray(img, dtype=float)
    img = img - img.mean()
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2
    H, W = img.shape
    cy, cx = H / 2.0, W / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    bins = np.linspace(0, r_max, nbins + 1)
    which = np.digitize(r.ravel(), bins) - 1
    which = np.clip(which, 0, nbins - 1)
    prof = np.bincount(which, weights=P.ravel(), minlength=nbins)
    counts = np.bincount(which, minlength=nbins)
    prof = prof / np.maximum(counts, 1)        # mean power per ring
    centers = 0.5 * (bins[:-1] + bins[1:])
    s = prof.sum()
    prof = prof / (s + 1e-300)                  # normalise to a distribution
    return centers, prof


def spectral_overlap(imfs, nbins=40):
    """
    Bhattacharyya overlap between the radial power spectra of *adjacent* IMFs.
    For each pair (i, i+1):  BC = sum_r sqrt( p_i(r) p_{i+1}(r) )  in [0,1].
    Low BC  -> adjacent IMFs occupy separate frequency bands (clean filter bank).
    High BC -> adjacent IMFs share a band  -> scale-leakage mode mixing.
    Returns (list_of_adjacent_BC, mean_BC).
    """
    profs = [radial_power_spectrum(c, nbins)[1] for c in imfs]
    bcs = []
    for i in range(len(profs) - 1):
        bc = float(np.sum(np.sqrt(profs[i] * profs[i + 1])))
        bcs.append(bc)
    mean_bc = float(np.mean(bcs)) if bcs else float("nan")
    return bcs, mean_bc


# ---------------------------------------------------------------------------
# 6. Per-IMF directional purity (structure-tensor coherence)
# ---------------------------------------------------------------------------
def imf_mean_coherence(imf, sigma_grad=1.0, sigma_tensor=2.0):
    """
    Coherence-weighted mean structure-tensor coherence of a single IMF. A
    directionally pure mode (one local orientation) has high coherence; an IMF
    that superimposes two orientations (mode mixing) has its coherence pulled
    down by the competing gradients.
    """
    from structure_tensor import orientation_and_coherence
    _, coh = orientation_and_coherence(imf, sigma_grad, sigma_tensor)
    return float(np.mean(coh))


def summarise(signal, imfs, residual, components=None):
    """Convenience: compute all standard metrics for one decomposition."""
    iec = index_of_energy_conservation(signal, imfs, residual)
    _, mean_bc = spectral_overlap(imfs)
    out = {
        "index_of_orthogonality": index_of_orthogonality(imfs, residual),
        "energy_ratio": iec["ratio"],
        "energy_leakage": iec["leakage"],
        "imf_mean_abs_corr": mean_abs_offdiag_correlation(imfs),
        "spectral_overlap_mean": mean_bc,
        "imf1_coherence": imf_mean_coherence(imfs[0]) if imfs else float("nan"),
    }
    return out
