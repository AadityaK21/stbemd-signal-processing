"""
metrics.py
----------
Quantitative metrics used to benchmark the baseline 2D-EMD methods. These are
the metrics named in the proposal: reconstruction fidelity, mode orthogonality,
orientation error / invariance, and mode-mixing (separation) quality.
"""
import numpy as np


def reconstruction_error(signal, imfs, residual):
    """Relative l2 error of  signal vs  sum(imfs)+residual."""
    recon = np.sum(imfs, axis=0) + residual
    return np.linalg.norm(signal - recon) / (np.linalg.norm(signal) + 1e-12)


def orthogonality_index(imfs):
    """
    Mean absolute normalised cross-correlation over all IMF pairs.
    0 = perfectly orthogonal modes; larger = more mode leakage between IMFs.
    """
    n = len(imfs)
    if n < 2:
        return 0.0
    flat = [c.ravel() for c in imfs]
    norms = [np.linalg.norm(c) + 1e-12 for c in flat]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(abs(np.dot(flat[i], flat[j]) / (norms[i] * norms[j])))
    return float(np.mean(vals))


def orientation_error(true_deg, est_deg):
    """Absolute angular error on [0,180) folded into [0,90] degrees."""
    d = abs((est_deg - true_deg) % 180.0)
    return min(d, 180.0 - d)


def best_component_match(imf, components):
    """
    Largest absolute correlation between a recovered IMF and any ground-truth
    component. Used to quantify how cleanly a single component was isolated.
    """
    a = imf.ravel() - imf.mean()
    best = 0.0
    for c in components:
        b = c.ravel() - c.mean()
        r = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        best = max(best, r)
    return best


def separation_score(imfs, components):
    """
    For a multi-component ground truth, average over components of the best
    correlation achieved by any single IMF. 1.0 = each true component is
    captured by one IMF; lower = mode mixing.
    """
    scores = []
    for c in components:
        c0 = c.ravel() - c.mean()
        best = 0.0
        for imf in imfs:
            a = imf.ravel() - imf.mean()
            r = abs(np.dot(a, c0) / (np.linalg.norm(a) * np.linalg.norm(c0) + 1e-12))
            best = max(best, r)
        scores.append(best)
    return float(np.mean(scores))


def local_orientation_error(imf, ref_signal, sigma_grad=1.0, sigma_tensor=2.0,
                            coh_floor=0.05):
    """
    Mean per-pixel angular error (deg) between the local orientation of a
    recovered IMF and the local orientation of the reference (input) signal,
    weighted by the reference coherence. This is the metric that exposes
    DEMD's single-global-direction failure on spatially-varying-orientation
    signals: a method that imposes one direction cannot track a rotating field.
    """
    from structure_tensor import orientation_and_coherence
    th_ref, coh_ref = orientation_and_coherence(ref_signal, sigma_grad, sigma_tensor)
    th_imf, _ = orientation_and_coherence(imf, sigma_grad, sigma_tensor)
    # angular difference on [0,pi), folded to [0,pi/2]
    d = np.abs((th_imf - th_ref))
    d = np.minimum(d, np.pi - d)
    w = np.where(coh_ref > coh_floor, coh_ref, 0.0)
    if w.sum() < 1e-9:
        return float("nan")
    return float(np.rad2deg(np.sum(w * d) / np.sum(w)))
