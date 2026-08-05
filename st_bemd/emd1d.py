"""
emd1d.py
--------
A from-scratch implementation of one-dimensional Empirical Mode Decomposition
(Huang et al., 1998). Deliberately written in a transparent, step-by-step style:
extrema detection -> spline envelopes -> local-mean subtraction -> sifting -> IMF.

No external EMD library is used. Only numpy + scipy's cubic spline are imported,
and the sifting logic is written out explicitly so the mechanics are visible.

This module is the shared engine for the three "1D-based" baseline 2D methods
(pseudo-BEMD, serial-EMD, DEMD).
"""
import numpy as np
from scipy.interpolate import CubicSpline


def find_extrema(x):
    """Return indices of local maxima and minima of a 1D array (strict interior)."""
    dx = np.diff(x)
    # sign changes of the first difference mark turning points
    # a maximum: slope goes + -> - ; a minimum: - -> +
    maxima, minima = [], []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            maxima.append(i)
        elif x[i] < x[i - 1] and x[i] <= x[i + 1]:
            minima.append(i)
    return np.array(maxima, dtype=int), np.array(minima, dtype=int)


def _extend_extrema(idx, vals, n):
    """
    Simple mirror-style boundary extension to suppress spline end effects.
    We reflect the first/last extremum about the signal boundary so that the
    cubic spline does not swing wildly near the edges.
    """
    if len(idx) == 0:
        return idx, vals
    idx = list(idx); vals = list(vals)
    # left side
    left_i = -idx[0]
    left_v = vals[0]
    # right side
    right_i = 2 * (n - 1) - idx[-1]
    right_v = vals[-1]
    idx = [left_i] + idx + [right_i]
    vals = [left_v] + vals + [right_v]
    return np.array(idx), np.array(vals)


def _envelope(idx, vals, n):
    """Cubic-spline envelope through the given extrema, sampled on 0..n-1."""
    ei, ev = _extend_extrema(idx, vals, n)
    cs = CubicSpline(ei, ev)
    return cs(np.arange(n))


def sift_once(h):
    """One sifting iteration: subtract the mean of the upper/lower envelopes."""
    n = len(h)
    maxima, minima = find_extrema(h)
    if len(maxima) < 1 or len(minima) < 1:
        return h, 0  # cannot sift further
    upper = _envelope(maxima, h[maxima], n)
    lower = _envelope(minima, h[minima], n)
    mean = 0.5 * (upper + lower)
    return h - mean, len(maxima) + len(minima)


def emd1d(signal, max_imfs=8, sd_thresh=0.2, max_sift=50):
    """
    Decompose a 1D signal into IMFs + residual.

    Stopping for each IMF uses Huang's Cauchy-type standard-deviation criterion
        SD = sum( (h_{k-1}-h_k)^2 / h_{k-1}^2 )  <  sd_thresh
    capped at max_sift iterations to guarantee termination.

    Returns: imfs (list of arrays), residual (array). signal = sum(imfs)+residual.
    """
    signal = np.asarray(signal, dtype=float)
    residual = signal.copy()
    imfs = []

    for _ in range(max_imfs):
        # if the residual is monotonic (fewer than 2 extrema total) we stop
        maxima, minima = find_extrema(residual)
        if len(maxima) + len(minima) < 2:
            break

        h = residual.copy()
        for _s in range(max_sift):
            h_prev = h
            h, n_ext = sift_once(h)
            if n_ext == 0:
                break
            denom = np.sum(h_prev ** 2) + 1e-12
            sd = np.sum((h_prev - h) ** 2) / denom
            if sd < sd_thresh:
                break
        imfs.append(h)
        residual = residual - h

    return imfs, residual


if __name__ == "__main__":
    # quick self-test: a 2-tone AM-FM-like 1D signal must reconstruct
    t = np.linspace(0, 1, 600)
    s = (1 + 0.3 * np.cos(2 * np.pi * 2 * t)) * np.cos(2 * np.pi * 40 * t) \
        + 0.7 * np.cos(2 * np.pi * 8 * t) + 0.2 * t
    imfs, res = emd1d(s)
    recon = np.sum(imfs, axis=0) + res
    err = np.linalg.norm(s - recon) / np.linalg.norm(s)
    print(f"n_imfs={len(imfs)}  reconstruction rel-L2 error={err:.2e}")
