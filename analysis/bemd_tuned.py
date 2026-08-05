"""
bemd_tuned.py
-------------
Isotropic BEMD identical to bemd.py but with two exposed knobs so the standard
baseline can be tuned fairly:
  neighbors : if not None, RBFInterpolator uses only the k nearest extrema (local
              RBF) instead of a single global thin-plate-spline solve.
  smoothing : RBF smoothing/regularisation (0 = pure interpolation).

FINDING: tuning (local neighbors, larger smoothing) does NOT rescue standard
BEMD on real fingerprints -- every RBF-interpolation variant sits at ~12-15 deg.
The large accuracy gain comes from switching the envelope PARADIGM
(interpolation -> local averaging), not from tuning the RBF.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from bemd import find_extrema_2d, _corner_anchor


def _env(mask, f, grid_pts, shape, neighbors, smoothing):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([ys, xs]).astype(float)
    vals = f[ys, xs]
    cpts, cvals = _corner_anchor(f)
    pts = np.vstack([pts, cpts]); vals = np.concatenate([vals, cvals])
    _, keep = np.unique(pts, axis=0, return_index=True)
    pts, vals = pts[np.sort(keep)], vals[np.sort(keep)]
    if len(pts) < 4:
        return np.full(shape, f.mean())
    k = None if neighbors is None else min(neighbors, len(pts))
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                          smoothing=smoothing, neighbors=k)
    return rbf(grid_pts).reshape(shape)


def bemd_tuned(signal, max_imfs=4, max_sift=8, sd_thresh=0.25, extrema_min=8,
               neighbors=None, smoothing=1e-6):
    signal = np.asarray(signal, dtype=float)
    N, M = signal.shape
    yy, xx = np.mgrid[0:N, 0:M]
    grid_pts = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)
    residual = signal.copy(); imfs = []
    for _ in range(max_imfs):
        mx0, mn0 = find_extrema_2d(residual)
        if mx0.sum() < extrema_min or mn0.sum() < extrema_min:
            break
        h = residual.copy()
        for _s in range(max_sift):
            mx, mn = find_extrema_2d(h)
            if mx.sum() < extrema_min or mn.sum() < extrema_min:
                break
            U = _env(mx, h, grid_pts, (N, M), neighbors, smoothing)
            L = _env(mn, h, grid_pts, (N, M), neighbors, smoothing)
            h_new = h - 0.5 * (U + L)
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_new
            if sd < sd_thresh:
                break
        imfs.append(h); residual = residual - h
    return imfs, residual
