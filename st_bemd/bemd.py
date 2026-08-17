"""
bemd.py
-------
Genuine bidimensional EMD (Nunes et al., 2003): direct 2D sifting. At each
iteration we detect 2D local maxima and minima, fit an upper and lower envelope
*surface* through these scattered extrema by radial-basis-function (thin-plate)
interpolation, subtract the mean surface, and repeat.

Crucially the RBF kernel is ISOTROPIC: each extremum's influence spreads equally
in all directions. On oriented signals this makes the envelope "bleed" across
oscillation crests in the perpendicular direction -- the artifact the proposed
ST-BEMD is designed to remove. This file is therefore the primary baseline.
"""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.interpolate import RBFInterpolator


def find_extrema_2d(f, size=3):
    """2D local maxima and minima via neighbourhood comparison (size x size)."""
    mx = maximum_filter(f, size=size, mode="nearest")
    mn = minimum_filter(f, size=size, mode="nearest")
    maxima = (f == mx)
    minima = (f == mn)
    # discard pixels that are simultaneously max and min (flat regions)
    flat = maxima & minima
    maxima &= ~flat
    minima &= ~flat
    return maxima, minima


def _corner_anchor(f):
    """Image corners as anchors to stabilise the envelope near the border."""
    N, M = f.shape
    pts = np.array([[0, 0], [0, M - 1], [N - 1, 0], [N - 1, M - 1]], float)
    vals = np.array([f[0, 0], f[0, M - 1], f[N - 1, 0], f[N - 1, M - 1]], float)
    return pts, vals


def _envelope_surface(mask, f, grid_pts, shape):
    """Thin-plate-spline surface through the masked extrema (+ corner anchors)."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([ys, xs]).astype(float)
    vals = f[ys, xs]
    cpts, cvals = _corner_anchor(f)
    pts = np.vstack([pts, cpts])
    vals = np.concatenate([vals, cvals])
    # drop duplicate coordinates (corner anchors may coincide with extrema),
    # which would make the interpolation matrix singular
    _, keep = np.unique(pts, axis=0, return_index=True)
    pts, vals = pts[np.sort(keep)], vals[np.sort(keep)]
    if len(pts) < 4:
        return np.full(shape, f.mean())
    # tiny smoothing regularises near-degenerate extremum layouts
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-6)
    return rbf(grid_pts).reshape(shape)


def bemd(signal, max_imfs=4, max_sift=8, sd_thresh=0.25, extrema_min=8):
    """Isotropic BEMD. Returns (imfs, residual)."""
    signal = np.asarray(signal, dtype=float)
    N, M = signal.shape
    yy, xx = np.mgrid[0:N, 0:M]
    grid_pts = np.column_stack([yy.ravel(), xx.ravel()]).astype(float)

    residual = signal.copy()
    imfs = []
    for _ in range(max_imfs):
        maxima, minima = find_extrema_2d(residual)
        if maxima.sum() < extrema_min or minima.sum() < extrema_min:
            break
        h = residual.copy()
        for _s in range(max_sift):
            mx, mn = find_extrema_2d(h)
            if mx.sum() < extrema_min or mn.sum() < extrema_min:
                break
            U = _envelope_surface(mx, h, grid_pts, (N, M))
            L = _envelope_surface(mn, h, grid_pts, (N, M))
            mean = 0.5 * (U + L)
            h_new = h - mean
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_new
            if sd < sd_thresh:
                break
        imfs.append(h)
        residual = residual - h
    return imfs, residual


if __name__ == "__main__":
    import time
    from signals import two_orientation
    from metrics import reconstruction_error
    f = two_orientation(72)[0]
    t0 = time.time()
    imfs, res = bemd(f)
    print(f"n_imfs={len(imfs)} recon_err={reconstruction_error(f, imfs, res):.2e} "
          f"time={time.time()-t0:.1f}s")
