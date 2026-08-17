"""
stbemd.py
---------
Structure-Tensor-guided Bidimensional EMD (the proposed method, Goal 3).

This is a drop-in modification of isotropic BEMD (bemd.py): the sifting loop is
identical, but the envelope surface is built with an ANISOTROPIC kernel whose
shape is set by the local structure tensor. At every pixel the kernel is
elongated ALONG the local ridge (the coherent oscillation crest) and compressed
ACROSS it, so the upper/lower envelopes average many extrema belonging to the
*same* crest without bleeding onto the neighbouring crest.

Relation to the proposal's kernel
----------------------------------
The proposal writes k_S(x_i,x_j)=phi( sqrt( (x_i-x_j)^T S_i^{-1} (x_i-x_j) ) )
with S_i the structure tensor. Taken literally with the raw structure tensor
this elongates the kernel along the tensor's MAJOR eigenvector (the gradient /
across-crest direction) -- which makes bleeding worse, not better. The geometry
the proposal actually wants ("elongated along the dominant local direction,
compressed perpendicular") is obtained by elongating along the MINOR eigenvector
(the ridge). We therefore build the metric from the structure-tensor orientation
and a coherence-driven anisotropy ratio, elongating along the ridge. At zero
coherence the kernel becomes isotropic, so the method degrades gracefully to
ordinary BEMD exactly where orientation is undefined (the eps-regularisation the
proposal asks for, achieved through the coherence weighting).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from bemd import find_extrema_2d
from structure_tensor import orientation_and_coherence


def _aniso_envelope(mask, f, theta, coh, bw, rho_max, elong_along_ridge=True):
    """
    Anisotropic kernel-weighted envelope through the masked extrema.

    At pixel x the envelope is a normalised weighted average of nearby extrema,
    U(x) = sum_i v_i w_i(x) / sum_i w_i(x), with an oriented Gaussian weight
        w_i(x) = exp(-0.5 [ proj_long^2 / sig_long(x)^2 + proj_short^2 / sig_short(x)^2 ])
    where the long axis is the local ridge direction and
        sig_short = bw,  sig_long = bw * (1 + (rho_max-1) * coh(x)).
    A spatial window of +-3*sig_long around each extremum keeps the cost low.
    """
    H, W = f.shape
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return np.full((H, W), float(f.mean()))
    vals = f[ys, xs]

    # per-pixel long-axis unit vector (in array coords: axis0=row=y, axis1=col=x)
    # theta is the gradient (across-ridge) direction; ridge = theta + 90 deg.
    ang = theta + (np.pi / 2.0 if elong_along_ridge else 0.0)
    # gradient unit vector in (row,col): (sin theta, cos theta); rotate accordingly
    ux = np.cos(ang)          # component along columns (x)
    uy = np.sin(ang)          # component along rows (y)
    # short axis is perpendicular
    vx, vy = -uy, ux

    sig_short = float(bw)
    sig_long = bw * (1.0 + (rho_max - 1.0) * coh)     # HxW field
    half = int(np.ceil(3.0 * bw * rho_max))

    num = np.zeros((H, W))
    den = np.zeros((H, W))
    for (r0, c0, v0) in zip(ys, xs, vals):
        r1, r2 = max(0, r0 - half), min(H, r0 + half + 1)
        c1, c2 = max(0, c0 - half), min(W, c0 + half + 1)
        rr, cc = np.mgrid[r1:r2, c1:c2]
        dr = (rr - r0).astype(float)        # row offset (y)
        dc = (cc - c0).astype(float)        # col offset (x)
        # project offset onto local long/short axes (use field at the pixel)
        ul = ux[r1:r2, c1:c2]; vl = uy[r1:r2, c1:c2]
        proj_long = dc * ul + dr * vl
        proj_short = dc * (-vl) + dr * (ul)   # perpendicular
        sl = sig_long[r1:r2, c1:c2]
        w = np.exp(-0.5 * (proj_long ** 2 / (sl ** 2 + 1e-9)
                           + proj_short ** 2 / (sig_short ** 2 + 1e-9)))
        num[r1:r2, c1:c2] += v0 * w
        den[r1:r2, c1:c2] += w
    env = np.where(den > 1e-9, num / (den + 1e-12), f.mean())
    # light smoothing to remove residual graininess of the scattered average
    return gaussian_filter(env, 0.6)


def _scale_estimate(mask, shape):
    """Rough extremum spacing: sqrt(area / n_extrema), clamped."""
    n = max(int(mask.sum()), 1)
    s = np.sqrt(shape[0] * shape[1] / n)
    return float(np.clip(0.5 * s, 1.5, 8.0))


def stbemd(signal, max_imfs=4, max_sift=8, sd_thresh=0.25, extrema_min=8,
           rho_max=4.0, elong_along_ridge=True, sigma_grad=1.0, sigma_tensor=2.0):
    """
    Structure-Tensor-guided BEMD. Returns (imfs, residual).
    rho_max: maximum anisotropy ratio (long/short kernel axis) at full coherence.
    elong_along_ridge: elongate along the ridge (True, correct) vs across (False).
    """
    signal = np.asarray(signal, dtype=float)
    H, W = signal.shape
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
            # orientation field of the CURRENT iterate (recomputed each sift)
            theta, coh = orientation_and_coherence(h, sigma_grad, sigma_tensor)
            bw = _scale_estimate(mx | mn, (H, W))
            U = _aniso_envelope(mx, h, theta, coh, bw, rho_max, elong_along_ridge)
            L = _aniso_envelope(mn, h, theta, coh, bw, rho_max, elong_along_ridge)
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
    from signals import plane_wave, two_orientation
    from metrics import reconstruction_error, separation_score
    # correctness + a quick separation check vs isotropic
    f, info = two_orientation(72)
    t0 = time.time()
    imfs, res = stbemd(f)
    print(f"[two-orientation] n_imfs={len(imfs)} "
          f"recon_err={reconstruction_error(f, imfs, res):.2e} "
          f"sep={separation_score(imfs, info['components']):.3f} "
          f"time={time.time()-t0:.1f}s")
