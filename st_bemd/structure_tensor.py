"""
structure_tensor.py
-------------------
The structure tensor S = G_sigma * (grad f)(grad f)^T gives, at each pixel,
the local dominant orientation and a coherence measure. Here we use it as an
*orientation-measurement* tool for the benchmark (e.g. to read off the
orientation of a recovered IMF and compare it to ground truth). It is also the
core ingredient of the proposed ST-BEMD method (Goal 3), so this code is reused
in the second half of the project.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def structure_tensor(f, sigma_grad=1.0, sigma_tensor=2.0):
    """
    Returns the three independent components (Jxx, Jxy, Jyy) of the windowed
    structure tensor field.
    """
    f = np.asarray(f, dtype=float)
    fx = gaussian_filter(f, sigma_grad, order=(0, 1))
    fy = gaussian_filter(f, sigma_grad, order=(1, 0))
    Jxx = gaussian_filter(fx * fx, sigma_tensor)
    Jxy = gaussian_filter(fx * fy, sigma_tensor)
    Jyy = gaussian_filter(fy * fy, sigma_tensor)
    return Jxx, Jxy, Jyy


def orientation_and_coherence(f, sigma_grad=1.0, sigma_tensor=2.0):
    """
    Per-pixel dominant orientation (radians, the direction of the *gradient*,
    i.e. perpendicular to the oscillation crests) and coherence c in [0,1].
    coherence c = ((l1-l2)/(l1+l2))^2.
    """
    Jxx, Jxy, Jyy = structure_tensor(f, sigma_grad, sigma_tensor)
    # eigen-analysis of a 2x2 symmetric matrix, closed form
    diff = Jxx - Jyy
    tmp = np.sqrt(diff ** 2 + 4 * Jxy ** 2)
    l1 = 0.5 * (Jxx + Jyy + tmp)   # larger eigenvalue
    l2 = 0.5 * (Jxx + Jyy - tmp)   # smaller eigenvalue
    # orientation of the dominant gradient direction
    theta = 0.5 * np.arctan2(2 * Jxy, diff)
    coherence = np.where((l1 + l2) > 1e-12, ((l1 - l2) / (l1 + l2 + 1e-12)) ** 2, 0.0)
    return theta, coherence


def dominant_orientation_deg(f, **kw):
    """
    A single global orientation estimate (degrees, in [0,180)) for an image,
    obtained as the coherence-weighted circular mean of the per-pixel
    orientations. The structure-tensor dominant eigenvector is the gradient
    (wave-vector) direction, matching the plane-wave 'theta' convention in
    signals.py, so no perpendicular rotation is applied.
    """
    theta, coh = orientation_and_coherence(f, **kw)
    # orientations live on [0,pi): use doubled-angle circular mean, weighted by coherence
    w = coh
    s = np.sum(w * np.sin(2 * theta))
    c = np.sum(w * np.cos(2 * theta))
    ang = 0.5 * np.arctan2(s, c)
    deg = np.rad2deg(ang) % 180.0
    return deg


if __name__ == "__main__":
    from signals import plane_wave
    for t in [0, 30, 75, 120]:
        f = plane_wave(96, 12, t)[0]
        est = dominant_orientation_deg(f)
        print(f"true={t:5.1f}  est={est:6.1f}")
