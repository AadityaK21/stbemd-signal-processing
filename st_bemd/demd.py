"""
demd.py
-------
Directional EMD (Jha, Dutta Roy & Lall, 2009): pick a single global dominant
orientation for the whole field, then perform 1D EMD along that direction.

Here the dominant direction is estimated from the coherence-weighted structure
tensor (a robust stand-in for the Wold-decomposition direction). The field is
rotated so the dominant direction lies along the rows, decomposed row-wise, and
rotated back. Because a *single* direction is applied globally, DEMD is expected
to work on constant-orientation signals but to fail where the local orientation
varies across space -- exactly the regime the proposal targets.
"""
import numpy as np
from scipy.ndimage import rotate
from emd1d import emd1d
from structure_tensor import dominant_orientation_deg


def demd(signal, max_imfs=6, theta_deg=None, **kw):
    """Directional EMD. Returns (imfs, residual, theta_used)."""
    signal = np.asarray(signal, dtype=float)
    if theta_deg is None:
        theta_deg = dominant_orientation_deg(signal)

    # rotate so the dominant (wave-vector) direction aligns with the row axis.
    # reshape=False keeps the grid fixed so forward/back rotations stay aligned;
    # any interpolation loss is absorbed into the residual (defined below).
    rot = rotate(signal, angle=theta_deg, reshape=False, order=3, mode="reflect")

    # 1D EMD along each row of the rotated field
    per_row = [emd1d(rot[i], max_imfs=max_imfs, **kw) for i in range(rot.shape[0])]
    K = max(len(imfs) for imfs, _ in per_row)
    rot_imfs = [np.zeros_like(rot) for _ in range(K)]
    for i, (imfs, res) in enumerate(per_row):
        for k, c in enumerate(imfs):
            rot_imfs[k][i] = c

    # rotate each directional mode back onto the original grid
    def back(a):
        return rotate(a, angle=-theta_deg, reshape=False, order=3, mode="reflect")

    imfs = [back(c) for c in rot_imfs]
    # residual = everything the directional modes did not capture (exact recon)
    residual = signal - np.sum(imfs, axis=0)
    return imfs, residual, theta_deg


if __name__ == "__main__":
    from signals import plane_wave, varying_orientation
    from metrics import reconstruction_error
    f = plane_wave(80, 12, 35)[0]
    imfs, res, th = demd(f)
    print(f"plane: est_dir={th:.1f} n_imfs={len(imfs)} "
          f"recon_err={reconstruction_error(f, imfs, res):.2e}")
    g = varying_orientation(80)[0]
    imfs, res, th = demd(g)
    print(f"vary : est_dir={th:.1f} n_imfs={len(imfs)} "
          f"recon_err={reconstruction_error(g, imfs, res):.2e}")
