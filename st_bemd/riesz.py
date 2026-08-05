"""
riesz.py
--------
Riesz transform and the monogenic signal (proposal Component 3): the
rotation-invariant 2D generalisation of the 1D analytic signal. For a 2D field
f the Riesz transform is, in the frequency domain,

    R1_hat(w) = -i w_x/|w| f_hat(w),   R2_hat(w) = -i w_y/|w| f_hat(w).

The monogenic signal (f, R1 f, R2 f) then yields, per pixel,
    local amplitude    A     = sqrt(f^2 + R1^2 + R2^2)
    local phase        phi   = atan2( sqrt(R1^2+R2^2), f )
    local orientation  theta = atan2( R2, R1 )                (mod pi)

This replaces the direction-fixed Hilbert transform used to analyse isotropic-
BEMD IMFs, and gives an instantaneous amplitude/phase/orientation for each IMF
without fixing a direction in advance.
"""
import numpy as np


def riesz_transform(f):
    """Return (R1, R2), the two Riesz components of a real 2D field f."""
    f = np.asarray(f, dtype=float)
    H, W = f.shape
    F = np.fft.fft2(f)
    wy = np.fft.fftfreq(H).reshape(-1, 1) * 2 * np.pi
    wx = np.fft.fftfreq(W).reshape(1, -1) * 2 * np.pi
    wmag = np.sqrt(wx ** 2 + wy ** 2)
    wmag[0, 0] = 1.0                       # avoid 0/0 at DC; DC has no orientation
    H1 = -1j * wx / wmag
    H2 = -1j * wy / wmag
    R1 = np.real(np.fft.ifft2(F * H1))
    R2 = np.real(np.fft.ifft2(F * H2))
    return R1, R2


def monogenic(f):
    """
    Monogenic analysis of f. Returns dict with local amplitude, phase, and
    orientation (radians, folded to [0,pi)).
    """
    R1, R2 = riesz_transform(f)
    amp = np.sqrt(f ** 2 + R1 ** 2 + R2 ** 2)
    rmag = np.sqrt(R1 ** 2 + R2 ** 2)
    phase = np.arctan2(rmag, f)
    orient = np.arctan2(R2, R1) % np.pi
    return {"amplitude": amp, "phase": phase, "orientation": orient,
            "R1": R1, "R2": R2}


def instantaneous_amplitude(f):
    """Convenience: just the monogenic local amplitude (the AM envelope)."""
    R1, R2 = riesz_transform(f)
    return np.sqrt(f ** 2 + R1 ** 2 + R2 ** 2)


if __name__ == "__main__":
    # sanity: amplitude of A*cos(phi) should recover A for a narrow-band signal
    N = 96
    xs = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, xs)
    A = 0.6 + 0.4 * np.cos(np.pi * X) * np.cos(np.pi * Y)
    phi = 2 * np.pi * 8 * (X * np.cos(np.deg2rad(30)) + Y * np.sin(np.deg2rad(30)))
    f = A * np.cos(phi)
    est = instantaneous_amplitude(f)
    err = np.mean(np.abs(est - A)) / np.mean(A)
    print(f"monogenic amplitude rel-err vs true AM = {err:.3f}  (lower = better)")
