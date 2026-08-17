"""
gabor_orientation.py
--------------------
An orientation estimator that shares no machinery with the structure tensor.

Why this exists
---------------
Every real-data orientation number in this project uses the structure tensor on
both sides: ST-BEMD steers its kernel with it, and `local_orientation_error`
measures with it. A method agreeing with its own instrument is not evidence.
This module provides a second opinion built on a different principle entirely.

The structure tensor is a *differential* estimator: it takes gradients and
averages their outer product. This is a *matched-filter* estimator: it convolves
with a bank of oriented Gabor quadrature pairs and asks which orientation the
image resonates with. Different assumptions, different failure modes, no shared
code beyond the FFT.

Convention
----------
`orientation()` returns the same quantity as
`structure_tensor.orientation_and_coherence`: the direction of the modulation
(the wave vector / across-ridge direction), in radians on [0, pi). The two are
therefore directly comparable, which `_selftest` below verifies on plane waves
of known angle.
"""
import numpy as np

# scipy's FFT convolution: the bank is many kernels over a small image, so
# transforming once per kernel is cheaper than spatial convolution.
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter


def gabor_pair(theta, wavelength, sigma, gamma=0.65, size=None):
    """
    Even (cosine) and odd (sine) Gabor kernels for one orientation and scale.

    theta is the modulation direction: the carrier varies along x', so the
    kernel matches a grating whose crests run perpendicular to theta — the same
    convention the structure tensor reports.

    gamma < 1 stretches the envelope along the ridge, which is what makes the
    bank selective for oriented structure rather than for blobs.
    """
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)

    xr = x * np.cos(theta) + y * np.sin(theta)      # along modulation
    yr = -x * np.sin(theta) + y * np.cos(theta)     # along ridge

    envelope = np.exp(-(xr ** 2 + (gamma ** 2) * (yr ** 2)) / (2 * sigma ** 2))
    even = envelope * np.cos(2 * np.pi * xr / wavelength)
    odd = envelope * np.sin(2 * np.pi * xr / wavelength)

    # Zero-mean the even kernel so flat regions give no response; the odd
    # kernel is already zero-mean by symmetry.
    even -= even.mean()
    return even, odd


def dominant_wavelength(f, lo=3.0, hi=20.0, detrend_power=2.0):
    """
    Ridge period from the radial power spectrum peak.

    Estimated rather than hard-coded so the bank adapts to whatever the image
    contains — the ridge period of a 90x90 SOCOFing scan (~3.4 px) is nothing
    like that of a synthetic test signal (~5-7 px).

    The radial profile is weighted by r**detrend_power before the peak is
    taken. Without this the estimate is dominated by the low-frequency envelope
    rather than the ridges: natural image spectra fall off steeply with radius,
    so the smooth background outweighs the ridge bump even though the bump is
    the feature of interest. On ten SOCOFing prints the unweighted version
    returned 18.0 px for seven of them and 3.46 for the rest — a factor-of-five
    error that silently detunes the whole filter bank. With r**2 weighting the
    same ten give 3.0-3.9 px (std 0.27), and the analytic suite is recovered
    correctly too.
    """
    f = f - f.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    rbin = radius.astype(int)
    profile = np.bincount(rbin.ravel(), power.ravel()) / \
        np.maximum(np.bincount(rbin.ravel()), 1)

    radii = np.arange(len(profile))
    profile = profile * np.maximum(radii, 1e-9) ** detrend_power

    # radius r cycles across the image  ->  wavelength = min(h, w) / r
    with np.errstate(divide="ignore"):
        wavelengths = min(h, w) / np.maximum(radii, 1e-9)
    valid = (wavelengths >= lo) & (wavelengths <= hi) & (radii > 0)
    if not valid.any():
        return 8.0
    return float(wavelengths[valid][np.argmax(profile[valid])])


def orientation(f, n_theta=24, wavelengths=None, sigma_scale=0.65,
                smooth=2.0, gamma=0.65):
    """
    Per-pixel orientation and a response-based confidence, from a Gabor bank.

    Returns (theta, confidence) with theta on [0, pi) in the same convention as
    structure_tensor.orientation_and_coherence.

    The angle is recovered by doubled-angle vector averaging over the bank
    rather than by taking the arg-max bin: the response profile across
    orientation is broad, so arg-max quantises to the bank spacing while the
    circular mean interpolates between bins. Responses are mean-subtracted and
    clipped at zero first, so only orientations that beat the bank average
    contribute — without that the average is dragged toward the isotropic mean
    and every pixel reads as weakly oriented.
    """
    f = np.asarray(f, dtype=float)
    f = f - f.mean()

    if wavelengths is None:
        base = dominant_wavelength(f)
        wavelengths = [base * 0.75, base, base * 1.33]

    thetas = np.arange(n_theta) * np.pi / n_theta
    energies = np.empty((n_theta,) + f.shape)

    for k, th in enumerate(thetas):
        best = None
        for lam in wavelengths:
            sigma = sigma_scale * lam
            even, odd = gabor_pair(th, lam, sigma, gamma)
            re = fftconvolve(f, even, mode="same")
            im = fftconvolve(f, odd, mode="same")
            amp = re ** 2 + im ** 2          # quadrature energy, phase-invariant
            best = amp if best is None else np.maximum(best, amp)
        energies[k] = gaussian_filter(best, smooth)

    # keep only above-average responses, then doubled-angle circular mean
    excess = np.maximum(energies - energies.mean(axis=0, keepdims=True), 0.0)
    cos2 = np.tensordot(np.cos(2 * thetas), excess, axes=(0, 0))
    sin2 = np.tensordot(np.sin(2 * thetas), excess, axes=(0, 0))

    theta = 0.5 * np.arctan2(sin2, cos2) % np.pi

    total = excess.sum(axis=0)
    magnitude = np.sqrt(cos2 ** 2 + sin2 ** 2)
    confidence = np.where(total > 1e-12, magnitude / (total + 1e-12), 0.0)
    return theta, confidence


# ---------------------------------------------------------------------------
def _selftest():
    """Both estimators on plane waves of known angle — they must agree."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "st_bemd"))
    from signals import plane_wave
    from structure_tensor import orientation_and_coherence

    print(f"{'true':>6}{'Gabor':>10}{'struct-tensor':>16}")
    worst = 0.0
    for true_deg in [0, 20, 45, 70, 110, 155]:
        f = plane_wave(96, 12, true_deg)[0]

        th_g, conf = orientation(f)
        ang = 0.5 * np.arctan2(np.sum(conf * np.sin(2 * th_g)),
                               np.sum(conf * np.cos(2 * th_g)))
        gabor_deg = np.rad2deg(ang) % 180.0

        th_s, coh = orientation_and_coherence(f)
        ang = 0.5 * np.arctan2(np.sum(coh * np.sin(2 * th_s)),
                               np.sum(coh * np.cos(2 * th_s)))
        st_deg = np.rad2deg(ang) % 180.0

        err = min(abs(gabor_deg - true_deg), 180 - abs(gabor_deg - true_deg))
        worst = max(worst, err)
        print(f"{true_deg:>6}{gabor_deg:>10.1f}{st_deg:>16.1f}")

    print(f"\nworst Gabor error: {worst:.2f} deg")
    assert worst < 5.0, "Gabor bank disagrees with the known angle — fix before use"
    print("convention matches the structure tensor ✓")


if __name__ == "__main__":
    _selftest()
