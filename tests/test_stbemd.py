"""
Correctness tests for the ST-BEMD suite.

The properties that matter for an EMD implementation are structural, not
statistical: decompositions must reconstruct exactly, IMFs must be
zero-mean-ish oscillations, and the structure tensor must recover orientation
on signals where the answer is known analytically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import st_bemd as S  # noqa: E402


def _arr(r):
    return np.asarray(r[0] if isinstance(r, tuple) else r, dtype=float)


def _demd(sig, **kw):
    imfs, res, _ = S.demd(sig, **kw)
    return imfs, res


METHODS = {
    "stbemd": S.stbemd, "bemd": S.bemd, "pseudo_bemd": S.pseudo_bemd,
    "demd": _demd, "serial_emd": S.serial_emd,
}


@pytest.fixture(scope="module")
def curved():
    return _arr(S.varying_orientation(N=64))


# ------------------------------------------------------- reconstruction
@pytest.mark.parametrize("name", list(METHODS))
def test_perfect_reconstruction(name, curved):
    """
    The defining property of EMD: the decomposition is complete and lossless.
    sum(IMFs) + residual must equal the input to floating-point precision.
    If this fails the method is losing or inventing energy.
    """
    imfs, res = METHODS[name](curved, max_imfs=2, max_sift=3)
    assert np.abs(curved - (sum(imfs) + res)).max() < 1e-9


@pytest.mark.parametrize("name", list(METHODS))
def test_returns_at_least_one_imf(name, curved):
    imfs, _ = METHODS[name](curved, max_imfs=2, max_sift=3)
    assert len(imfs) >= 1
    assert all(i.shape == curved.shape for i in imfs)


def test_reconstruction_holds_on_noisy_input():
    sig = _arr(S.varying_orientation(N=64))
    noisy = S.add_noise(sig, snr_db=10.0, seed=1)
    noisy = _arr(noisy)
    imfs, res = S.stbemd(noisy, max_imfs=2, max_sift=3)
    assert np.abs(noisy - (sum(imfs) + res)).max() < 1e-9


# ------------------------------------------------------------- extrema
def test_find_extrema_on_a_known_field():
    """A single Gaussian bump has exactly one maximum, at its centre."""
    n = 41
    yy, xx = np.mgrid[0:n, 0:n]
    f = np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / 50.0)
    mx, mn = S.find_extrema_2d(f)
    assert mx[20, 20], "peak of the bump was not detected as a maximum"
    assert mx.sum() >= 1


# --------------------------------------------------- structure tensor
def test_structure_tensor_recovers_known_orientation():
    """
    On a plane wave the orientation is known exactly, so this is a real
    correctness check rather than a smoke test. Orientation is mod 180 deg.
    """
    for true_deg in (0.0, 30.0, 60.0, 120.0):
        sig = _arr(S.plane_wave(N=96, freq=8.0, theta_deg=true_deg))
        est = S.dominant_orientation_deg(sig)
        diff = abs((est - true_deg + 90) % 180 - 90)
        assert diff < 12.0, f"true {true_deg}, got {est} (off by {diff:.1f} deg)"


def test_coherence_is_high_for_plane_wave_and_low_for_noise():
    """Coherence must distinguish oriented structure from isotropic noise."""
    wave = _arr(S.plane_wave(N=96, freq=8.0, theta_deg=25.0))
    _, coh_wave = S.orientation_and_coherence(wave)
    noise = np.random.default_rng(0).standard_normal((96, 96))
    _, coh_noise = S.orientation_and_coherence(noise)
    assert coh_wave.mean() > coh_noise.mean()
    assert coh_wave.mean() > 0.5


def test_coherence_is_bounded():
    sig = _arr(S.varying_orientation(N=64))
    _, coh = S.orientation_and_coherence(sig)
    assert coh.min() >= -1e-9 and coh.max() <= 1.0 + 1e-9


# ---------------------------------------------------------- the claim
def test_stbemd_beats_isotropic_bemd_on_curved_structure():
    """
    The project's central claim. On a signal whose orientation rotates across
    space, the anisotropic local envelope should track it better than the
    isotropic global one. This is the test that would fail if the method were
    quietly broken.
    """
    sig = _arr(S.two_orientation(N=96))
    imfs_st, _ = S.stbemd(sig, max_imfs=2, max_sift=3)
    imfs_bm, _ = S.bemd(sig, max_imfs=2, max_sift=3)
    err_st = S.local_orientation_error(imfs_st[0], sig)
    err_bm = S.local_orientation_error(imfs_bm[0], sig)
    assert err_st < err_bm, f"ST-BEMD {err_st:.2f} deg did not beat BEMD {err_bm:.2f} deg"


def test_global_orientation_methods_lose_on_multi_orientation_signals():
    """
    DEMD rotates the whole image to one dominant angle. On a signal containing
    two different orientations that must do worse than a locally adaptive
    method — this documents *why* the proposed approach exists.
    """
    sig = _arr(S.two_orientation(N=96))
    imfs_st, _ = S.stbemd(sig, max_imfs=2, max_sift=3)
    imfs_d, _, _ = S.demd(sig, max_imfs=2, max_sift=3)
    assert S.local_orientation_error(imfs_st[0], sig) < \
           S.local_orientation_error(imfs_d[0], sig)


# ---------------------------------------------------------- determinism
def test_stbemd_is_deterministic():
    sig = _arr(S.varying_orientation(N=64))
    a, _ = S.stbemd(sig, max_imfs=2, max_sift=3)
    b, _ = S.stbemd(sig, max_imfs=2, max_sift=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_degenerate_inputs_do_not_crash():
    """Constant and tiny fields have no extrema — must degrade, not explode."""
    for sig in (np.zeros((32, 32)), np.ones((32, 32)), np.zeros((8, 8))):
        imfs, res = S.stbemd(sig, max_imfs=2, max_sift=3)
        assert np.all(np.isfinite(res))
        assert np.abs(sig - (sum(imfs) + res)).max() < 1e-9 if imfs else True


# ------------------------------------------------------------- metrics
def test_orientation_error_is_zero_against_itself():
    sig = _arr(S.plane_wave(N=64, freq=8.0, theta_deg=40.0))
    assert S.local_orientation_error(sig, sig) < 1e-6


def test_orientation_error_is_angle_folded():
    """Orientation is defined mod 180 deg, so a sign flip must not register."""
    sig = _arr(S.plane_wave(N=64, freq=8.0, theta_deg=40.0))
    assert S.local_orientation_error(-sig, sig) < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--no-header"]))
