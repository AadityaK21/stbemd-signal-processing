"""
signals.py
----------
Synthetic two-dimensional AM-FM test signals of the form

        f(x,y) = A(x,y) cos(phi(x,y))

These are the controlled ground truth for the benchmark. We can dial the
orientation, scale, number of superposed components, and the rate at which the
local orientation varies in space -- which is exactly the axis along which the
existing baselines are expected to succeed or fail.
"""
import numpy as np


def _grid(N):
    """Normalised coordinate grid on [0,1]^2 with shape (N,N)."""
    u = np.linspace(0, 1, N)
    X, Y = np.meshgrid(u, u)
    return X, Y


def plane_wave(N=96, freq=12.0, theta_deg=30.0, amp=1.0):
    """
    A single-orientation AM-FM component: a plane wave at angle theta.
    This is the simplest locally-coherent signal -- orientation is constant.
    Returns (signal, info-dict with the ground-truth orientation).
    """
    X, Y = _grid(N)
    th = np.deg2rad(theta_deg)
    phase = 2 * np.pi * freq * (X * np.cos(th) + Y * np.sin(th))
    f = amp * np.cos(phase)
    return f, {"theta_deg": theta_deg, "freq": freq}


def two_orientation(N=96, f1=8.0, t1=20.0, f2=14.0, t2=110.0):
    """
    Sum of two plane-wave components at different scales AND orientations.
    Ground-truth components are returned so we can measure mode separation.
    """
    c1, _ = plane_wave(N, f1, t1, amp=1.0)
    c2, _ = plane_wave(N, f2, t2, amp=0.8)
    f = c1 + c2
    return f, {"components": [c1, c2], "thetas": [t1, t2], "freqs": [f1, f2]}


def varying_orientation(N=96, freq=10.0, curvature=1.4):
    """
    A locally-coherent AM-FM signal whose oscillation direction rotates
    smoothly across space (curved phase). This is the class the proposal
    targets and where global-direction methods (DEMD) are expected to break.
    """
    X, Y = _grid(N)
    # phase whose gradient direction rotates across the field
    phase = 2 * np.pi * freq * (X + curvature * (Y - 0.5) * X)
    # gentle amplitude modulation
    A = 1.0 + 0.3 * np.cos(2 * np.pi * 1.5 * Y)
    f = A * np.cos(phase)
    return f, {"amplitude": A}


def add_noise(f, snr_db=20.0, seed=0):
    """Add white Gaussian noise at a target SNR (dB)."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(f ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return f + rng.normal(0, np.sqrt(p_noise), size=f.shape)


def rotate_signal(N, freq, theta_deg, amp=1.0):
    """Convenience: a fresh plane wave at a given orientation (for rotation tests)."""
    return plane_wave(N, freq, theta_deg, amp)[0]


if __name__ == "__main__":
    for name, fn in [("plane", plane_wave), ("two", two_orientation),
                     ("vary", varying_orientation)]:
        f = fn()[0]
        print(f"{name:6s} shape={f.shape} range=({f.min():.2f},{f.max():.2f})")
