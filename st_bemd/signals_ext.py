"""
signals_ext.py
--------------
Richer synthetic AM-FM test signals with *known* multidirectional structure,
added for the mid-semester review. The point of these signals is to make mode
mixing visible and measurable: each is a sum of components that differ in BOTH
scale and orientation, with the ground-truth components returned so that a
recovered IMF can be matched against them.

A well-behaved 2D-EMD acts as a dyadic filter bank, so components that differ in
scale should fall into *different* IMFs. If two ground-truth components show up
in the *same* IMF (or one component is split across IMFs), that is mode mixing,
and the returned components let us see and score it directly.
"""
import numpy as np
from signals import plane_wave, _grid


def multiscale_multiorientation(N=96, freqs=(4.0, 8.0, 16.0),
                                thetas=(25.0, 80.0, 135.0),
                                amps=(1.0, 0.9, 0.8)):
    """
    Sum of three plane waves at (roughly dyadic) DIFFERENT scales and DIFFERENT
    orientations. This is the canonical mode-mixing probe: scale separation says
    the three components should land in three different IMFs, while the distinct
    orientations let us *see* whether a method mixes directions within one mode.

    Returns (signal, info) where info['components'] = [c_coarse, c_mid, c_fine]
    ordered coarse -> fine, and info['thetas'], info['freqs'] the ground truth.
    """
    comps = []
    for f, t, a in zip(freqs, thetas, amps):
        c, _ = plane_wave(N, f, t, amp=a)
        comps.append(c)
    f = np.sum(comps, axis=0)
    return f, {"components": comps, "thetas": list(thetas),
               "freqs": list(freqs), "order": "coarse_to_fine"}


def two_scale_crossing(N=96, f_low=5.0, t_low=15.0, f_high=15.0, t_high=105.0):
    """
    Two crossing wave-trains at well-separated scales and near-orthogonal
    orientations. A minimal, clean mode-mixing test: a good decomposition puts
    the coarse train in a later IMF and the fine train in IMF-1.
    """
    c_lo, _ = plane_wave(N, f_low, t_low, amp=1.0)
    c_hi, _ = plane_wave(N, f_high, t_high, amp=0.9)
    return c_lo + c_hi, {"components": [c_lo, c_hi],
                         "thetas": [t_low, t_high], "freqs": [f_low, f_high]}


if __name__ == "__main__":
    f, info = multiscale_multiorientation()
    print("multiscale shape", f.shape, "n_components", len(info["components"]))
    for c, th, fr in zip(info["components"], info["thetas"], info["freqs"]):
        print(f"  component: freq={fr:5.1f}  theta={th:5.1f}  energy={np.sum(c**2):.1f}")
