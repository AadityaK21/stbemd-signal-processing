"""
analytic_signals.py
-------------------
Synthetic 2D AM-FM signals whose local orientation, instantaneous frequency, and
amplitude are known IN CLOSED FORM (not estimated from a structure tensor). This
removes the self-reference present in the fingerprint metric: the reference
orientation is the analytic gradient direction of the phase phi(x,y).

Each generator returns a dict with:
    f, carrier, bg, theta (analytic orientation, folded [0,pi)),
    nu (|grad phi|/(2pi)), amp (analytic amplitude A).
"""
import numpy as np

def _grid(N):
    xs = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, xs)
    return X, Y

def _pack(f, carrier, bg, phix, phiy, amp):
    theta = np.arctan2(phiy, phix) % np.pi
    nu = np.sqrt(phix**2 + phiy**2) / (2*np.pi)
    return {"f": f, "carrier": carrier, "bg": bg,
            "theta": theta, "nu": nu, "amp": amp}


def concentric_core(N=96, freq=9.0, am=True, background=True):
    X, Y = _grid(N); r = np.sqrt(X**2 + Y**2) + 1e-6
    phi = 2*np.pi*freq*r
    A = (0.6 + 0.4*np.cos(np.pi*X)*np.cos(np.pi*Y)) if am else np.ones_like(X)
    carrier = A*np.cos(phi)
    B = 0.7*np.exp(-((X-0.2)**2+(Y+0.1)**2)/0.7) if background else None
    f = carrier + (B if B is not None else 0)
    phix = 2*np.pi*freq*X/r; phiy = 2*np.pi*freq*Y/r
    return _pack(f, carrier, B, phix, phiy, A)


def curved_chirp(N=96, freq=7.0, a=0.6, am=False, background=True):
    X, Y = _grid(N)
    phi = 2*np.pi*freq*(X + a*Y**2)
    A = (0.6 + 0.4*np.cos(np.pi*Y)) if am else np.ones_like(X)
    carrier = A*np.cos(phi)
    B = 0.7*np.exp(-((X+0.1)**2+(Y-0.2)**2)/0.7) if background else None
    f = carrier + (B if B is not None else 0)
    phix = 2*np.pi*freq*np.ones_like(X); phiy = 2*np.pi*freq*2*a*Y
    return _pack(f, carrier, B, phix, phiy, A)


def rotating_amfm(N=96, freq=7.0, a=0.9, background=True):
    return curved_chirp(N, freq, a, am=True, background=background)


def gentle_rotation(N=96, freq=8.0, a=0.12, background=True):
    return curved_chirp(N, freq, a, am=False, background=background)


def whorl(N=96, freq=8.0, sep=0.35, background=True):
    X, Y = _grid(N)
    r1 = np.sqrt((X-sep)**2 + Y**2)+1e-6
    r2 = np.sqrt((X+sep)**2 + Y**2)+1e-6
    phi = 2*np.pi*freq*(r1 + r2)/2
    carrier = np.cos(phi)
    B = 0.6*np.exp(-(X**2+Y**2)/0.8) if background else None
    f = carrier + (B if B is not None else 0)
    phix = 2*np.pi*freq*0.5*((X-sep)/r1 + (X+sep)/r2)
    phiy = 2*np.pi*freq*0.5*(Y/r1 + Y/r2)
    return _pack(f, carrier, B, phix, phiy, np.ones_like(X))


SUITE = {
    "concentric_core (high curv)": concentric_core,
    "whorl (2 singular)":          whorl,
    "curved_chirp (FM)":           curved_chirp,
    "rotating_AMFM (AM+FM)":       rotating_amfm,
    "gentle_rotation (low curv)":  gentle_rotation,
}

if __name__ == "__main__":
    for name, gen in SUITE.items():
        d = gen(96)
        print(f"{name:30} theta-range {np.ptp(np.rad2deg(d['theta'])):.0f}deg  "
              f"nu-range {d['nu'].min():.2f}-{d['nu'].max():.2f}")
