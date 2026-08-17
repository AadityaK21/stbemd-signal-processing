#!/usr/bin/env python3
"""
ST-BEMD benchmarking suite.

Runs the proposed method against all four baselines on the same inputs and
reports the two things the project claims: local orientation error, and runtime.

Why the runtime gap exists
--------------------------
Isotropic BEMD builds its envelope by fitting a **global** thin-plate-spline RBF
through every extremum. That means solving a dense linear system of size n x n
where n is the extremum count, which costs roughly O(n^3). ST-BEMD replaces it
with a **local** kernel-weighted average inside a +-3-sigma window, which costs
O(n * window).

So the speedup is not a constant — it grows with extremum density. On a smooth
synthetic chirp (a few hundred extrema) the two are comparable. On a fingerprint
(several thousand extrema) the gap is large. `--mode runtime` measures this
directly across densities rather than quoting a single number.

Usage
-----
    python benchmark.py                       # orientation error, synthetic
    python benchmark.py --mode runtime        # speedup vs extrema density
    python benchmark.py --mode all --size 128
    python benchmark.py --data data/fp_batch.npy   # real fingerprints
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import st_bemd as S

HERE = Path(__file__).parent
OUT = HERE / "outputs"


# --------------------------------------------------------------------------
# method registry — every entry returns (imfs, residual)
# --------------------------------------------------------------------------
def _demd(sig, **kw):
    imfs, res, _theta = S.demd(sig, **kw)      # demd also returns its angle
    return imfs, res


METHODS = {
    "ST-BEMD (proposed)": S.stbemd,
    "BEMD (global RBF)": S.bemd,
    "Pseudo-BEMD": S.pseudo_bemd,
    "DEMD (global rotate)": _demd,
    "Serial-EMD": S.serial_emd,
}


# --------------------------------------------------------------------------
# test data
# --------------------------------------------------------------------------
def _array(result):
    """Signal builders return either an array or (array, metadata) — normalise."""
    return np.asarray(result[0] if isinstance(result, tuple) else result, dtype=float)


def synthetic_suite(n: int, seed: int = 0):
    """Three signal families, hardest last."""
    return [
        ("plane wave", _array(S.plane_wave(N=n, freq=10.0, theta_deg=30.0))),
        ("two orientations", _array(S.two_orientation(N=n))),
        ("curved / rotating", _array(S.varying_orientation(N=n))),
    ]


def fingerprint_like(n: int, noise: float = 0.35, seed: int = 0) -> np.ndarray:
    """
    Synthetic stand-in with fingerprint-scale extremum density.

    The real SOCOFing scans are not redistributable, and extremum *density* is
    what drives the runtime comparison — so this reproduces that property:
    curved concentric ridges plus noise, giving several thousand extrema at
    256x256, the same order as a real print.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy = n * 0.45, n * 0.55
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ang = np.arctan2(yy - cy, xx - cx)
    f = np.sin(0.55 * r * (256.0 / n) + 2.5 * np.sin(ang))
    f = f + noise * rng.standard_normal((n, n))
    return f - f.mean()


def load_real(path: Path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None]
    return [(f"print {i}", a - a.mean()) for i, a in enumerate(arr)]


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def run_orientation(signals, max_imfs, max_sift):
    print("\n" + "=" * 78)
    print("ORIENTATION ERROR — lower is better (degrees)")
    print("=" * 78)
    print(f"{'method':<24s} " + " ".join(f"{n[:14]:>15s}" for n, _ in signals) + f"{'mean':>10s}")
    print("-" * 78)

    table = {}
    for name, fn in METHODS.items():
        errs = []
        for _label, sig in signals:
            try:
                imfs, _res = fn(sig, max_imfs=max_imfs, max_sift=max_sift)
                errs.append(float(S.local_orientation_error(imfs[0], sig)) if imfs else np.nan)
            except Exception:
                errs.append(np.nan)
        table[name] = errs
        cells = " ".join(f"{e:>15.2f}" for e in errs)
        print(f"{name:<24s} {cells} {np.nanmean(errs):>9.2f}")

    base = np.nanmean(table.get("BEMD (global RBF)", [np.nan]))
    prop = np.nanmean(table.get("ST-BEMD (proposed)", [np.nan]))
    if np.isfinite(base) and np.isfinite(prop) and base > 0:
        print("-" * 78)
        print(f"ST-BEMD vs isotropic BEMD: {base:.2f} -> {prop:.2f} deg "
              f"({(base - prop) / base * 100:+.1f}%)")
    return table


def run_runtime(sizes, max_imfs, max_sift):
    print("\n" + "=" * 78)
    print("RUNTIME — global RBF envelope vs local anisotropic envelope")
    print("=" * 78)
    print(f"{'size':>8s} {'extrema':>9s} {'BEMD (s)':>10s} {'ST-BEMD (s)':>12s} {'speedup':>9s}")
    print("-" * 78)

    rows = []
    for n in sizes:
        sig = fingerprint_like(n)
        mx, mn = S.find_extrema_2d(sig)
        n_ext = int(mx.sum() + mn.sum())

        t0 = time.perf_counter()
        S.bemd(sig, max_imfs=max_imfs, max_sift=max_sift)
        t_bemd = time.perf_counter() - t0

        t0 = time.perf_counter()
        S.stbemd(sig, max_imfs=max_imfs, max_sift=max_sift)
        t_st = time.perf_counter() - t0

        rows.append({"size": n, "extrema": n_ext, "bemd_s": t_bemd,
                     "stbemd_s": t_st, "speedup": t_bemd / t_st})
        print(f"{n:>6d}^2 {n_ext:>9d} {t_bemd:>10.2f} {t_st:>12.2f} {t_bemd / t_st:>8.1f}x")

    print("-" * 78)
    print("Speedup grows with extremum count: the global RBF solve is ~O(n^3),")
    print("the local anisotropic envelope is ~O(n). That is the whole point.")
    return rows


def run_sanity(n=64):
    """Perfect-reconstruction check — every method must satisfy sum(IMFs)+residual == signal."""
    print("\n" + "=" * 78)
    print("SANITY — perfect reconstruction (sum of IMFs + residual == input)")
    print("=" * 78)
    sig, _ = S.varying_orientation(N=n)
    ok_all = True
    for name, fn in METHODS.items():
        imfs, res = fn(sig, max_imfs=2, max_sift=3)
        err = float(np.abs(sig - (sum(imfs) + res)).max())
        ok = err < 1e-8
        ok_all &= ok
        print(f"  {name:<24s} {len(imfs)} IMFs   max |error| = {err:.2e}   {'OK' if ok else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ST-BEMD benchmarking suite")
    p.add_argument("--mode", choices=["orientation", "runtime", "sanity", "all"],
                   default="orientation")
    p.add_argument("--size", type=int, default=96, help="synthetic image side length")
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 192, 256],
                   help="sizes for the runtime sweep")
    p.add_argument("--max-imfs", type=int, default=2)
    p.add_argument("--max-sift", type=int, default=3)
    p.add_argument("--data", type=Path, default=None,
                   help="path to a .npy of real images (e.g. data/fp_batch.npy)")
    p.add_argument("--limit", type=int, default=10, help="max real images to use")
    p.add_argument("--save", action="store_true", help="write results to outputs/")
    args = p.parse_args(argv)

    report = {}

    if args.mode in ("sanity", "all"):
        report["reconstruction_ok"] = run_sanity()

    if args.mode in ("orientation", "all"):
        if args.data and args.data.exists():
            signals = load_real(args.data)[: args.limit]
            print(f"\nUsing {len(signals)} real images from {args.data}")
        else:
            if args.data:
                print(f"\n{args.data} not found — falling back to synthetic signals.")
            signals = synthetic_suite(args.size)
        report["orientation"] = {k: list(map(float, v))
                                 for k, v in run_orientation(
                                     signals, args.max_imfs, args.max_sift).items()}

    if args.mode in ("runtime", "all"):
        report["runtime"] = run_runtime(args.sizes, args.max_imfs, args.max_sift)

    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "benchmark.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"\nsaved -> {OUT / 'benchmark.json'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
