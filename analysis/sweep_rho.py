"""
sweep_rho.py
------------
Explore the anisotropy ratio rho. Measures the coherence-weighted leading-mode
orientation error on a set of fingerprints for each rho, and the GAIN relative to
the fair isotropic-local baseline (rho = 1, same envelope machinery, circular kernel).

To try every integer 0..100 (slow): RHOS = list(range(1, 101)), or CLI:
    python3 sweep_rho.py 1 100 1        # start stop step

FINDING (10 prints, regenerated 2026-08-17): a shallow bowl, essentially flat from
rho 0.5 to 6 (~1 deg over the iso-local baseline of 7.49), degrading and slowing
past rho ~12. The lowest error on this set lands at rho = 0.5 -- a slightly
COMPRESSED kernel, not an elongated one. No magic value; the 'best' rho is unstable
across subsets (selection-on-the-test-set bias) so read the SHAPE, not the argmin.
The flatness is itself evidence that anisotropy is not doing much work.

Uses the first N_PRINTS of data/fp_batch100.npy -- run scripts/fetch_data.py first.
"""
import _bootstrap  # noqa: F401  (puts the source dirs on sys.path)
import sys, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from metrics import local_orientation_error as loe
from stbemd import stbemd
from _paths import FIGURES, fingerprint_batch

RHOS = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100]
N_PRINTS = 10

def main(rhos=RHOS, n=N_PRINTS):
    batch = fingerprint_batch(n, large=True)
    base = np.array([loe(stbemd(fp, rho_max=1.0)[0][0], fp) for fp in batch])
    print(f"iso-local baseline (rho=1): {base.mean():.2f} deg  (n={n} prints)\n")
    print(f"{'rho':>6}{'mean err':>10}{'gain vs iso-local':>20}{'time/print':>13}")
    means = []
    for r in rhos:
        t = time.time()
        errs = np.array([loe(stbemd(fp, rho_max=float(r))[0][0], fp) for fp in batch])
        dt = (time.time()-t)/n
        means.append(errs.mean())
        print(f"{r:>6}{errs.mean():>10.2f}{base.mean()-errs.mean():>+20.2f}{dt:>12.2f}s")
    best = rhos[int(np.argmin(means))]
    print(f"\nlowest error on this set at rho = {best} (biased by selecting on the eval set).")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(rhos, means, "o-")
    ax.axhline(base.mean(), ls="--", c="gray", label="iso-local (rho=1)")
    ax.set_xlabel(r"anisotropy ratio $\rho$"); ax.set_ylabel("orientation error (deg)")
    ax.set_xscale("log"); ax.set_title("orientation error vs anisotropy ratio"); ax.legend()
    out = FIGURES / "rho_sweep.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        a, b, s = (float(x) for x in sys.argv[1:])
        rr = [r for r in np.arange(a, b + 1e-9, s) if r > 0]
        main(rhos=rr)
    else:
        main()
