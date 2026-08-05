"""
sweep_rho.py
------------
Explore the anisotropy ratio rho. Measures the coherence-weighted leading-mode
orientation error on a set of fingerprints for each rho, and the GAIN relative to
the fair isotropic-local baseline (rho = 1, same envelope machinery, circular kernel).

To try every integer 0..100 (slow): RHOS = list(range(1, 101)), or CLI:
    python3 sweep_rho.py 1 100 1        # start stop step

FINDING: shallow bowl, best around rho 4-8 (~1 deg over iso-local); degrades and
gets slower past rho ~16. No magic value. The 'best' rho is unstable across
subsets -- selection-on-the-test-set bias -- so read the SHAPE, not the argmin.
Requires data/fp_batch.npy (or fp_batch100.npy -- edit the path).
"""
import sys, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from metrics import local_orientation_error as loe
from stbemd import stbemd

RHOS = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100]
N_PRINTS = 10
BATCH = "data/fp_batch100.npy"

def main(rhos=RHOS, n=N_PRINTS):
    batch = np.load(BATCH)[:n]
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
    fig.tight_layout(); fig.savefig("rho_sweep.png", dpi=130)
    print("saved rho_sweep.png")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        a, b, s = (float(x) for x in sys.argv[1:])
        rr = [r for r in np.arange(a, b + 1e-9, s) if r > 0]
        main(rhos=rr)
    else:
        main()
