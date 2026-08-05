"""
ablation.py
-----------
THE decisive experiment. Separates the two variables that were confounded in the
headline result: (a) envelope method (global RBF interpolation vs local average)
and (b) anisotropy. The fair control for anisotropy is iso-LOCAL = stbemd with
rho_max=1 (same local machinery, circular kernel).

Reports on the fingerprint batch:
  iso-GLOBAL (RBF)   = bemd()                 -- Nunes 2003 baseline
  iso-LOCAL  (rho=1) = stbemd(rho_max=1.0)    -- same envelope, circular kernel
  aniso-LOCAL(rho=4) = stbemd(rho_max=4.0)    -- the proposed method

Findings (30 prints): total gain 11.84 -> 7.62 decomposes as
  local-vs-global envelope : ~3.57 deg (~84%)   <- FABEMD-class idea, NOT novel
  anisotropy itself        : ~0.65 deg (~16%), Wilcoxon p~0.045 (borderline)
and the anisotropy gain does NOT grow with curvature (low +0.97, med +0.67,
high -0.28) -- contradicting the proposed mechanism.

Requires data/fp_batch.npy (or edit to fp_batch100.npy).
"""
import numpy as np
from scipy import stats
from structure_tensor import orientation_and_coherence
from metrics import local_orientation_error as loe
from bemd import bemd
from stbemd import stbemd

batch = np.load("data/fp_batch.npy")

def fold(a): return np.minimum(np.abs(a), np.pi-np.abs(a))
def curv(th):
    c2, s2 = np.cos(2*th), np.sin(2*th)
    cy, cx = np.gradient(c2); sy, sx = np.gradient(s2)
    return 0.5*np.sqrt(cx**2+cy**2+sx**2+sy**2)
def emap(imf, thref):
    thi, _ = orientation_and_coherence(imf); return np.rad2deg(fold(thi-thref))
def wm(x, w): w = np.where(w > 0.05, w, 0.); return float(np.sum(w*x)/(np.sum(w)+1e-12))

# --- per-print three-way ---
ig = np.array([loe(bemd(fp)[0][0], fp) for fp in batch])
il = np.array([loe(stbemd(fp, rho_max=1.0)[0][0], fp) for fp in batch])
al = np.array([loe(stbemd(fp, rho_max=4.0)[0][0], fp) for fp in batch])
print("               mean+-std      vs iso-global   vs iso-local")
print(f"iso-GLOBAL (RBF)   {ig.mean():5.2f}+-{ig.std():.2f}")
print(f"iso-LOCAL  (rho=1) {il.mean():5.2f}+-{il.std():.2f}   d={ig.mean()-il.mean():+.2f}")
print(f"aniso-LOCAL(rho=4) {al.mean():5.2f}+-{al.std():.2f}   d={ig.mean()-al.mean():+.2f}      d={il.mean()-al.mean():+.2f}")
print(f"\nanisotropy effect (iso-local vs aniso-local): {(il-al).mean():.2f} deg, "
      f"better on {np.mean(al<il)*100:.0f}%, Wilcoxon p={stats.wilcoxon(il,al).pvalue:.3f}")
print("decomposition of total gain:")
print(f"  local-vs-global (envelope) : {ig.mean()-il.mean():.2f} deg")
print(f"  anisotropy (the claim)     : {il.mean()-al.mean():.2f} deg")

# --- curvature-binned proper ablation (iso-local vs aniso-local) ---
K, EL, EA, W = [], [], [], []
for fp in batch:
    thref, coh = orientation_and_coherence(fp); k = curv(thref); m = coh > 0.05
    il1 = stbemd(fp, rho_max=1.0)[0][0]; al1 = stbemd(fp, rho_max=4.0)[0][0]
    K.append(k[m]); EL.append(emap(il1, thref)[m]); EA.append(emap(al1, thref)[m]); W.append(coh[m])
K = np.concatenate(K); EL = np.concatenate(EL); EA = np.concatenate(EA); W = np.concatenate(W)
q1, q2 = np.quantile(K, [1/3, 2/3])
print("\nPROPER ablation binned by curvature (iso-local vs aniso-local):")
print(f"{'bin':8}{'iso-local':>11}{'aniso-local':>13}{'anisotropy gain':>17}")
for nm, mask in [('low', K <= q1), ('medium', (K > q1) & (K <= q2)), ('high', K > q2)]:
    a, b = wm(EL[mask], W[mask]), wm(EA[mask], W[mask])
    print(f"{nm:8}{a:>11.2f}{b:>13.2f}{a-b:>+17.2f}")
