"""
stbemd_rbf.py
-------------
FAITHFUL implementation of the proposal's ST-BEMD: the envelope is built by
ANISOTROPIC RBF *INTERPOLATION* through the 2D extrema, with a per-extremum
kernel shaped by the structure tensor S_i:

    k_S(x_i, x_j) = phi( sqrt( (x_i - x_j)^T M_j (x_i - x_j) ) ),   phi(r)=r^2 log r,

where M_j is built from the structure tensor at extremum j, regularised as
S_j <- S_j + eps*I (the proposal's S_i + eps*I). This is the literal interpolation
form -- a dense M x M solve per envelope, O(M^3) -- as opposed to the windowed
local-average variant in stbemd.py.

KEY FINDING: this literal method performs far WORSE than isotropic BEMD on real
fingerprints (~42 deg vs ~12 deg). An eps-sweep shows the isotropic limit
(large eps) recovers standard-BEMD behaviour and it degrades monotonically as
anisotropy increases -- i.e. anisotropic RBF interpolation is ill-conditioned and
overshoots. Anisotropy is only safe in a bounded (averaging) envelope.

metric='Sinv' (faithful): M_j proportional to S_j^{-1} (elongates ACROSS ridge,
    as literally written). metric='S': geometric correction, elongates ALONG ridge.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from bemd import find_extrema_2d, _corner_anchor


def _tensor_fields(f, sigma_grad=1.0, sigma_tensor=2.0):
    fx = gaussian_filter(f, sigma_grad, order=(0, 1))
    fy = gaussian_filter(f, sigma_grad, order=(1, 0))
    Jxx = gaussian_filter(fx * fx, sigma_tensor)
    Jxy = gaussian_filter(fx * fy, sigma_tensor)
    Jyy = gaussian_filter(fy * fy, sigma_tensor)
    return Jxx, Jxy, Jyy


def _metrics_at(pts, Jxx, Jxy, Jyy, h, eps, metric, rho_cap):
    """Per-point 2x2 metric matrices M (in (x,y)=(col,row)), scaled to base width h."""
    Ms = []
    for (y, x) in pts:
        yi, xi = int(round(y)), int(round(x))
        yi = min(max(yi, 0), Jxx.shape[0] - 1); xi = min(max(xi, 0), Jxx.shape[1] - 1)
        S = np.array([[Jxx[yi, xi], Jxy[yi, xi]], [Jxy[yi, xi], Jyy[yi, xi]]])
        tr = np.trace(S) + 1e-12
        S = S + eps * tr * np.eye(2)                 # proposal's S + eps*I
        w, V = np.linalg.eigh(S)                     # w[0]<=w[1]
        l2, l1 = w[0], w[1]                          # l1 large (across-ridge)
        ratio = np.sqrt(l1 / max(l2, 1e-12))
        ratio = min(ratio, rho_cap)
        if metric == "Sinv":                         # faithful: elongate across ridge
            a_along_e1, a_along_e2 = h * ratio, h / ratio
        else:                                        # 'S': corrected, elongate along ridge
            a_along_e1, a_along_e2 = h / ratio, h * ratio
        e1, e2 = V[:, 1], V[:, 0]
        M = np.outer(e1, e1) / a_along_e1 ** 2 + np.outer(e2, e2) / a_along_e2 ** 2
        Ms.append(M)
    return Ms


def _phi(d2):
    out = np.zeros_like(d2)
    m = d2 > 1e-12
    out[m] = 0.5 * d2[m] * np.log(d2[m])             # thin-plate-spline r^2 log r
    return out


def _aniso_rbf_surface(mask, f, fields, grid_xy, shape, h, eps, metric, rho_cap, smoothing):
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([ys, xs]).astype(float)
    vals = f[ys, xs]
    cpts, cvals = _corner_anchor(f)
    pts = np.vstack([pts, cpts]); vals = np.concatenate([vals, cvals])
    _, keep = np.unique(pts, axis=0, return_index=True)
    pts, vals = pts[np.sort(keep)], vals[np.sort(keep)]
    n = len(pts)
    if n < 4:
        return np.full(shape, f.mean())
    Jxx, Jxy, Jyy = fields
    Ms = _metrics_at(pts, Jxx, Jxy, Jyy, h, eps, metric, rho_cap)
    P = pts[:, ::-1]                                  # (x,y) per centre
    A = np.zeros((n, n))
    for j in range(n):
        d = P - P[j]
        M = Ms[j]
        d2 = (d[:, 0] ** 2 * M[0, 0] + 2 * d[:, 0] * d[:, 1] * M[0, 1]
              + d[:, 1] ** 2 * M[1, 1])
        A[:, j] = _phi(d2)
    A += smoothing * np.eye(n)
    Pol = np.column_stack([np.ones(n), P[:, 0], P[:, 1]])
    K = np.zeros((n + 3, n + 3))
    K[:n, :n] = A; K[:n, n:] = Pol; K[n:, :n] = Pol.T
    rhs = np.concatenate([vals, np.zeros(3)])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    lam, c = sol[:n], sol[n:]
    gx, gy = grid_xy
    U = c[0] + c[1] * gx + c[2] * gy
    for j in range(n):
        M = Ms[j]
        dx = gx - P[j, 0]; dy = gy - P[j, 1]
        d2 = dx ** 2 * M[0, 0] + 2 * dx * dy * M[0, 1] + dy ** 2 * M[1, 1]
        U += lam[j] * _phi(d2)
    return U.reshape(shape)


def stbemd_rbf(signal, max_imfs=4, max_sift=8, sd_thresh=0.25, extrema_min=8,
               eps=0.3, metric="Sinv", rho_cap=4.0, smoothing=1.0,
               sigma_grad=1.0, sigma_tensor=2.0):
    """Faithful anisotropic-RBF-interpolation ST-BEMD. Returns (imfs, residual)."""
    signal = np.asarray(signal, dtype=float)
    N, Mw = signal.shape
    yy, xx = np.mgrid[0:N, 0:Mw]
    grid_xy = (xx.ravel().astype(float), yy.ravel().astype(float))
    residual = signal.copy(); imfs = []
    for _ in range(max_imfs):
        mx0, mn0 = find_extrema_2d(residual)
        if mx0.sum() < extrema_min or mn0.sum() < extrema_min:
            break
        h = residual.copy()
        for _s in range(max_sift):
            mx, mn = find_extrema_2d(h)
            if mx.sum() < extrema_min or mn.sum() < extrema_min:
                break
            fields = _tensor_fields(h, sigma_grad, sigma_tensor)
            hbw = float(np.clip(0.5 * np.sqrt(N * Mw / max(int((mx | mn).sum()), 1)), 1.5, 8.0))
            U = _aniso_rbf_surface(mx, h, fields, grid_xy, (N, Mw), hbw, eps, metric, rho_cap, smoothing)
            L = _aniso_rbf_surface(mn, h, fields, grid_xy, (N, Mw), hbw, eps, metric, rho_cap, smoothing)
            h_new = h - 0.5 * (U + L)
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_new
            if sd < sd_thresh:
                break
        imfs.append(h); residual = residual - h
    return imfs, residual


if __name__ == "__main__":
    import time
    from metrics import reconstruction_error, local_orientation_error
    fp = np.load("data/fingerprint.npy")
    for metric in ["Sinv", "S"]:
        t = time.time(); imfs, res = stbemd_rbf(fp, metric=metric)
        print(f"metric={metric:5} recon={reconstruction_error(fp,imfs,res):.1e} "
              f"orient_err={local_orientation_error(imfs[0],fp):.2f} deg "
              f"nimf={len(imfs)} time={time.time()-t:.1f}s")
