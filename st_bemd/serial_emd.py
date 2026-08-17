"""
serial_emd.py
-------------
Serial-EMD: serialise the 2D field into a 1D sequence, run standard 1D EMD, then
reshape the IMFs back to 2D. Reconstruction is exact (only reshaping is involved),
but the result depends on the *scan order*, which silently encodes a directional
assumption -- the artifact this baseline is meant to expose.
"""
import numpy as np
from emd1d import emd1d


def _order_index(shape, order):
    """Return (flatten_idx, inverse_idx) for a given scan order."""
    N, M = shape
    grid = np.arange(N * M).reshape(N, M)
    if order == "row":          # row-major
        flat = grid.ravel()
    elif order == "col":        # column-major
        flat = grid.T.ravel()
    elif order == "snake":      # boustrophedon: alternate row direction
        rows = []
        for i in range(N):
            r = grid[i]
            rows.append(r if i % 2 == 0 else r[::-1])
        flat = np.concatenate(rows)
    elif order == "diag":       # anti-diagonal scan
        flat = np.concatenate([np.diagonal(grid[:, ::-1], off)
                               for off in range(-(N - 1), M)])
    else:
        raise ValueError(order)
    inv = np.argsort(flat)
    return flat, inv


def serial_emd(signal, order="row", max_imfs=6, **kw):
    """Serial-EMD with a chosen scan order. Returns (imfs, residual)."""
    signal = np.asarray(signal, dtype=float)
    shape = signal.shape
    flat, inv = _order_index(shape, order)
    seq = signal.ravel()[flat]
    imfs1d, res1d = emd1d(seq, max_imfs=max_imfs, **kw)
    imfs = [c[inv].reshape(shape) for c in imfs1d]
    residual = res1d[inv].reshape(shape)
    return imfs, residual


if __name__ == "__main__":
    from signals import two_orientation
    from metrics import reconstruction_error
    f = two_orientation(80)[0]
    for o in ["row", "col", "snake", "diag"]:
        imfs, res = serial_emd(f, order=o)
        print(f"order={o:5s} n_imfs={len(imfs)} recon_err={reconstruction_error(f, imfs, res):.2e}")
