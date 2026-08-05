"""
pseudo_bemd.py
--------------
Pseudo-BEMD: the simplest 2D extension, applying 1D EMD separably along the
coordinate axes (rows first, then columns). It is "pseudo" because it never does
genuine 2D sifting -- the decomposition is tied to the chosen coordinate frame,
which is the source of its orientation (axis) bias.

Separable scheme used here (reconstructs the input exactly):
  1. Row pass:    decompose every row -> index-aligned row-IMF images R_k.
  2. Column pass: decompose every column of each R_k; the column detail is kept
                  as the refined 2D IMF, the column trend is pushed to residual.
"""
import numpy as np
from emd1d import emd1d


def _row_decompose(signal, **kw):
    """Decompose every row; return index-aligned IMF images and a residual image."""
    N, M = signal.shape
    per_row = [emd1d(signal[i], **kw) for i in range(N)]
    K = max(len(imfs) for imfs, _ in per_row)
    imf_imgs = [np.zeros_like(signal) for _ in range(K)]
    res_img = np.zeros_like(signal)
    for i, (imfs, res) in enumerate(per_row):
        for k, c in enumerate(imfs):
            imf_imgs[k][i] = c
        res_img[i] = res
    return imf_imgs, res_img


def pseudo_bemd(signal, max_imfs=6, **kw):
    """Separable pseudo-BEMD. Returns (imfs, residual)."""
    signal = np.asarray(signal, dtype=float)
    kw.setdefault("max_imfs", max_imfs)

    row_imfs, row_res = _row_decompose(signal, **kw)
    final_imfs = []
    residual = row_res.copy()

    for Rk in row_imfs:
        # column pass on this row-IMF image: keep column detail, shed column trend
        col_detail = np.zeros_like(Rk)
        col_trend = np.zeros_like(Rk)
        for j in range(Rk.shape[1]):
            imfs_c, res_c = emd1d(Rk[:, j], **kw)
            if imfs_c:
                col_detail[:, j] = np.sum(imfs_c, axis=0)
            col_trend[:, j] = res_c
        final_imfs.append(col_detail)
        residual = residual + col_trend

    return final_imfs, residual


if __name__ == "__main__":
    from signals import two_orientation
    from metrics import reconstruction_error
    f = two_orientation(80)[0]
    imfs, res = pseudo_bemd(f)
    print(f"n_imfs={len(imfs)} recon_err={reconstruction_error(f, imfs, res):.2e}")
