"""
Project paths, resolved from this file's own location.

Every script in the repo writes into the same place through this module, so
results land in ``outputs/`` no matter which directory you launch from. Before
this existed each script had its own idea of where things went — some wrote to
``.``, some to ``../figures``, and two carried absolute paths from the machine
they were first written on.

Usage from anywhere in the repo::

    from _paths import DATA, FIGURES, RESULTS, fingerprint_batch

``bootstrap()`` additionally puts ``st_bemd/`` and ``analysis/`` on ``sys.path``
so the modules' original bare-name imports (``from bemd import ...``) resolve
without the caller having to set PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "st_bemd"
ANALYSIS = ROOT / "analysis"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
RESULTS = OUTPUTS / "results"

_MISSING_DATA = (
    "\n{path} not found.\n"
    "The fingerprint scans are not bundled with the repo. Rebuild them with:\n"
    "    python scripts/fetch_data.py\n"
)


def bootstrap() -> None:
    """Put the source directories on sys.path, package dir first."""
    for directory in (ANALYSIS, PACKAGE):
        entry = str(directory)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def ensure_outputs() -> None:
    """Create the output tree. Safe to call repeatedly."""
    for directory in (OUTPUTS, FIGURES, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)


def load_data(name: str) -> np.ndarray:
    """
    Load an array from ``data/``, with an actionable message when it is absent.

    A bare FileNotFoundError here is unhelpful — the file is *meant* to be
    missing on a fresh clone, and the fix is one command.
    """
    path = DATA / name
    if not path.exists():
        raise SystemExit(_MISSING_DATA.format(path=path))
    return np.load(path)


def fingerprint(name: str = "fingerprint.npy") -> np.ndarray:
    """A single zero-mean print, for qualitative figures."""
    return load_data(name)


def fingerprint_batch(n: int | None = None, large: bool = False) -> np.ndarray:
    """
    The evaluation batch: 30 prints by default, 100 with ``large=True``.

    ``n`` truncates the batch, which is what the sweep scripts want when they
    trade sample size for runtime.
    """
    batch = load_data("fp_batch100.npy" if large else "fp_batch.npy")
    return batch[:n] if n is not None else batch


__all__ = [
    "ROOT", "PACKAGE", "ANALYSIS", "DATA", "OUTPUTS", "FIGURES", "RESULTS",
    "bootstrap", "ensure_outputs", "load_data", "fingerprint",
    "fingerprint_batch",
]
