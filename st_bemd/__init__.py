"""
ST-BEMD — Direction-Adaptive Empirical Mode Decomposition for 2-D signals.

The research modules were written as a flat set of scripts that import each
other by bare name (``from bemd import find_extrema_2d``), which only worked
when the interpreter's working directory was the source folder. Rather than
rewrite working research code and risk changing its numerical behaviour, this
package puts its own directory on ``sys.path`` first, so the original imports
resolve unchanged while the whole thing is also importable as a normal package:

    from stbemd import stbemd, bemd, pseudo_bemd, demd, serial_emd

Methods
-------
``stbemd``       proposed: anisotropic, structure-tensor-oriented local envelope
``bemd``         baseline 1: isotropic BEMD (Nunes 2003), global thin-plate RBF
``pseudo_bemd``  baseline 2: 1-D EMD along rows/columns, recombined
``demd``         baseline 3: globally rotated to the dominant orientation
``serial_emd``   baseline 4: serialised 1-D EMD over the flattened image
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__version__ = "1.0.0"

# --- decomposition methods -------------------------------------------------
from bemd import bemd, find_extrema_2d                    # noqa: E402
from stbemd import stbemd                                 # noqa: E402
from pseudo_bemd import pseudo_bemd                       # noqa: E402
from demd import demd                                     # noqa: E402
from serial_emd import serial_emd                         # noqa: E402
from emd1d import emd1d                                   # noqa: E402

# --- analysis tools --------------------------------------------------------
from structure_tensor import (                            # noqa: E402
    structure_tensor, orientation_and_coherence, dominant_orientation_deg,
)
from riesz import riesz_transform, monogenic              # noqa: E402
from metrics import (                                     # noqa: E402
    local_orientation_error, orthogonality_index, reconstruction_error,
)

# --- test signals ----------------------------------------------------------
from signals import (                                     # noqa: E402
    plane_wave, two_orientation, varying_orientation, add_noise,
)

__all__ = [
    "stbemd", "bemd", "pseudo_bemd", "demd", "serial_emd", "emd1d",
    "find_extrema_2d", "structure_tensor", "orientation_and_coherence",
    "dominant_orientation_deg", "riesz_transform", "monogenic",
    "local_orientation_error", "orthogonality_index", "reconstruction_error",
    "plane_wave", "two_orientation", "varying_orientation", "add_noise",
]

# Names that collide with their module (``stbemd.stbemd``) resolve to the
# function above, which is what callers want.
