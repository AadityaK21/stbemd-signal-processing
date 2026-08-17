"""
Import side-effect: make the analysis scripts runnable from anywhere.

The scripts in this directory import the research modules by bare name
(``from bemd import bemd``), which needs ``st_bemd/`` on ``sys.path``. Importing
this module first arranges that, so ``python analysis/ablation.py`` works from
the project root with no PYTHONPATH juggling.

    import _bootstrap  # noqa: F401
    from _paths import DATA, FIGURES
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "st_bemd"))

from _paths import bootstrap, ensure_outputs  # noqa: E402

bootstrap()
ensure_outputs()
