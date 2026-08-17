#!/usr/bin/env python3
"""
Check that every number quoted in README.md is backed by a file in outputs/.

This exists because the docs drifted once already: the README carried a runtime
table whose last two rows had no artifact behind them, and quoted a percentage
from the synthetic suite next to one from the real prints without saying which
was which. Both were the kind of error that a reader trusts and an author never
notices, because nothing was checking.

Each claim below names a number in the README and the path through the results
JSON it should equal. Run it after regenerating results; if a claim fails, fix
whichever of the two is wrong.

    python scripts/check_claims.py

Exit status is 0 when everything matches, 1 otherwise, so CI can gate on it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
README = ROOT / "README.md"

# Tolerance is relative. Timings are not reproducible across machines, so the
# runtime claims are checked loosely — enough to catch a stale table, not so
# tight that a slower runner fails the build.
TIGHT, LOOSE = 0.02, 0.60


def load(*parts: str):
    path = OUT.joinpath(*parts)
    if not path.exists():
        raise SystemExit(
            f"\n{path} is missing — regenerate the results first:\n"
            f"    python scripts/fetch_data.py && python run_all.py\n"
        )
    # encoding is explicit everywhere in this file: Path.read_text() defaults to
    # the locale encoding, which is cp1252 on Windows, and the README is full of
    # degree signs and Greek letters. Without this the script dies on Windows
    # after printing every claim as ok — which is the most confusing possible
    # way to fail.
    return json.loads(path.read_text(encoding="utf-8"))


def build_claims():
    bench = load("benchmark.json")
    full = load("results", "results_full.json")
    imf = load("results", "results_imf.json")

    fp = full["fingerprints"]
    summary = imf["summary"]["orient_err"]
    runtime = {r["size"]: r for r in bench["runtime"]}
    smooth = full["runtime"]
    orient = bench["orientation"]

    iso_syn = sum(orient["BEMD (global RBF)"]) / 3
    st_syn = sum(orient["ST-BEMD (proposed)"]) / 3

    claims = [
        # (label, README value, measured value, tolerance)
        ("synthetic mean, ST-BEMD", 1.83, st_syn, TIGHT),
        ("synthetic mean, isotropic BEMD", 3.08, iso_syn, TIGHT),
        ("synthetic reduction (%)", 40.6, (iso_syn - st_syn) / iso_syn * 100, TIGHT),
        ("synthetic, two orientations, ST", 5.14, orient["ST-BEMD (proposed)"][1], TIGHT),
        ("synthetic, two orientations, DEMD", 34.89, orient["DEMD (global rotate)"][1], TIGHT),

        ("100 prints, isotropic", 12.87, summary["iso_mean"], TIGHT),
        ("100 prints, ST-BEMD", 8.27, summary["st_mean"], TIGHT),
        ("100 prints, reduction (%)", 35.7,
         (summary["iso_mean"] - summary["st_mean"]) / summary["iso_mean"] * 100, TIGHT),
        ("100 prints, orthogonality iso", 0.119, imf["summary"]["orth_meancorr"]["iso_mean"], TIGHT),
        ("100 prints, orthogonality ST", 0.084, imf["summary"]["orth_meancorr"]["st_mean"], TIGHT),

        ("30 prints, isotropic", 13.71, fp["iso_mean"], TIGHT),
        ("30 prints, ST-BEMD", 8.25, fp["st_mean"], TIGHT),
        ("30 prints, % better", 100.0, fp["pct_st_better"], TIGHT),

        ("curvature high, isotropic", 19.91, full["curvature_bins"]["high"]["iso"], TIGHT),
        ("curvature high, ST-BEMD", 12.93, full["curvature_bins"]["high"]["st"], TIGHT),

        ("runtime 256^2 extrema", 8313, runtime[256]["extrema"], TIGHT),
        ("runtime 256^2 speedup", 31.4, runtime[256]["speedup"], LOOSE),
        ("runtime 192^2 speedup", 5.9, runtime[192]["speedup"], LOOSE),
        ("smooth-signal speedup at 256^2", 2.6,
         smooth["iso_s"][-1] / smooth["st_s"][-1], LOOSE),
    ]
    return claims


def check_readme_mentions(claims):
    """Every claimed value should literally appear in the README text."""
    # Strip thousands separators so "8,313" in a table matches the value 8313.
    text = re.sub(r"(?<=\d),(?=\d\d\d\b)", "",
                  README.read_text(encoding="utf-8"))
    missing = []
    for label, stated, _measured, _tol in claims:
        needle = f"{stated:g}"
        if not re.search(re.escape(needle) + r"(?!\d)", text):
            missing.append(f"{label} ({needle})")
    return missing


def main() -> int:
    claims = build_claims()
    width = max(len(c[0]) for c in claims)
    failures = 0

    print(f"{'claim':<{width}}  {'README':>9} {'measured':>9}  {'rel err':>8}")
    print("-" * (width + 32))
    for label, stated, measured, tol in claims:
        rel = abs(measured - stated) / max(abs(stated), 1e-12)
        ok = rel <= tol
        failures += not ok
        print(f"{label:<{width}}  {stated:>9.4g} {measured:>9.4g}  "
              f"{rel:>7.1%}  {'ok' if ok else 'MISMATCH'}")

    missing = check_readme_mentions(claims)
    if missing:
        print("\nValues not found anywhere in README.md:")
        for item in missing:
            print(f"  {item}")
        failures += len(missing)

    print()
    if failures:
        print(f"{failures} claim(s) failed — the README and outputs/ disagree.")
        return 1
    print(f"All {len(claims)} claims match the committed results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
