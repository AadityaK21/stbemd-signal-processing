#!/usr/bin/env python3
"""
Regenerate every analysis result and figure in outputs/.

Cross-platform replacement for the old run_all.sh, which was bash-only and so
did not run on Windows — where this project is actually developed.

    python run_all.py              # everything, ~4 minutes
    python run_all.py --quick      # skip the two slowest studies
    python run_all.py --list       # show what would run

Each script's stdout goes to outputs/logs/<name>.log; the console gets a
one-line pass/fail per script and a summary at the end. A failure in one script
does not stop the others — you want the whole picture, not the first error.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "outputs" / "logs"

# (path, needs fingerprint data, slow)
SCRIPTS = [
    ("analysis/ablation.py",              True,  False),
    ("analysis/stbemd_rbf.py",            True,  False),
    ("analysis/sweep_rho.py",             True,  True),
    ("analysis/eval_full.py",             True,  False),
    ("analysis/eval_imf_study.py",        True,  True),
    ("st_bemd/plot_fingerprint_results.py", True, False),
    ("st_bemd/experiments.py",            False, False),
    ("st_bemd/experiments_part2.py",      True,  False),
    ("st_bemd/experiments_stbemd.py",     True,  False),
]


def data_present() -> bool:
    return (ROOT / "data" / "fp_batch100.npy").exists()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true",
                        help="skip the two slowest studies (rho sweep, 100-print IMF)")
    parser.add_argument("--list", action="store_true",
                        help="print the scripts that would run, then exit")
    args = parser.parse_args(argv)

    selected = [(s, needs) for s, needs, slow in SCRIPTS if not (args.quick and slow)]

    if args.list:
        for script, needs in selected:
            print(f"  {script}{'   (needs data/)' if needs else ''}")
        return 0

    if not data_present() and any(needs for _, needs in selected):
        print("data/ is empty — most scripts will be skipped.\n"
              "Rebuild it first:  python scripts/fetch_data.py\n")

    LOGS.mkdir(parents=True, exist_ok=True)
    results, skipped = [], []

    for script, needs in selected:
        if needs and not data_present():
            skipped.append(script)
            print(f"  skip  {script}  (no data/)")
            continue

        name = Path(script).stem
        log = LOGS / f"{name}.log"
        print(f"  run   {script} ...", end="", flush=True)

        start = time.perf_counter()
        # sys.executable, not "python" — so this uses whichever interpreter
        # launched it, including a virtualenv that is not on PATH.
        proc = subprocess.run([sys.executable, str(ROOT / script)],
                              capture_output=True, text=True, cwd=ROOT)
        elapsed = time.perf_counter() - start
        log.write_text(proc.stdout + proc.stderr)

        ok = proc.returncode == 0
        results.append((script, ok, elapsed))
        print(f"\r  {'ok  ' if ok else 'FAIL'}  {script}  ({elapsed:.0f}s)"
              f"{'' if ok else f'  -> {log}'}")
        if not ok:
            for line in (proc.stdout + proc.stderr).strip().splitlines()[-3:]:
                print(f"          {line}")

    failed = [s for s, ok, _ in results if not ok]
    total = sum(t for _, _, t in results)
    print(f"\n{len(results) - len(failed)}/{len(results)} scripts ok "
          f"in {total:.0f}s.", end="")
    print(f"  {len(skipped)} skipped." if skipped else "")

    figures = len(list((ROOT / "outputs" / "figures").glob("*.png")))
    jsons = len(list((ROOT / "outputs" / "results").glob("*.json")))
    print(f"outputs/: {figures} figures, {jsons} result files")

    if failed:
        print("\nfailed: " + ", ".join(failed))
        return 1
    if not skipped:
        print("\nNow verify the docs still match:  python scripts/check_claims.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
