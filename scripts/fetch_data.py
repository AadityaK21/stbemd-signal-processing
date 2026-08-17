#!/usr/bin/env python3
"""
Rebuild the fingerprint arrays this project evaluates on.

The SOCOFing scans are not redistributable, so ``data/`` ships empty and is
git-ignored. This script fetches a public mirror of the same scans and writes
the three arrays the analysis scripts expect:

    data/fingerprint.npy    one print, 90x90, zero-mean   (qualitative figures)
    data/fp_batch.npy       30 prints                     (eval_full, ablation)
    data/fp_batch100.npy    100 prints                    (sweep_rho, imf study)

Selection is seeded (numpy default_rng(42)) and low-contrast prints are dropped,
so repeated runs on any machine produce byte-identical arrays.

Usage
-----
    python scripts/fetch_data.py              # skips work if files already exist
    python scripts/fetch_data.py --force      # rebuild from scratch
    python scripts/fetch_data.py --keep-raw   # leave the downloaded .npz in place
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REPO = "https://github.com/kairess/fingerprint_recognition.git"
TARBALL = "https://codeload.github.com/kairess/fingerprint_recognition/tar.gz/master"
MEMBER = "x_real.npz"

SEED = 42
POOL = 300          # candidates drawn before filtering
WANTED = 100        # prints kept for the large batch
SMALL = 30          # prints kept for the small batch
MIN_STD = 0.12      # contrast floor — drops blank / near-blank scans

OUTPUTS = ("fingerprint.npy", "fp_batch.npy", "fp_batch100.npy")


# ---------------------------------------------------------------------------
# acquiring the raw archive
# ---------------------------------------------------------------------------
def _via_tarball(dest: Path) -> bool:
    """Stream the repo tarball and extract just the one member we need."""
    try:
        print(f"  trying tarball  {TARBALL}")
        raw = urllib.request.urlopen(TARBALL, timeout=180).read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  tarball unavailable ({exc}) — falling back to git")
        return False

    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(MEMBER)]
        if not members:
            print(f"  tarball has no {MEMBER} — falling back to git")
            return False
        extracted = tar.extractfile(members[0])
        if extracted is None:
            return False
        dest.write_bytes(extracted.read())
    return True


def _via_git(dest: Path) -> bool:
    """Shallow-clone the repo and copy the archive out."""
    if shutil.which("git") is None:
        print("  git not on PATH")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        print(f"  cloning        {REPO}")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", REPO, tmp + "/repo"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  clone failed: {proc.stderr.strip().splitlines()[-1:]}")
            return False

        found = list(Path(tmp, "repo").rglob(MEMBER))
        if not found:
            print(f"  clone has no {MEMBER}")
            return False
        shutil.copy2(found[0], dest)
    return True


def acquire(dest: Path) -> None:
    if dest.exists():
        print(f"  reusing        {dest.relative_to(ROOT)}")
        return

    print("Fetching the raw fingerprint archive")
    if _via_tarball(dest) or _via_git(dest):
        size_mb = dest.stat().st_size / 1e6
        print(f"  wrote          {dest.relative_to(ROOT)}  ({size_mb:.1f} MB)")
        return

    sys.exit(
        "\nCould not download the fingerprint archive.\n"
        "Both the tarball URL and the git clone failed — most likely no network\n"
        "access. Download x_real.npz manually from\n"
        f"  {REPO}\n"
        f"and place it at {dest}, then re-run this script.\n"
    )


# ---------------------------------------------------------------------------
# building the arrays
# ---------------------------------------------------------------------------
def normalise(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. Kept separate so the contrast filter is unambiguous."""
    lo, hi = image.min(), image.max()
    return (image - lo) / (hi - lo + 1e-9)


def build(raw_npz: Path) -> dict[str, np.ndarray]:
    arr = np.load(raw_npz)["data"][..., 0].astype(float)
    print(f"\nBuilding arrays from {len(arr)} scans of shape {arr.shape[1:]}")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(arr), POOL, replace=False)

    kept, skipped = [], 0
    for i in idx:
        image = normalise(arr[i])
        if image.std() < MIN_STD:
            skipped += 1
            continue
        kept.append(image - image.mean())          # zero-mean, as EMD expects
        if len(kept) >= WANTED:
            break

    if len(kept) < WANTED:
        sys.exit(f"only {len(kept)} prints passed the contrast filter, need {WANTED}")

    print(f"  kept {len(kept)} prints, skipped {skipped} low-contrast "
          f"(std < {MIN_STD})")

    first = normalise(arr[0])
    return {
        "fingerprint.npy": first - first.mean(),
        "fp_batch.npy": np.array(kept[:SMALL]),
        "fp_batch100.npy": np.array(kept),
    }


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--force", action="store_true",
                   help="rebuild even if the .npy files already exist")
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the downloaded x_real.npz (~35 MB)")
    args = p.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)

    if not args.force and all((DATA / name).exists() for name in OUTPUTS):
        print("All arrays already present — nothing to do. Use --force to rebuild.")
        return 0

    raw = DATA / "x_real.npz"
    acquire(raw)

    for name, array in build(raw).items():
        path = DATA / name
        np.save(path, array)
        print(f"  wrote  {path.relative_to(ROOT)}  shape {array.shape}  "
              f"({path.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_raw:
        raw.unlink(missing_ok=True)
        print(f"\n  removed {raw.name} (pass --keep-raw to keep it)")

    print("\nDone. Now try:")
    print("  python benchmark.py --data data/fp_batch.npy --limit 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
