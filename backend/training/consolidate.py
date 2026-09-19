"""
Pack 94k per-sequence .npy files into one array. Offline; does not ship.

Loading 94,477 small files at the start of every training run is dominated by
per-file overhead -- it was still going after 8 minutes. One contiguous
array plus an id->row map reads in seconds and can be memory-mapped, so
repeated runs and multiple workers cost nothing.

    python -m backend.training.consolidate
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config                                  # noqa: E402

DATA = config.ROOT / "data" / "gislr"
FEAT = DATA / "features"
PACK = DATA / "features_all.npy"
IDS = DATA / "features_ids.npy"


def build(force=False):
    files = sorted(FEAT.glob("*.npy"), key=lambda p: int(p.stem))
    if not files:
        sys.exit("no features found")
    if PACK.exists() and IDS.exists() and not force:
        ids = np.load(IDS)
        if len(ids) == len(files):
            print(f"pack already current: {len(ids):,} sequences")
            return
        print(f"pack is stale ({len(ids):,} vs {len(files):,}); rebuilding")

    ids = np.array([int(p.stem) for p in files], dtype=np.int64)
    shape = np.load(files[0]).shape
    print(f"packing {len(files):,} sequences of {shape} -> {PACK}")
    out = np.lib.format.open_memmap(PACK, mode="w+",
                                    dtype=np.float32, shape=(len(files), *shape))
    t0 = time.perf_counter()
    for i, p in enumerate(files):
        out[i] = np.load(p)
        if (i + 1) % 20000 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:,}/{len(files):,}  {(i+1)/el:.0f}/s  "
                  f"eta {(len(files)-i-1)/max((i+1)/el,1e-9)/60:.1f} min", flush=True)
    out.flush(); del out
    np.save(IDS, ids)
    gb = PACK.stat().st_size / 1e9
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min -> {gb:.2f} GB")

    t0 = time.perf_counter()
    a = np.load(PACK, mmap_mode="r")
    print(f"memmap open: {(time.perf_counter()-t0)*1000:.0f} ms, shape {a.shape}")


if __name__ == "__main__":
    build(force="--force" in sys.argv)
