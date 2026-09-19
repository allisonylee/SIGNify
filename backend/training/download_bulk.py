"""
Bulk GISLR: one archive download, then local conversion. Offline; does not ship.

Why this exists: fetching 94,477 files one at a time trips Kaggle's rate
limiter, which reports itself inconsistently (we saw 429 AND 404-on-files-that-
exist). One archive request avoids the problem entirely, and conversion then
runs locally with no network in the loop.

Disk: the archive is ~51 GB. Parquets are read STRAIGHT OUT OF THE ZIP into
memory -- never extracted -- so peak usage is the zip plus ~2 GB of features.
Pass --delete-archive to reclaim the 51 GB when conversion finishes.

    python -m backend.training.download_bulk --download
    python -m backend.training.download_bulk --convert --workers 8
    python -m backend.training.download_bulk --convert --delete-archive
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features                       # noqa: E402
from backend.app.landmarks import (                            # noqa: E402
    N_HAND, N_POSE_UPPER, pose_upper_indices,
)

DATA = config.ROOT / "data" / "gislr"
FEAT = DATA / "features"
ARCHIVE = DATA / "asl-signs.zip"
COMP = "asl-signs"
MIN_FREE_GB = 3.0

EXPECTED_COLS = {"frame", "type", "landmark_index", "x", "y"}


def free_gb() -> float:
    return shutil.disk_usage(DATA if DATA.exists() else DATA.parent).free / 1e9


def do_download():
    need = 55.0
    print(f"free disk: {free_gb():.1f} GB   (archive ~51 GB, want >= {need} GB)")
    if free_gb() < need:
        sys.exit(f"ABORT: need ~{need} GB free, have {free_gb():.1f} GB")
    DATA.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate()
    print(f"downloading the full {COMP} archive -> {DATA}")
    t0 = time.perf_counter()
    a.competition_download_files(COMP, path=str(DATA), quiet=False)
    el = time.perf_counter() - t0
    zips = sorted(DATA.glob("*.zip"), key=lambda p: -p.stat().st_size)
    if not zips:
        sys.exit("no archive produced")
    if zips[0] != ARCHIVE:
        zips[0].rename(ARCHIVE)
    gb = ARCHIVE.stat().st_size / 1e9
    print(f"done: {gb:.2f} GB in {el/60:.1f} min ({gb*1000/el:.0f} MB/s)")


def _convert_member(args):
    """Runs in a worker process. Reads one parquet from the zip, never to disk."""
    zpath, member, seq, pose_idx = args
    try:
        with zipfile.ZipFile(zpath) as z:
            raw = z.read(member)
        df = pd.read_parquet(io.BytesIO(raw),
                             columns=["frame", "type", "landmark_index", "x", "y"])
        if EXPECTED_COLS - set(df.columns):
            return seq, "error", f"missing columns {EXPECTED_COLS - set(df.columns)}"

        frames = np.sort(df.frame.unique())
        want = set(pose_idx)
        out = np.zeros((len(frames), 53, 3), dtype=np.float32)
        for i, (_, g) in enumerate(df.groupby("frame", sort=True)):
            hands = {}
            for side in ("left", "right"):
                h = g[g.type == f"{side}_hand"].sort_values("landmark_index")
                arr = h[["x", "y"]].to_numpy(np.float32)
                hands[side] = arr if (len(arr) == N_HAND
                                      and not np.isnan(arr).all()) else None
            pz = (g[(g.type == "pose") & (g.landmark_index.isin(want))]
                  .set_index("landmark_index").reindex(pose_idx))
            pose = pz[["x", "y"]].to_numpy(np.float32)
            if pose.shape != (N_POSE_UPPER, 2):
                pose = np.full((N_POSE_UPPER, 2), np.nan, np.float32)
            out[i] = features.normalise(features.assemble(hands, pose))

        t = np.arange(len(frames), dtype=np.float64) / max(len(frames) - 1, 1)
        w = features.resample_by_time(t, out, config.WINDOW_FRAMES, 1.0)
        np.save(FEAT / f"{seq}.npy", w)
        return seq, "ok", ""
    except Exception as e:                                       # noqa: BLE001
        return seq, "error", f"{type(e).__name__}: {str(e)[:70]}"


def do_convert(workers, limit, delete_archive):
    if not ARCHIVE.exists():
        sys.exit(f"archive not found: {ARCHIVE}  (run --download first)")
    FEAT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE) as z:
        names = [n for n in z.namelist() if n.endswith(".parquet")]
        if "train.csv" in z.namelist():
            (DATA / "train.csv").write_bytes(z.read("train.csv"))
    print(f"archive holds {len(names):,} parquet members")

    have = {int(p.stem) for p in FEAT.glob("*.npy")}
    jobs = []
    pose_idx = pose_upper_indices()
    for n in names:
        try:
            seq = int(Path(n).stem)
        except ValueError:
            continue
        if seq not in have:
            jobs.append((str(ARCHIVE), n, seq, pose_idx))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(have):,} already converted; {len(jobs):,} to do on {workers} workers")
    if not jobs:
        print("nothing to convert")
    else:
        t0 = time.perf_counter()
        counts = {"ok": 0, "error": 0}
        errs = []
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(_convert_member, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                seq, st, msg = f.result()
                counts[st] += 1
                if st == "error" and len(errs) < 5:
                    errs.append(f"{seq}: {msg}")
                if i % 2000 == 0 or i == len(futs):
                    el = time.perf_counter() - t0
                    eta = (len(futs) - i) / max(i / el, 1e-9) / 60
                    print(f"  {i:,}/{len(futs):,}  {counts}  {i/el:.0f}/s  "
                          f"eta {eta:.1f} min  free {free_gb():.1f} GB")
        print(f"\nconverted in {(time.perf_counter()-t0)/60:.1f} min: {counts}")
        for e in errs:
            print(f"  sample error: {e}")

    n = len(list(FEAT.glob("*.npy")))
    mb = sum(p.stat().st_size for p in FEAT.glob("*.npy")) / 1e6
    print(f"features: {n:,} files, {mb:.0f} MB")

    # Build the index the trainer reads, restricted to what actually converted.
    if (DATA / "train.csv").exists():
        idx = pd.read_csv(DATA / "train.csv")
        got = {int(p.stem) for p in FEAT.glob("*.npy")}
        sub = idx[idx.sequence_id.isin(got)].reset_index(drop=True)
        sub.to_parquet(DATA / "subset_index.parquet")
        print(f"index: {len(sub):,} rows, {sub.sign.nunique()} signs, "
              f"{sub.participant_id.nunique()} signers")

    if delete_archive and ARCHIVE.exists():
        gb = ARCHIVE.stat().st_size / 1e9
        ARCHIVE.unlink()
        print(f"deleted archive, reclaimed {gb:.1f} GB -> free {free_gb():.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delete-archive", action="store_true")
    ap.add_argument("--archive", default=None,
                    help="path to an already-downloaded asl-signs.zip "
                         "(e.g. ~/Downloads/asl-signs.zip). Read-only; never moved.")
    args = ap.parse_args()
    if args.archive:
        global ARCHIVE
        ARCHIVE = Path(args.archive).expanduser()
        if not ARCHIVE.exists():
            sys.exit(f"archive not found: {ARCHIVE}")
        if args.delete_archive:
            sys.exit("refusing --delete-archive on a user-supplied path")
    if not (args.download or args.convert):
        ap.error("pass --download and/or --convert")
    if args.download:
        do_download()
    if args.convert:
        do_convert(args.workers, args.limit, args.delete_archive)


if __name__ == "__main__":
    main()
