"""
GISLR download -> features -> delete. Offline; does not ship.

The full asl-signs corpus is tens of GB and this machine has single-digit GB
free, so we NEVER materialise it. Each sequence is its own parquet, so we:

    download one parquet -> convert to (32,53,3) features -> write .npy
    -> DELETE the parquet

Features are ~20 KB per sequence against ~0.5-3 MB of parquet, so a 6,000
sequence subset costs ~120 MB on disk with a peak transient of a few MB.

Safety rails, because the disk is nearly full:
  * hard abort if free space drops below MIN_FREE_GB
  * resume manifest, so a crash costs nothing
  * balanced subset selection, capped per class

Usage:
    python -m backend.training.download --signs 100 --per-sign 60
    python -m backend.training.download --all          # every sequence
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features           # noqa: E402
from backend.app.landmarks import (                # noqa: E402
    N_HAND, N_POSE_UPPER, POSE_UPPER_NAMES, pose_upper_indices,
)

DATA = config.ROOT / "data" / "gislr"
FEAT = DATA / "features"
MIN_FREE_GB = 1.5
COMP = "asl-signs"

# GISLR parquet schema, asserted on the first file so a change fails loudly.
EXPECTED_COLS = {"frame", "type", "landmark_index", "x", "y"}


def free_gb(path=DATA) -> float:
    return shutil.disk_usage(path.parent if not path.exists() else path).free / 1e9


_thread_local = threading.local()


def api():
    """
    One KaggleApi per thread. The client is not documented as thread-safe and
    sharing a single instance across workers correlated with bogus 404s.
    """
    a = getattr(_thread_local, "api", None)
    if a is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        a = KaggleApi(); a.authenticate()
        _thread_local.api = a
    return a


class RateLimiter:
    """
    Token bucket. Kaggle throttles bursts and reports it inconsistently -- we
    saw both 429 and a *404 masquerading as a missing file* for paths that had
    downloaded successfully minutes earlier. So pace deliberately.
    """

    def __init__(self, per_sec: float):
        self.interval = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_at)
            self.next_at = t + self.interval
        if t > now:
            time.sleep(t - now)


class Throttled(Exception):
    pass


def load_index() -> pd.DataFrame:
    p = DATA / "train.csv"
    if not p.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        api().competition_download_file(COMP, "train.csv", path=str(DATA))
        for z in DATA.glob("train.csv.zip"):
            import zipfile
            zipfile.ZipFile(z).extractall(DATA); z.unlink()
    return pd.read_csv(p)


def choose_subset(df, n_signs, per_sign, seed=0):
    """Most-common signs first, capped per class, sampled deterministically."""
    keep = df.sign.value_counts().head(n_signs).index if n_signs else df.sign.unique()
    sub = df[df.sign.isin(keep)]
    if per_sign:
        # Shuffle once, then take the first `per_sign` of each group. Avoids
        # groupby.apply, which folds the grouping column into the index.
        sub = (sub.sample(frac=1.0, random_state=seed)
                  .groupby("sign", sort=False)
                  .head(per_sign))
    return sub.reset_index(drop=True)


def parquet_to_features(path: Path, pose_idx: list[int]) -> np.ndarray:
    """
    One GISLR parquet -> (32, 53, 3), normalised, resampled on frame index.

    GISLR sequences carry a frame number but no wall-clock timestamp, so we
    treat frames as evenly spaced -- correct here, because these were recorded
    at a fixed rate. Live input uses real timestamps (app/features.py).
    """
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y"])
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    frames = np.sort(df.frame.unique())
    want_pose = set(pose_idx)
    out = np.zeros((len(frames), 53, 3), dtype=np.float32)

    by_frame = {f: g for f, g in df.groupby("frame")}
    for i, f in enumerate(frames):
        g = by_frame[f]
        hands = {}
        for side in ("left", "right"):
            h = g[g.type == f"{side}_hand"].sort_values("landmark_index")
            arr = h[["x", "y"]].to_numpy(np.float32)
            hands[side] = arr if len(arr) == N_HAND and not np.isnan(arr).all() else None
        pz = g[(g.type == "pose") & (g.landmark_index.isin(want_pose))]
        pz = pz.set_index("landmark_index").reindex(pose_idx)
        pose = pz[["x", "y"]].to_numpy(np.float32)
        if pose.shape != (N_POSE_UPPER, 2):
            pose = np.full((N_POSE_UPPER, 2), np.nan, np.float32)
        out[i] = features.normalise(features.assemble(hands, pose))

    t = np.arange(len(frames), dtype=np.float64) / max(len(frames) - 1, 1)
    return features.resample_by_time(t, out, config.WINDOW_FRAMES, 1.0)


def fetch_one(row, pose_idx, tmp: Path, limiter: RateLimiter,
              max_retries=5) -> tuple[str, str]:
    seq = int(row.sequence_id)
    dest = FEAT / f"{seq}.npy"
    if dest.exists():
        return "cached", ""
    if free_gb() < MIN_FREE_GB:
        return "abort", f"free disk {free_gb():.2f} GB < {MIN_FREE_GB}"

    name = Path(row.path).name
    got = tmp / name
    for attempt in range(max_retries):
        try:
            limiter.wait()
            api().competition_download_file(COMP, row.path, path=str(tmp), force=True)
            for z in tmp.glob(f"{name}.zip"):
                import zipfile
                zipfile.ZipFile(z).extractall(tmp); z.unlink()
            if not got.exists():
                raise Throttled("no file produced")
            arr = parquet_to_features(got, pose_idx)
            np.save(dest, arr)
            return "ok", ""
        except Exception as e:                                   # noqa: BLE001
            msg = str(e)
            transient = ("429" in msg or "404" in msg or "500" in msg
                         or "Throttled" in type(e).__name__ or "no file" in msg)
            if not transient or attempt == max_retries - 1:
                return "error", f"{type(e).__name__}: {msg[:80]}"
            # Exponential backoff with jitter. 404 is included deliberately:
            # Kaggle returns it for throttled requests on files that exist.
            time.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
        finally:
            if got.exists():
                got.unlink()                                     # DELETE IMMEDIATELY
    return "error", "exhausted retries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", type=int, default=100, help="0 = all signs")
    ap.add_argument("--per-sign", type=int, default=60, help="0 = uncapped")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rate", type=float, default=3.0,
                    help="max downloads per second across all workers")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (smoke test)")
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    tmp = DATA / "_tmp"; tmp.mkdir(exist_ok=True)

    df = load_index()
    print(f"index: {len(df):,} sequences, {df.sign.nunique()} signs, "
          f"{df.participant_id.nunique()} signers")
    sub = choose_subset(df, args.signs, args.per_sign)
    if args.limit:
        sub = sub.head(args.limit)
    print(f"subset: {len(sub):,} sequences over {sub.sign.nunique()} signs")
    print(f"est. features on disk: {len(sub) * 32 * 53 * 3 * 4 / 1e6:.0f} MB")
    print(f"free disk now: {free_gb():.2f} GB (abort below {MIN_FREE_GB})")
    print(f"pacing: {args.rate}/s across {args.workers} workers, exponential backoff on 429/404\n")

    sub.to_parquet(DATA / "subset_index.parquet")
    limiter = RateLimiter(args.rate)
    pose_idx = pose_upper_indices()
    counts = {"ok": 0, "cached": 0, "error": 0, "abort": 0}
    t0 = time.perf_counter()
    errs = []

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_one, r, pose_idx, tmp, limiter): r
                for r in sub.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            st, msg = f.result()
            counts[st] = counts.get(st, 0) + 1
            if st == "error" and len(errs) < 5:
                errs.append(msg)
            if st == "abort":
                print(f"\n!! ABORT: {msg}"); break
            if i % 50 == 0 or i == len(futs):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(futs)}  {counts}  {i/el:.1f}/s  "
                      f"free {free_gb():.2f} GB")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\ndone in {time.perf_counter()-t0:.0f}s: {counts}")
    for e in errs:
        print(f"  sample error: {e}")
    n = len(list(FEAT.glob('*.npy')))
    print(f"features on disk: {n:,} files, "
          f"{sum(p.stat().st_size for p in FEAT.glob('*.npy'))/1e6:.0f} MB")


if __name__ == "__main__":
    main()
