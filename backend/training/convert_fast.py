"""
Vectorised GISLR parquet -> features. Offline; does not ship.

The original per-frame loop managed 6 sequences/s, projecting to >4 hours for
the archive. Every GISLR parquet has an identical, contiguous layout:

    543 rows per frame, always ordered
    face 0..467 | left_hand 0..20 | pose 0..32 | right_hand 0..20

so the whole file reshapes to (n_frames, 543, 2) in one step and normalisation
vectorises across frames. The layout is VERIFIED per file, with a fallback to
the slow path if a file ever disagrees.

    python -m backend.training.convert_fast --archive ~/Downloads/asl-signs.zip
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features
from backend.app.features import MIN_SHOULDER_WIDTH
from backend.app.landmarks import (
    N_HAND, N_LANDMARKS, N_POSE_UPPER, ROW_LEFT_SHOULDER, ROW_RIGHT_SHOULDER,
    SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND, pose_upper_indices,
)

DATA = config.ROOT / "data" / "gislr"
FEAT = DATA / "features"

ROWS_PER_FRAME = 543
OFF_FACE, OFF_LH, OFF_POSE, OFF_RH = 0, 468, 489, 522   # block start offsets


def _layout_ok(df: pd.DataFrame, nf: int) -> bool:
    """Confirm the assumed block layout on this file's first frame."""
    if len(df) != nf * ROWS_PER_FRAME:
        return False
    head = df.iloc[:ROWS_PER_FRAME]
    t = head.type.to_numpy()
    li = head.landmark_index.to_numpy()
    return (
        (t[OFF_FACE:OFF_LH] == "face").all()
        and (t[OFF_LH:OFF_POSE] == "left_hand").all()
        and (t[OFF_POSE:OFF_RH] == "pose").all()
        and (t[OFF_RH:] == "right_hand").all()
        and np.array_equal(li[OFF_LH:OFF_POSE], np.arange(N_HAND))
        and np.array_equal(li[OFF_POSE:OFF_RH], np.arange(33))
        and np.array_equal(li[OFF_RH:], np.arange(N_HAND))
    )


def assemble_vec(xy: np.ndarray, pose_idx: list[int]) -> np.ndarray:
    """
    (nf, 543, 2) -> (nf, 53, 3). Mirrors app/features.assemble exactly:
      * a hand is "detected" only if NOT entirely NaN (per frame)
      * pose is nan_to_num'd with a PER-LANDMARK mask
    """
    nf = xy.shape[0]
    out = np.zeros((nf, N_LANDMARKS, 3), dtype=np.float32)

    for off, sl in ((OFF_LH, SLICE_LEFT_HAND), (OFF_RH, SLICE_RIGHT_HAND)):
        block = xy[:, off:off + N_HAND]                     # (nf, 21, 2)
        present = ~np.isnan(block).all(axis=(1, 2))         # (nf,)
        out[present, sl, :2] = block[present]
        out[present, sl, 2] = 1.0

    pose = xy[:, OFF_POSE:OFF_POSE + 33][:, pose_idx]       # (nf, 11, 2)
    out[:, SLICE_POSE, :2] = np.nan_to_num(pose)
    out[:, SLICE_POSE, 2] = (~np.isnan(pose).any(axis=2)).astype(np.float32)
    return out


def normalise_vec(w: np.ndarray) -> np.ndarray:
    """Vectorised app/features.normalise, frame by frame, same guards."""
    out = w.copy()
    lsh, rsh = out[:, ROW_LEFT_SHOULDER], out[:, ROW_RIGHT_SHOULDER]
    width = np.abs(lsh[:, 0] - rsh[:, 0])
    good = (lsh[:, 2] >= 0.5) & (rsh[:, 2] >= 0.5) & (width >= MIN_SHOULDER_WIDTH)
    if not good.any():
        return out
    origin = (lsh[good, :2] + rsh[good, :2]) / 2.0          # (g, 2)
    sub = out[good]                                          # (g, 53, 3)
    det = sub[:, :, 2] > 0.5
    scaled = (sub[:, :, :2] - origin[:, None, :]) / width[good][:, None, None]
    sub[:, :, :2] = np.where(det[:, :, None], scaled, sub[:, :, :2])
    out[good] = sub
    return out


def to_features_fast(raw: bytes, pose_idx: list[int]):
    df = pd.read_parquet(io.BytesIO(raw),
                         columns=["frame", "type", "landmark_index", "x", "y"])
    nf = df.frame.nunique()
    if not (df.frame.is_monotonic_increasing and _layout_ok(df, nf)):
        return None                                          # caller falls back
    xy = df[["x", "y"]].to_numpy(np.float32).reshape(nf, ROWS_PER_FRAME, 2)
    w = normalise_vec(assemble_vec(xy, pose_idx))
    t = np.arange(nf, dtype=np.float64) / max(nf - 1, 1)
    return features.resample_by_time(t, w, config.WINDOW_FRAMES, 1.0)


def to_features_slow(raw: bytes, pose_idx: list[int]) -> np.ndarray:
    """Original per-frame path. Kept as the fallback and as ground truth."""
    df = pd.read_parquet(io.BytesIO(raw),
                         columns=["frame", "type", "landmark_index", "x", "y"])
    want = set(pose_idx)
    groups = list(df.groupby("frame", sort=True))
    out = np.zeros((len(groups), N_LANDMARKS, 3), dtype=np.float32)
    for i, (_, g) in enumerate(groups):
        hands = {}
        for side in ("left", "right"):
            h = g[g.type == f"{side}_hand"].sort_values("landmark_index")
            arr = h[["x", "y"]].to_numpy(np.float32)
            hands[side] = arr if (len(arr) == N_HAND and not np.isnan(arr).all()) else None
        pz = (g[(g.type == "pose") & (g.landmark_index.isin(want))]
              .set_index("landmark_index").reindex(pose_idx))
        pose = pz[["x", "y"]].to_numpy(np.float32)
        if pose.shape != (N_POSE_UPPER, 2):
            pose = np.full((N_POSE_UPPER, 2), np.nan, np.float32)
        out[i] = features.normalise(features.assemble(hands, pose))
    t = np.arange(len(groups), dtype=np.float64) / max(len(groups) - 1, 1)
    return features.resample_by_time(t, out, config.WINDOW_FRAMES, 1.0)


_Z = None


def _worker(args):
    global _Z
    zpath, member, seq, pose_idx = args
    try:
        if _Z is None:
            _Z = zipfile.ZipFile(zpath)
        raw = _Z.read(member)
        arr = to_features_fast(raw, pose_idx)
        used = "fast"
        if arr is None:
            arr = to_features_slow(raw, pose_idx)
            used = "slow"
        np.save(FEAT / f"{seq}.npy", arr)
        return seq, used, ""
    except Exception as e:                                       # noqa: BLE001
        return seq, "error", f"{type(e).__name__}: {str(e)[:70]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(Path.home() / "Downloads" / "asl-signs.zip"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    Z = Path(args.archive).expanduser()
    if not Z.exists():
        sys.exit(f"archive not found: {Z}")
    FEAT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(Z) as z:
        names = [n for n in z.namelist() if n.endswith(".parquet")]
        if "train.csv" in z.namelist():
            (DATA / "train.csv").write_bytes(z.read("train.csv"))
    have = {int(p.stem) for p in FEAT.glob("*.npy")}
    pose_idx = pose_upper_indices()
    jobs = [(str(Z), n, int(Path(n).stem), pose_idx)
            for n in names if int(Path(n).stem) not in have]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"archive {Z}  ({Z.stat().st_size/1e9:.1f} GB, {len(names):,} parquets)")
    print(f"{len(have):,} already converted; {len(jobs):,} to do on {args.workers} workers\n")

    counts, errs = {}, []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            seq, used, msg = f.result()
            counts[used] = counts.get(used, 0) + 1
            if used == "error" and len(errs) < 5:
                errs.append(f"{seq}: {msg}")
            if i % 5000 == 0 or i == len(futs):
                el = time.perf_counter() - t0
                print(f"  {i:,}/{len(futs):,}  {counts}  {i/el:6.0f}/s  "
                      f"eta {(len(futs)-i)/max(i/el,1e-9)/60:5.1f} min")
    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min: {counts}")
    for e in errs:
        print(f"  error: {e}")

    got = {int(p.stem) for p in FEAT.glob("*.npy")}
    idx = pd.read_csv(DATA / "train.csv")
    idx = idx[idx.sequence_id.isin(got)].reset_index(drop=True)
    idx.to_parquet(DATA / "subset_index.parquet")
    mb = sum(p.stat().st_size for p in FEAT.glob("*.npy")) / 1e6
    print(f"features {len(got):,} files, {mb:.0f} MB")
    print(f"index {len(idx):,} rows, {idx.sign.nunique()} signs, "
          f"{idx.participant_id.nunique()} signers")


if __name__ == "__main__":
    main()
