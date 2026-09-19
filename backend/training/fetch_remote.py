"""
Fetch GISLR sequences by HTTP RANGE out of the remote zip. Offline; does not ship.

This supersedes both earlier approaches:

  per-file Kaggle API  -> tripped the rate limiter (429, and 404s on files that
                          demonstrably existed). Unusable at scale.
  full archive download -> 40 GB on disk, needing ~60 GB free.

The competition archive lives on GCS, which honours `Accept-Ranges: bytes`. So
we read the ZIP64 central directory from the tail (1.1 s), then range-request
only the members we want. Bytes transferred is proportional to the SUBSET, not
the archive, and nothing large ever touches disk.

    python -m backend.training.fetch_remote --signs 250 --per-sign 60
    python -m backend.training.fetch_remote --all --workers 24
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features                        # noqa: E402
from backend.app.landmarks import (                             # noqa: E402
    N_HAND, N_POSE_UPPER, pose_upper_indices,
)

DATA = config.ROOT / "data" / "gislr"
FEAT = DATA / "features"
COMP = "asl-signs"
MIN_FREE_GB = 1.0
EXPECTED_COLS = {"frame", "type", "landmark_index", "x", "y"}

_local = threading.local()
_URL: str | None = None


def free_gb() -> float:
    return shutil.disk_usage(DATA if DATA.exists() else DATA.parent).free / 1e9


def signed_url() -> str:
    """Signed GCS URL for the competition archive (valid ~68 h)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import (
        ApiDownloadDataFilesRequest,
    )
    a = KaggleApi(); a.authenticate()
    req = ApiDownloadDataFilesRequest(); req.competition_name = COMP
    with a.build_kaggle_client() as kc:
        return kc.competitions.competition_api_client.download_data_files(req).url


def zip_handle():
    """One RemoteZip per thread -- it holds connection + directory state."""
    z = getattr(_local, "zip", None)
    if z is None:
        from remotezip import RemoteZip
        z = RemoteZip(_URL)
        _local.zip = z
    return z


def ensure_index() -> pd.DataFrame:
    p = DATA / "train.csv"
    if not p.exists():
        from remotezip import RemoteZip
        DATA.mkdir(parents=True, exist_ok=True)
        with RemoteZip(_URL) as z:
            p.write_bytes(z.read("train.csv"))
    return pd.read_csv(p)


def choose_subset(df, n_signs, per_sign, seed=0):
    keep = df.sign.value_counts().head(n_signs).index if n_signs else df.sign.unique()
    sub = df[df.sign.isin(keep)]
    if per_sign:
        sub = (sub.sample(frac=1.0, random_state=seed)
                  .groupby("sign", sort=False).head(per_sign))
    return sub.reset_index(drop=True)


def to_features(raw: bytes, pose_idx: list[int]) -> np.ndarray:
    df = pd.read_parquet(io.BytesIO(raw),
                         columns=["frame", "type", "landmark_index", "x", "y"])
    if EXPECTED_COLS - set(df.columns):
        raise ValueError(f"missing columns {EXPECTED_COLS - set(df.columns)}")
    want = set(pose_idx)
    groups = list(df.groupby("frame", sort=True))
    out = np.zeros((len(groups), 53, 3), dtype=np.float32)
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


def fetch_one(member: str, seq: int, pose_idx, retries=3):
    dest = FEAT / f"{seq}.npy"
    if dest.exists():
        return "cached", 0
    if free_gb() < MIN_FREE_GB:
        return "abort", 0
    for attempt in range(retries):
        try:
            raw = zip_handle().read(member)
            np.save(dest, to_features(raw, pose_idx))
            return "ok", len(raw)
        except Exception as e:                                   # noqa: BLE001
            _local.zip = None                                    # force reconnect
            if attempt == retries - 1:
                return f"error:{type(e).__name__}:{str(e)[:50]}", 0
            time.sleep(1.5 * (attempt + 1))
    return "error", 0


def main():
    global _URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", type=int, default=250)
    ap.add_argument("--per-sign", type=int, default=60)
    ap.add_argument("--all", action="store_true", help="every sequence")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    print("resolving signed archive URL ...")
    _URL = signed_url()

    df = ensure_index()
    print(f"index: {len(df):,} sequences, {df.sign.nunique()} signs, "
          f"{df.participant_id.nunique()} signers")
    sub = df if args.all else choose_subset(df, args.signs, args.per_sign)
    if args.limit:
        sub = sub.head(args.limit)

    from remotezip import RemoteZip
    t0 = time.perf_counter()
    with RemoteZip(_URL) as z:
        members = {int(Path(n).stem): n for n in z.namelist() if n.endswith(".parquet")}
    print(f"central directory: {len(members):,} members in {time.perf_counter()-t0:.1f}s")

    have = {int(p.stem) for p in FEAT.glob("*.npy")}
    jobs = [(members[int(r.sequence_id)], int(r.sequence_id))
            for r in sub.itertuples() if int(r.sequence_id) in members
            and int(r.sequence_id) not in have]
    print(f"target {len(sub):,} sequences over {sub.sign.nunique()} signs; "
          f"{len(have):,} already local; {len(jobs):,} to fetch")
    print(f"est. transfer {len(jobs)*0.425/1000:.2f} GB   "
          f"est. features {len(jobs)*20/1000:.0f} MB   free {free_gb():.1f} GB\n")

    pose_idx = pose_upper_indices()
    counts, nbytes, errs = {}, 0, []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_one, m, s, pose_idx) for m, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            st, nb = f.result()
            key = st.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            nbytes += nb
            if key == "error" and len(errs) < 5:
                errs.append(st)
            if i % 250 == 0 or i == len(futs):
                el = time.perf_counter() - t0
                eta = (len(futs) - i) / max(i / el, 1e-9) / 60
                print(f"  {i:,}/{len(futs):,}  {counts}  {i/el:5.1f}/s  "
                      f"{nbytes/1e9:.2f} GB  eta {eta:5.1f} min  free {free_gb():.1f} GB")

    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min: {counts}")
    for e in errs:
        print(f"  sample error: {e}")

    got = {int(p.stem) for p in FEAT.glob("*.npy")}
    idx = df[df.sequence_id.isin(got)].reset_index(drop=True)
    idx.to_parquet(DATA / "subset_index.parquet")
    mb = sum(p.stat().st_size for p in FEAT.glob("*.npy")) / 1e6
    print(f"features: {len(got):,} files, {mb:.0f} MB")
    print(f"index: {len(idx):,} rows, {idx.sign.nunique()} signs, "
          f"{idx.participant_id.nunique()} signers")


if __name__ == "__main__":
    main()
