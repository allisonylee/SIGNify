"""
Fetch INCLUDE metadata from Hugging Face, then stream Zenodo zips one at a
time: extract INCLUDE-50 videos -> landmarks -> delete video and zip.

    python -m backend.trainingISL.download_include --calibrate 20
    python -m backend.trainingISL.download_include
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from backend.trainingISL.extract_video import extract_clip, rest_window_from_padding
from backend.trainingISL.paths import (
    DATA, FEAT, HF_BASE, INDEX, LABELS, MANIFEST, META, REST_SIGN, TMP,
    ZENODO_API, abort_if_low_disk, ensure_dirs, free_gb,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.landmarks import HolisticExtractor  # noqa: E402


def _urlretrieve(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def fetch_metadata() -> pd.DataFrame:
    ensure_dirs()
    frames = []
    for split in ("train", "val", "test"):
        dest = META / f"{split}.parquet"
        if not dest.exists():
            print(f"fetch HF {split} ...", flush=True)
            _urlretrieve(
                f"{HF_BASE}/data/{split}-00000-of-00001.parquet", dest)
        df = pd.read_parquet(dest)
        df["split_official"] = split
        frames.append(df)
    all_ = pd.concat(frames, ignore_index=True)
    return all_


def gloss_from_path(path: str) -> str:
    folder = str(path).replace("\\", "/").split("/")[-2]
    return re.sub(r"^\d+\.\s*", "", folder).strip()


def mvi_id(path: str) -> int:
    m = re.search(r"MVI_(\d+)", str(path), re.I)
    return int(m.group(1)) if m else -1


def assign_participant_proxy(df: pd.DataFrame, n_bins=7) -> pd.DataFrame:
    """INCLUDE does not ship signer ids. Bin MVI_ numbers into n_bins proxies."""
    out = df.copy()
    ids = out["mvi"].where(out["mvi"] >= 0)
    # rank-based bins so empty ranges still split
    ranks = ids.rank(method="average")
    n_bins = min(n_bins, max(int(ids.nunique()), 1))
    try:
        out["participant_id"] = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        out["participant_id"] = 0
    out["participant_id"] = out["participant_id"].fillna(0).astype(int)
    return out


def build_index(meta: pd.DataFrame) -> pd.DataFrame:
    sub = meta[meta["include_50"] == True].copy()  # noqa: E712
    sub["sign"] = sub["video_path"].map(gloss_from_path)
    sub["video_path"] = sub["video_path"].astype(str).str.replace("\\", "/", regex=False)
    sub["category"] = sub["video_path"].str.split("/").str[0]
    sub["mvi"] = sub["video_path"].map(mvi_id)
    sub = assign_participant_proxy(sub)
    sub = sub.reset_index(drop=True)
    sub.insert(0, "sequence_id", np.arange(1, len(sub) + 1, dtype=np.int64))
    cols = ["sequence_id", "sign", "category", "video_path", "split_official",
            "participant_id", "mvi", "parent_label"]
    extra = [c for c in ("include_50",) if c in sub.columns]
    return sub[cols + extra]


def write_labels(index: pd.DataFrame):
    signs = sorted(index.sign.unique())
    labels = signs + [REST_SIGN]
    LABELS.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def zip_category(name: str) -> str:
    stem = name[:-4] if name.lower().endswith(".zip") else name
    return re.sub(r"_\d+of\d+$", "", stem)


def zenodo_files() -> list[dict]:
    cache = META / "zenodo_4010759.json"
    if not cache.exists():
        print("fetch zenodo file list ...", flush=True)
        with urllib.request.urlopen(ZENODO_API, timeout=60) as r:
            cache.write_bytes(r.read())
    rec = json.loads(cache.read_text(encoding="utf-8"))
    return rec["files"]


def needed_zips(index: pd.DataFrame, files: list[dict]) -> list[dict]:
    cats = set(index.category.str.replace(" ", "_", regex=False))
    out = []
    for f in files:
        key = f["key"]
        if not key.lower().endswith(".zip"):
            continue
        if zip_category(key).replace(" ", "_") in cats:
            out.append(f)
    return sorted(out, key=lambda x: x["key"])


def load_done() -> set[int]:
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ok"):
                done.add(int(row["sequence_id"]))
    return done


def append_manifest(row: dict):
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def index_zip_members(z: zipfile.ZipFile) -> dict[tuple[int, str], str]:
    """Key is (MVI number, parent folder lowercased). MVI ids collide across signs."""
    out = {}
    for n in z.namelist():
        m = re.search(r"MVI_(\d+)", n, re.I)
        if not m:
            continue
        parts = n.replace("\\", "/").split("/")
        folder = parts[-2].lower() if len(parts) >= 2 else ""
        out[(int(m.group(1)), folder)] = n
    return out


def process_zip(zinfo: dict, index: pd.DataFrame, extractor: HolisticExtractor,
                remaining: int | None) -> int:
    key = zinfo["key"]
    cat = zip_category(key)
    want = index[index.category.str.replace(" ", "_", regex=False) == cat.replace(" ", "_")]
    done = load_done()
    want = want[~want.sequence_id.isin(done)]
    if remaining is not None:
        want = want.head(remaining)
    if want.empty:
        print(f"  skip {key} (nothing pending)", flush=True)
        return 0

    abort_if_low_disk()
    zip_path = TMP / key
    if not zip_path.exists():
        url = zinfo["links"]["self"]
        print(f"  download {key} ({zinfo['size']/1e9:.2f} GB) ...", flush=True)
        t0 = time.perf_counter()
        _urlretrieve(url, zip_path)
        print(f"    done in {time.perf_counter()-t0:.0f}s  free={free_gb():.2f} GB",
              flush=True)

    n_ok = 0
    with zipfile.ZipFile(zip_path) as z:
        by_key = index_zip_members(z)
        print(f"    zip video members: {len(by_key)}", flush=True)
        for rec in want.itertuples(index=False):
            abort_if_low_disk()
            folder = str(rec.video_path).replace("\\", "/").split("/")[-2].lower()
            member = by_key.get((int(rec.mvi), folder))
            if member is None:
                continue
            ext = Path(member).suffix or ".mp4"
            vpath = TMP / f"clip_{int(rec.sequence_id)}{ext}"
            try:
                with z.open(member) as src, vpath.open("wb") as dst:
                    dst.write(src.read())
                info = extract_clip(
                    vpath, extractor,
                    timestamp_base_ms=int(rec.sequence_id) * 1_000_000)
                np.save(FEAT / f"{int(rec.sequence_id)}.npy", info["window"])
                rest = rest_window_from_padding(info["raw"])
                rest_id = None
                if rest is not None:
                    rest_id = 1_000_000 + int(rec.sequence_id)
                    np.save(FEAT / f"{rest_id}.npy", rest)
                append_manifest({
                    "sequence_id": int(rec.sequence_id),
                    "ok": True,
                    "zip": key,
                    "n_frames": info["n_frames"],
                    "duration_s": info["duration_s"],
                    "hands_detected_frac": info["hands_detected_frac"],
                    "motion_energy": info["motion_energy"],
                    "rest_id": rest_id,
                })
                n_ok += 1
                print(f"    {rec.sequence_id} {rec.sign!r}  "
                      f"{info['n_frames']}f  hands={info['hands_detected_frac']:.0%}  "
                      f"E={info['motion_energy']:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                append_manifest({"sequence_id": int(rec.sequence_id), "ok": False,
                                 "error": str(e), "zip": key})
                print(f"    FAIL {rec.sequence_id}: {e}", flush=True)
            finally:
                if vpath.exists():
                    vpath.unlink()
            if remaining is not None and n_ok >= remaining:
                break
    try:
        zip_path.unlink()
    except OSError:
        pass
    return n_ok


def merge_rest_into_index(index: pd.DataFrame) -> pd.DataFrame:
    extra = []
    if not MANIFEST.exists():
        return index
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rid = row.get("rest_id")
        if row.get("ok") and rid:
            src = index[index.sequence_id == row["sequence_id"]]
            if src.empty:
                continue
            r = src.iloc[0].to_dict()
            r["sequence_id"] = int(rid)
            r["sign"] = REST_SIGN
            extra.append(r)
    if not extra:
        return index
    return pd.concat([index, pd.DataFrame(extra)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, default=0,
                    help="stop after N successful extracts (20 for the sanity pass)")
    ap.add_argument("--max-zips", type=int, default=0)
    args = ap.parse_args()

    ensure_dirs()
    print(f"free disk {abort_if_low_disk():.2f} GB", flush=True)

    meta = fetch_metadata()
    index = build_index(meta)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    labels = write_labels(index)
    print(f"INCLUDE-50 rows {len(index)}  signs {len(labels)-1} + rest", flush=True)
    print(f"proxy signers {sorted(index.participant_id.unique())}", flush=True)
    print(f"categories {sorted(index.category.unique())}", flush=True)

    files = needed_zips(index, zenodo_files())
    if args.max_zips:
        files = files[:args.max_zips]
    print(f"zenodo zips to scan: {len(files)}", flush=True)

    remaining = args.calibrate if args.calibrate else None
    extractor = HolisticExtractor()
    try:
        for zinfo in files:
            if remaining is not None and remaining <= 0:
                break
            n = process_zip(zinfo, index, extractor, remaining)
            if remaining is not None:
                remaining -= n
    finally:
        extractor.close()

    index2 = merge_rest_into_index(index)
    index2.to_parquet(INDEX)
    npy = list(FEAT.glob("*.npy"))
    print(f"\nfeatures on disk: {len(npy)}  index {INDEX}", flush=True)
    print(f"labels {LABELS}", flush=True)


if __name__ == "__main__":
    main()
