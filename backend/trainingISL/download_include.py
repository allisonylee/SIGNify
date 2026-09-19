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
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from backend.trainingISL.paths import (
    DATA, FEAT, HF_BASE, INDEX, LABELS, MANIFEST, META, REST_SIGN, TMP,
    ZENODO_API, abort_if_low_disk, ensure_dirs, free_gb,
)


def _urlretrieve(url: str, dest: Path, attempts=8):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except (OSError, urllib.error.URLError) as e:
            last = e
            wait = min(90, 5 * i)
            print(f"    download retry {i}/{attempts} after {e!r}  sleep {wait}s",
                  flush=True)
            time.sleep(wait)
    raise last


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


def load_failed() -> set[int]:
    failed = set()
    if not MANIFEST.exists():
        return failed
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok") is False:
            failed.add(int(row["sequence_id"]))
    return failed


def zips_fully_consumed() -> set[str]:
    """Skip zips that already finished. Keep the last successful zip so a
    mid-zip crash can resume without re-downloading earlier category parts.
    """
    keys: list[str] = []
    if not MANIFEST.exists():
        return set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok") and row.get("zip"):
            keys.append(row["zip"])
    if not keys:
        return set()
    return set(keys) - {keys[-1]}


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


def resolve_zip_path(key: str, zip_dirs: list[Path]) -> Path | None:
    for d in zip_dirs:
        p = d / key
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def extract_clip_subprocess(vpath: Path, sequence_id: int) -> dict:
    """Child process so a MediaPipe abort cannot kill the zip loop."""
    out = FEAT / f"{int(sequence_id)}.npy"
    rest_out = FEAT / f"{1_000_000 + int(sequence_id)}.npy"
    r = subprocess.run(
        [sys.executable, "-m", "backend.trainingISL.extract_one",
         str(vpath), str(out), str(int(sequence_id) * 1_000_000), str(rest_out)],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True, text=True, timeout=240,
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-400:].replace("\n", " ")
        raise RuntimeError(f"extract_one exit {r.returncode}: {tail}")
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError("extract_one produced no json")
    return json.loads(lines[-1])


def process_zip(zinfo: dict, index: pd.DataFrame,
                remaining: int | None, zip_dirs: list[Path], keep_zips: bool) -> int:
    key = zinfo["key"]
    cat = zip_category(key)
    want = index[index.category.str.replace(" ", "_", regex=False) == cat.replace(" ", "_")]
    skip = load_done() | load_failed()
    want = want[~want.sequence_id.isin(skip)]
    if remaining is not None:
        want = want.head(remaining)
    if want.empty:
        print(f"  skip {key} (nothing pending)", flush=True)
        return 0

    abort_if_low_disk()
    zip_path = resolve_zip_path(key, zip_dirs) or (TMP / key)
    downloaded = False
    if not zip_path.exists():
        url = zinfo["links"]["self"]
        print(f"  download {key} ({zinfo['size']/1e9:.2f} GB) ...", flush=True)
        t0 = time.perf_counter()
        _urlretrieve(url, zip_path)
        downloaded = True
        print(f"    done in {time.perf_counter()-t0:.0f}s  free={free_gb():.2f} GB",
              flush=True)
    else:
        print(f"  local {key} ({zip_path.stat().st_size/1e9:.2f} GB) {zip_path}",
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
                info = extract_clip_subprocess(vpath, int(rec.sequence_id))
                rest_id = None
                if info.get("rest_saved"):
                    rest_id = 1_000_000 + int(rec.sequence_id)
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
            except (Exception, subprocess.TimeoutExpired) as e:  # noqa: BLE001
                append_manifest({"sequence_id": int(rec.sequence_id), "ok": False,
                                 "error": str(e), "zip": key})
                print(f"    FAIL {rec.sequence_id}: {e}", flush=True)
            finally:
                if vpath.exists():
                    vpath.unlink()
            if remaining is not None and n_ok >= remaining:
                break
    if downloaded and not keep_zips:
        try:
            zip_path.unlink()
        except OSError:
            pass
    elif keep_zips:
        print(f"    keep {zip_path.name}", flush=True)
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
    ap.add_argument("--zip-dir", action="append", default=[],
                    help="directory of already-downloaded Zenodo zips (repeatable)")
    ap.add_argument("--keep-zips", action="store_true",
                    help="do not delete zips after landmark extract")
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
    done_ids = load_done()
    pending_cats = set(
        index.loc[~index.sequence_id.isin(done_ids), "category"]
        .str.replace(" ", "_", regex=False)
    )
    already = zips_fully_consumed()
    files = [f for f in files
             if zip_category(f["key"]).replace(" ", "_") in pending_cats
             and f["key"] not in already]
    if args.max_zips:
        files = files[:args.max_zips]
    print(f"pending INCLUDE-50 clips {len(index) - len(done_ids & set(index.sequence_id))}",
          flush=True)
    print(f"pending categories {sorted(pending_cats)}", flush=True)
    print(f"skip already-extracted zips {len(already)}", flush=True)
    print(f"zenodo zips to scan: {len(files)} {[f['key'] for f in files]}", flush=True)

    zip_dirs = [Path(p) for p in args.zip_dir] + [TMP, DATA / "zips", DATA]
    seen = set()
    uniq_dirs = []
    for d in zip_dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        uniq_dirs.append(d)
        d.mkdir(parents=True, exist_ok=True)
    print(f"zip search dirs {[str(d) for d in uniq_dirs]}", flush=True)

    remaining = args.calibrate if args.calibrate else None
    for zinfo in files:
        if remaining is not None and remaining <= 0:
            break
        n = process_zip(zinfo, index, remaining, uniq_dirs,
                        keep_zips=args.keep_zips)
        if remaining is not None:
            remaining -= n

    index2 = merge_rest_into_index(index)
    index2.to_parquet(INDEX)
    npy = list(FEAT.glob("*.npy"))
    print(f"\nfeatures on disk: {len(npy)}  index {INDEX}", flush=True)
    print(f"labels {LABELS}", flush=True)


if __name__ == "__main__":
    main()
