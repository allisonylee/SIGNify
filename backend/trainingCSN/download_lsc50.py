"""
Download LSC50 LANDMARKS.zip from Figshare, convert pose+hand CSVs to
(32, 53, 3) features, delete the zip. Isolated from backend/training.

    python -m backend.trainingCSN.download_lsc50 --calibrate 20
    python -m backend.trainingCSN.download_lsc50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from backend.trainingCSN.convert_csv import (
    classify_member, parse_tag, rest_sequence_id, rest_window_from_padding,
    sequence_id, to_window,
)
from backend.trainingCSN.paths import (
    DATA, FEAT, INDEX, LABELS, LANDMARKS_URL, LANDMARKS_ZIP_NAME,
    MANIFEST, MIN_FREE_GB_DOWNLOAD, REST_SIGN, TMP, abort_if_low_disk,
    ensure_dirs, free_gb,
)

# Table 2 of Sci Data 11:1347 (2024). Index is the 4-digit class id.
SIGNS = json.loads(Path(__file__).with_name("labels_csn50.json").read_text(encoding="utf-8"))
SIGNS = [s for s in SIGNS if s != REST_SIGN]


def _urlretrieve(url: str, dest: Path, attempts=8):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            print(f"    GET {url}", flush=True)
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


def load_done() -> set[int]:
    done = set()
    if not MANIFEST.exists():
        return done
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


def index_zip(z: zipfile.ZipFile) -> dict[str, dict[tuple[int, int, int], str]]:
    """kind -> tag -> member name."""
    out = {"pose": {}, "hands": {}, "face": {}, "unknown": {}}
    for n in z.namelist():
        kind = classify_member(n)
        if kind is None:
            continue
        tag = parse_tag(n)
        if tag is None:
            continue
        bucket = out.get(kind, out["unknown"])
        bucket[tag] = n
    return out


def write_labels():
    labels = SIGNS + [REST_SIGN]
    LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    return labels


def build_index_from_tags(tags: list[tuple[int, int, int]]) -> pd.DataFrame:
    rows = []
    for sign, vol, rep in sorted(tags):
        gloss = SIGNS[sign] if 0 <= sign < len(SIGNS) else f"SIGN_{sign:04d}"
        sid = sequence_id(sign, vol, rep)
        rows.append({
            "sequence_id": sid,
            "sign": gloss,
            "sign_id": sign,
            "participant_id": vol,
            "repetition": rep,
        })
    return pd.DataFrame(rows)


def merge_rest_into_index(index: pd.DataFrame) -> pd.DataFrame:
    extra = []
    if not MANIFEST.exists():
        return index
    by_id = {int(r.sequence_id): r for r in index.itertuples(index=False)}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rid = row.get("rest_id")
        if not (row.get("ok") and rid):
            continue
        src = by_id.get(int(row["sequence_id"]))
        if src is None:
            continue
        r = src._asdict() if hasattr(src, "_asdict") else dict(src)
        r["sequence_id"] = int(rid)
        r["sign"] = REST_SIGN
        extra.append(r)
    if not extra:
        return index
    return pd.concat([index, pd.DataFrame(extra)], ignore_index=True)


def process_zip(zip_path: Path, remaining: int | None) -> int:
    from backend.app.landmarks import pose_upper_indices
    pose_idx = pose_upper_indices()
    done = load_done()
    n_ok = 0
    with zipfile.ZipFile(zip_path) as z:
        idx = index_zip(z)
        print(f"  zip pose={len(idx['pose'])} hands={len(idx['hands'])} "
              f"face={len(idx['face'])} unknown={len(idx['unknown'])}", flush=True)
        if not idx["pose"] or not idx["hands"]:
            from backend.trainingCSN.convert_csv import csv_to_xy
            print("  pose/hands folders not named; sniffing unknown CSVs", flush=True)
            for tag, member in idx["unknown"].items():
                try:
                    xy = csv_to_xy(z.read(member))
                except Exception:
                    continue
                n = xy.shape[1]
                if n >= 42 and tag not in idx["hands"]:
                    idx["hands"][tag] = member
                elif 30 <= n <= 35 and tag not in idx["pose"]:
                    idx["pose"][tag] = member
        if not idx["pose"] or not idx["hands"]:
            sample = z.namelist()[:30]
            print("  sample members:", sample, flush=True)
            raise SystemExit("LANDMARKS.zip missing pose or hands CSVs")
        tags = sorted(set(idx["pose"]) & set(idx["hands"]))
        print(f"  paired clips {len(tags)}", flush=True)
        index = build_index_from_tags(tags)
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        for tag in tags:
            sign, vol, rep = tag
            sid = sequence_id(sign, vol, rep)
            if sid in done:
                continue
            abort_if_low_disk()
            try:
                pose_raw = z.read(idx["pose"][tag])
                hands_raw = z.read(idx["hands"][tag])
                info = to_window(pose_raw, hands_raw, pose_idx)
                out = FEAT / f"{sid}.npy"
                np.save(out, info["window"])
                rest_id = None
                rest = rest_window_from_padding(info["raw"])
                if rest is not None:
                    rest_id = rest_sequence_id(sid)
                    np.save(FEAT / f"{rest_id}.npy", rest)
                gloss = SIGNS[sign] if 0 <= sign < len(SIGNS) else f"SIGN_{sign:04d}"
                append_manifest({
                    "sequence_id": sid,
                    "ok": True,
                    "n_frames": info["n_frames"],
                    "hands_detected_frac": info["hands_detected_frac"],
                    "motion_energy": info["motion_energy"],
                    "rest_id": rest_id,
                    "sign": gloss,
                    "participant_id": vol,
                })
                n_ok += 1
                print(f"    {sid} {gloss!r} v{vol} r{rep}  "
                      f"{info['n_frames']}f  hands={info['hands_detected_frac']:.0%}  "
                      f"E={info['motion_energy']:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001
                append_manifest({"sequence_id": sid, "ok": False, "error": str(e),
                                 "tag": tag})
                print(f"    FAIL {tag}: {e}", flush=True)
            if remaining is not None and n_ok >= remaining:
                break
        index2 = merge_rest_into_index(index)
        index2.to_parquet(INDEX)
    return n_ok


def resolve_zip(zip_dirs: list[Path]) -> Path | None:
    for d in zip_dirs:
        p = d / LANDMARKS_ZIP_NAME
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, default=0)
    ap.add_argument("--zip-dir", action="append", default=[])
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    write_labels()
    print(f"LSC50 signs {len(SIGNS)} + rest   labels {LABELS}", flush=True)

    zip_dirs = [Path(p) for p in args.zip_dir] + [TMP, DATA, DATA / "zips"]
    zip_path = resolve_zip(zip_dirs)
    downloaded = False
    if zip_path is None:
        abort_if_low_disk(MIN_FREE_GB_DOWNLOAD)
        zip_path = TMP / LANDMARKS_ZIP_NAME
        print(f"download {LANDMARKS_ZIP_NAME} -> {zip_path}  free={free_gb():.2f} GB",
              flush=True)
        t0 = time.perf_counter()
        _urlretrieve(LANDMARKS_URL, zip_path)
        downloaded = True
        print(f"  done in {time.perf_counter()-t0:.0f}s  size={zip_path.stat().st_size/1e9:.2f} GB",
              flush=True)
    else:
        print(f"local {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB)", flush=True)

    remaining = args.calibrate if args.calibrate else None
    n = process_zip(zip_path, remaining)
    print(f"extracted {n}  features {len(list(FEAT.glob('*.npy')))}  index {INDEX}",
          flush=True)

    if downloaded and not args.keep_zip and remaining is None:
        try:
            zip_path.unlink()
            print(f"deleted {zip_path.name}", flush=True)
        except OSError as e:
            print(f"could not delete zip: {e}", flush=True)
    print(f"free disk {free_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
