"""
v010 -- record REST (not-signing) footage from the webcam. NOT SHIPPED.

Why this is needed
------------------
The encoder has never seen "nothing is happening". GISLR sequences are cropped
tight around the sign: measured over 60 random sequences, only ~11% of frames
are still and the motion profile is flat from the first frame. There is no
lead-in or lead-out to mine, so the rest class has to be recorded.

Without it, the only thing preventing constant spurious output is a motion
threshold, which cannot tell signing from scratching your nose.

    python backend/scripts/v010_record_rest.py                # ~3 min guided
    python backend/scripts/v010_record_rest.py --seconds 60   # shorter

Runs ACCUMULATE by default. Each session appends new windows with fresh ids,
so running it several times -- different clothing, lighting, time of day, or a
second person -- grows and diversifies the rest set rather than replacing it.

To start over instead:

    python backend/scripts/v010_record_rest.py --reset

--reset ARCHIVES the existing windows to data/gislr/rest_archive_<timestamp>/
rather than deleting them, and asks first. Recovering is a directory move, so
a change of mind costs nothing. Add --yes to skip the prompt, or
--reset --purge to delete outright (asks twice).

Deleting rest data does NOT break the trained checkpoint -- weights are already
baked in. It only matters the next time you run train --with-rest.

It walks through prompts. DO NOT SIGN during any of them -- the point is to
teach the model what your not-signing looks like, including the hand motion
that is not language.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features
from backend.app.landmarks import HolisticExtractor

DATA = config.ROOT / "data" / "gislr"
REST = DATA / "rest"
REST_LABEL = "__REST__"

# Diversity is the whole point: a rest class trained only on "sitting perfectly
# still" will not suppress the false positives that actually happen.
PROMPTS = [
    ("Sit still, hands in your lap or out of frame", 25),
    ("Sit still, hands resting visible on the desk", 25),
    ("Talk out loud, gesture naturally while you talk", 30),
    ("Scratch your face, adjust hair/glasses, rub your eyes", 25),
    ("Reach around: grab a cup, use your phone, point at the screen", 30),
    ("Lean in and out, turn your head, shift in your seat", 25),
    ("Type on a keyboard / use a mouse", 20),
]


def reset_rest(purge: bool = False, assume_yes: bool = False) -> bool:
    """
    Clear the existing rest set. ARCHIVES by default rather than deleting.

    Returns True if it is safe to proceed. Recording several minutes of footage
    is real effort, so the default is reversible and the destructive path asks
    twice.
    """
    idx = DATA / "rest_index.parquet"
    files = sorted(REST.glob("*.npy")) if REST.exists() else []
    if not files and not idx.exists():
        print("no existing rest data -- nothing to reset")
        return True

    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nEXISTING REST DATA: {len(files):,} windows, {mb:.0f} MB")
    if files:
        ids = [int(f.stem) for f in files]
        print(f"  ids {min(ids)}..{max(ids)}")

    if purge:
        print("\n--purge: these files will be DELETED PERMANENTLY.")
        if not assume_yes:
            if input("  type DELETE to confirm: ").strip() != "DELETE":
                return False
            if input(f"  really delete {len(files):,} windows? [y/N]: ").strip().lower() != "y":
                return False
        shutil.rmtree(REST, ignore_errors=True)
        idx.unlink(missing_ok=True)
        REST.mkdir(parents=True, exist_ok=True)
        print(f"  deleted {len(files):,} windows")
        return True

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = DATA / f"rest_archive_{stamp}"
    print(f"\nthese will be MOVED to {dest.name}/ (not deleted).")
    if not assume_yes and input("  proceed? [y/N]: ").strip().lower() != "y":
        return False

    dest.mkdir(parents=True, exist_ok=True)
    if REST.exists():
        shutil.move(str(REST), str(dest / "rest"))
    if idx.exists():
        shutil.move(str(idx), str(dest / "rest_index.parquet"))
    REST.mkdir(parents=True, exist_ok=True)
    print(f"  archived {len(files):,} windows -> {dest}")
    print(f"  to restore:  rm -rf {REST} && mv {dest}/rest {REST} && "
          f"mv {dest}/rest_index.parquet {idx}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seconds", type=int, default=0,
                    help="total seconds; 0 = use the full guided script (~3 min)")
    ap.add_argument("--window-frames", type=int, default=24,
                    help="frames per saved rest window")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--reset", action="store_true",
                    help="clear existing rest data first (archives by default)")
    ap.add_argument("--purge", action="store_true",
                    help="with --reset, DELETE instead of archiving")
    ap.add_argument("--yes", action="store_true",
                    help="skip confirmation prompts")
    args = ap.parse_args()
    if args.purge and not args.reset:
        ap.error("--purge requires --reset")

    if args.reset:
        if not reset_rest(purge=args.purge, assume_yes=args.yes):
            sys.exit("cancelled -- nothing was removed")

    prompts = PROMPTS
    if args.seconds:
        per = max(8, args.seconds // len(PROMPTS))
        prompts = [(t, per) for t, _ in PROMPTS]
    total = sum(d for _, d in prompts)

    REST.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("cannot open camera -- grant Camera permission to your terminal")
    extractor = HolisticExtractor()

    print(f"\nRecording ~{total}s of NOT-SIGNING footage across "
          f"{len(prompts)} prompts.")
    print("Press q any time to stop early.\n")

    times: list[float] = []
    frames: list[np.ndarray] = []
    t_start = time.perf_counter()
    saved = 0
    stop = False

    try:
        for pi, (prompt, dur) in enumerate(prompts, 1):
            if stop:
                break
            t_phase = time.perf_counter()
            while time.perf_counter() - t_phase < dur:
                ok, bgr = cap.read()
                if not ok:
                    stop = True
                    break
                if bgr.shape[1] > args.width:
                    sc = args.width / bgr.shape[1]
                    bgr = cv2.resize(bgr, (args.width, int(bgr.shape[0] * sc)),
                                     interpolation=cv2.INTER_AREA)
                t = time.perf_counter() - t_start
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                hands, pose = extractor.extract(rgb, int(t * 1000))
                times.append(t)
                frames.append(features.normalise(features.assemble(hands, pose)))

                left = dur - (time.perf_counter() - t_phase)
                view = cv2.flip(bgr, 1)
                view = cv2.copyMakeBorder(view, 0, 96, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(24, 24, 24))
                h = view.shape[0] - 96
                cv2.putText(view, f"[{pi}/{len(prompts)}]  DO NOT SIGN",
                            (14, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (60, 220, 240), 2)
                cv2.putText(view, prompt[:58], (14, h + 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                bar = int((1 - left / dur) * (view.shape[1] - 28))
                cv2.rectangle(view, (14, h + 72), (14 + bar, h + 84),
                              (90, 220, 90), -1)
                cv2.putText(view, f"{left:4.0f}s", (view.shape[1] - 70, h + 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.imshow("record rest (v010)", view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    stop = True
                    break
            print(f"  [{pi}/{len(prompts)}] {prompt}  -- {len(frames)} frames so far")
    finally:
        cap.release()
        try:
            extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    if len(frames) < args.window_frames * 2:
        sys.exit(f"only {len(frames)} frames captured -- too few to use")

    # Slice into overlapping windows, resampled exactly like a real segment.
    ts = np.array(times)
    fs = np.stack(frames)
    rows = []

    # IDs continue past whatever is already on disk, so repeated runs ACCUMULATE
    # instead of overwriting. (They previously restarted at 900_000_000 every
    # run, which silently replaced the earlier session.)
    existing = {int(f.stem) for f in REST.glob("*.npy")}
    base = (max(existing) + 1) if existing else 900_000_000
    if existing:
        print(f"\n{len(existing):,} rest windows already on disk; "
              f"appending from id {base}")

    for a in range(0, len(fs) - args.window_frames, args.stride):
        b = a + args.window_frames
        seg_t, seg_f = ts[a:b], fs[a:b]
        span = max(seg_t[-1] - seg_t[0], 1e-3)
        w = features.resample_by_time((seg_t - seg_t[0]) / span, seg_f,
                                      config.WINDOW_FRAMES, 1.0)
        sid = base + saved
        np.save(REST / f"{sid}.npy", w)
        rows.append({"sequence_id": sid, "sign": REST_LABEL,
                     "participant_id": -1, "path": ""})
        saved += 1

    df = pd.DataFrame(rows)
    out = DATA / "rest_index.parquet"
    if out.exists():
        prev = pd.read_parquet(out)
        df = pd.concat([prev[~prev.sequence_id.isin(df.sequence_id)], df],
                       ignore_index=True)
    df.to_parquet(out)

    dur = ts[-1] - ts[0]
    on_disk = len(list(REST.glob("*.npy")))
    print(f"\n{'='*58}")
    print(f"  captured   {len(frames):,} frames over {dur:.0f}s "
          f"({len(frames)/dur:.1f} fps)")
    print(f"  added      {saved:,} rest windows this session")
    print(f"  TOTAL      {on_disk:,} rest windows on disk "
          f"({len(existing):,} before + {saved:,} new)")
    print(f"  index      {out}  ({len(df):,} rows)")
    print(f"{'='*58}")
    if len(df) != on_disk:
        print(f"  !! index rows ({len(df)}) != files on disk ({on_disk})")
    print("\nNext:  python -m backend.training.train --epochs 40 --with-rest")


if __name__ == "__main__":
    main()
