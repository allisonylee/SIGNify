"""
v016 -- record your own signs for fine-tuning. NOT SHIPPED.

Type a word, sign it N times, move on. Repeat until you quit.

    python backend/scripts/v016_record_signs.py --signer 0
    python backend/scripts/v016_record_signs.py --signer 1 --reps 15

Each rep is AUTO-SEGMENTED by the same motion gate the live recognizer uses,
then run through the same assemble -> normalise -> resample pipeline, so what
you record matches what the model sees at inference. That matters: GISLR clips
are cropped tight around the sign (measured: motion is flat from frame 1), so
recording with dead air at the start and end would put your samples in a
different distribution from the training data.

Keys while recording a word:
    r   redo the last rep        s   skip the rest of this word
    q   finish this word early   ESC abort everything

WHY --signer MATTERS
--------------------
Pass a different --signer number per person. Without it every sample looks like
one signer and a signer-independent split is impossible -- which means any
accuracy number you get afterwards is inflated and cannot be trusted. This is
the same trap the plan flags for GISLR.

RECORD WORDS THE MODEL ALREADY KNOWS. Your samples then join a class that
already has ~380 GISLR examples, rather than needing a class built from your
handful alone. The script warns when a word is unknown.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import traceback
import time
from collections import deque
from pathlib import Path

# MediaPipe's C++ logging IGNORES GLOG_minloglevel in this build -- it writes
# straight to file descriptor 2, so even level 3 leaves WARNING and ERROR lines
# interleaved with this script's input prompt. The only reliable fix is to
# redirect fd 2 itself (see quiet_stderr below). These are kept anyway because
# they are harmless and may work in other builds.
# minloglevel: 0=INFO 1=WARNING 2=ERROR 3=FATAL. It must be 3 to silence the
# "Failed to send to clearcut" lines -- those are logged at ERROR level and
# were interleaving with this script's input prompt. They are Google telemetry
# uploads failing, not a problem with recording. Real MediaPipe failures still
# surface as Python exceptions, which this does not hide.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_alsologtostderr", "0")

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features
from backend.training.dataset import MINE_SIGNER_BASE, mine_signer_id
from backend.app.landmarks import (
    HolisticExtractor, SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)

DATA = config.ROOT / "data" / "gislr"
MINE = DATA / "mine"
INDEX = DATA / "mine_index.parquet"
ID_BASE = 800_000_000

WIN = "record signs (v016)"
WHITE, GREY, GREEN, RED, CYAN = ((255, 255, 255), (150, 150, 150),
                                 (90, 220, 90), (80, 80, 240), (60, 220, 240))


@contextlib.contextmanager
def quiet_stderr(path: Path):
    """
    Send fd 2 to a file for the duration.

    MediaPipe logs from C++ directly to the file descriptor, so Python-level
    suppression cannot touch it. Redirecting the descriptor is the only thing
    that works. The output is kept in a FILE rather than discarded, so a real
    failure is still recoverable -- and callers print tracebacks to stdout so
    they stay visible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = os.dup(2)
    f = open(path, "w")
    try:
        os.dup2(f.fileno(), 2)
        yield path
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        f.close()


def known_labels() -> list[str]:
    import torch
    ck_path = config.MODELS_DIR / "encoder_ase.pt"
    if not ck_path.exists():
        return []
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return [l for l in ck["labels"] if l != "__REST__"]


def next_id() -> int:
    ids = [int(p.stem) for p in MINE.glob("*.npy")] if MINE.exists() else []
    return (max(ids) + 1) if ids else ID_BASE


def draw(view, word, rep, reps, state, energy, det, msg, captured):
    h = view.shape[0] - 132
    cv2.putText(view, f"WORD: {word}", (14, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
    cv2.putText(view, f"rep {rep}/{reps}   captured {captured}", (14, h + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    col = GREEN if state == "RECORDING" else (GREY if state == "ready" else RED)
    cv2.putText(view, state, (14, h + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(view, f"motion {energy:.3f}   hands {det}", (200, h + 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
    cv2.putText(view, msg[:70], (14, h + 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def record_word(cap, extractor, word, reps, args, start_id, rows):
    """Capture `reps` motion-segmented takes of one word. Returns takes saved."""
    # Create the window explicitly and pump the event loop a few times.
    # cv2.imshow alone can fail to surface a window on macOS.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 720, 700)
    cv2.moveWindow(WIN, 60, 60)
    try:
        # You type the word in the TERMINAL, so focus stays there and the
        # window can open behind it. Pin it on top.
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    for _ in range(8):
        cv2.waitKey(1)

    times: deque[float] = deque(maxlen=400)
    frames: deque[np.ndarray] = deque(maxlen=400)
    saved, seg_start, quiet_since = 0, None, None
    last: tuple[int, np.ndarray] | None = None
    msg = "SPACE to start" if args.manual else "sign when ready"
    t0 = time.perf_counter()

    while saved < reps:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[1] > args.width:
            sc = args.width / bgr.shape[1]
            bgr = cv2.resize(bgr, (args.width, int(bgr.shape[0] * sc)),
                             interpolation=cv2.INTER_AREA)
        t = time.perf_counter() - t0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands, pose = extractor.extract(rgb, int(t * 1000))
        a = features.assemble(hands, pose)
        times.append(t)
        frames.append(features.normalise(a))

        energy = (features.motion_energy(np.stack(list(frames)[-3:]))
                  if len(frames) >= 3 else 0.0)
        moving = energy > args.motion_threshold
        det = (f"L{a[SLICE_LEFT_HAND,2].mean():3.0%} "
               f"R{a[SLICE_RIGHT_HAND,2].mean():3.0%}")
        state = "ready"

        # In MANUAL mode you decide the boundaries with SPACE; the motion gate
        # is only displayed, never acted on. More reliable when the gate
        # mis-fires, and it lets you capture signs with internal holds.
        stop_now = False
        if args.manual:
            if seg_start is not None:
                state = "RECORDING"
        elif seg_start is None:
            if moving:
                seg_start, quiet_since = len(times) - 1, None
                msg = "recording..."
        else:
            state = "RECORDING"
            if moving:
                quiet_since = None
            else:
                if quiet_since is None:
                    quiet_since = t
                elif t - quiet_since >= args.quiet_seconds:
                    stop_now = True
        if args.manual and seg_start is not None and getattr(args, "_stop", False):
            stop_now = True
            args._stop = False
        if stop_now:
            n = len(times) - seg_start
            if n >= args.min_frames:
                ts = np.array(list(times)[seg_start:])
                fs = np.stack(list(frames)[seg_start:])
                span = max(ts[-1] - ts[0], 1e-3)
                w = features.resample_by_time(
                    (ts - ts[0]) / span, fs, config.WINDOW_FRAMES, 1.0)
                sid = start_id + saved
                np.save(MINE / f"{sid}.npy", w)
                rows.append({"sequence_id": sid, "sign": word,
                             "participant_id": mine_signer_id(args.signer),
                             "source": "mine", "signer_name": args.name,
                             "path": ""})
                last = (sid, w)
                saved += 1
                msg = f"saved rep {saved} ({n} frames, {span:.1f}s)"
                print(f"    rep {saved}/{reps}: {n} frames, {span:.2f}s")
            else:
                msg = f"too short ({n} frames) -- discarded"
            seg_start, quiet_since = None, None

        view = cv2.flip(bgr, 1)
        view = cv2.copyMakeBorder(view, 0, 132, 0, 0, cv2.BORDER_CONSTANT,
                                  value=(24, 24, 24))
        draw(view, word, min(saved + 1, reps), reps, state, energy, det, msg, saved)
        hint = ("SPACE start/stop   r redo   s skip   q next word   ESC abort"
                if args.manual else
                "r redo   s skip word   q finish word   ESC abort")
        cv2.putText(view, hint,
                    (14, view.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
        cv2.imshow(WIN, view)

        k = cv2.waitKey(1) & 0xFF
        if args.manual and k == ord(" "):
            if seg_start is None:
                seg_start, quiet_since = len(times) - 1, None
                msg = "RECORDING -- SPACE to stop"
            else:
                args._stop = True
                msg = "stopping..."
        if k == 27:
            return saved, True
        if k == ord("q") or k == ord("s"):
            return saved, False
        if k == ord("r") and last is not None:
            sid, _ = last
            (MINE / f"{sid}.npy").unlink(missing_ok=True)
            rows[:] = [r for r in rows if r["sequence_id"] != sid]
            saved -= 1; last = None
            msg = "redo: last rep deleted"
            print(f"    redo -- deleted rep {saved + 1}")
    return saved, False


def drop_word(word: str, existing, rows) -> int:
    """
    Remove every existing sample of `word`, ARCHIVING the .npy files.

    Used by --replace when a word was recorded badly (bad framing, hands out of
    shot) and you want to start that word over rather than mix good takes with
    bad. Archived rather than deleted because a recording session is real
    effort and the files are tiny.
    """
    ids = [int(x) for x in existing[existing.sign == word].sequence_id]
    ids += [r["sequence_id"] for r in rows if r["sign"] == word]
    if not ids:
        return 0
    dest = DATA / f"mine_replaced_{word}"
    dest.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        f = MINE / f"{sid}.npy"
        if f.exists():
            shutil.move(str(f), str(dest / f"{sid}.npy"))
    rows[:] = [r for r in rows if r["sign"] != word]
    return len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", type=int, required=True,
                    help="0,1,2... one per PERSON. Stored as %d+N so the data "
                         "itself records that we are NOT GISLR signers "
                         "(GISLR uses 2044..62590)." % MINE_SIGNER_BASE)
    ap.add_argument("--name", default="",
                    help="optional human label for this signer, e.g. allison")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--motion-threshold", type=float, default=config.REC_MOTION_THRESHOLD)
    ap.add_argument("--quiet-seconds", type=float, default=config.REC_QUIET_SECONDS)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--quiet", action="store_true",
                    help="redirect MediaPipe's stderr spam to a log file. OFF "
                         "by default: it has twice hidden real errors, which "
                         "is worse than the noise it removes.")
    ap.add_argument("--replace", action="store_true",
                    help="discard existing samples of each word you record in "
                         "this session (archived, not deleted)")
    ap.add_argument("--manual", action="store_true",
                    help="press SPACE to start and stop each rep yourself, "
                         "instead of auto-detecting motion")
    args = ap.parse_args()
    args._stop = False

    MINE.mkdir(parents=True, exist_ok=True)
    known = set(known_labels())
    existing = pd.read_parquet(INDEX) if INDEX.exists() else pd.DataFrame(
        columns=["sequence_id", "sign", "participant_id", "source",
                 "signer_name", "path"])
    if len(existing):
        print(f"{len(existing)} samples already recorded "
              f"({existing.sign.nunique()} words, "
              f"signers {sorted(existing.participant_id.unique())})")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        # stderr is redirected to a log, so say this on stdout or it vanishes.
        print("\n!! CANNOT OPEN CAMERA. Grant Camera permission to your terminal:")
        print("   System Settings > Privacy & Security > Camera, then restart it.")
        sys.exit(1)
    ok, probe = cap.read()
    if not ok:
        print("\n!! camera opened but returned no frame -- is another program using it?")
        sys.exit(1)
    print(f"\ncamera OK: {probe.shape[1]}x{probe.shape[0]}")
    extractor = HolisticExtractor()
    rows: list[dict] = []
    aborted = False

    print(f"\nsigner {args.signer} -> participant_id "
          f"{mine_signer_id(args.signer)} (source=mine, NOT a GISLR signer)")
    print(f"{args.reps} reps per word, "
          f"{'MANUAL (SPACE to start/stop)' if args.manual else 'auto motion-gated'}.")
    print("Type a word and press enter, then the PREVIEW WINDOW OPENS.")
    print("(no window appears while you are at this prompt -- that is normal)")
    print("Blank line to finish.\n")
    try:
        while True:
            word = input("word (blank = done) > ").strip().lower().replace(" ", "")
            if not word:
                break
            if known and word not in known:
                print(f"  note: '{word}' is NOT one of the 250 GISLR signs.")
                print("     Supported -- fine-tuning will add a new class for it.")
                print("     But it learns from YOUR samples only, with no GISLR")
                print("     examples behind it, so record MORE reps than usual")
                print("     (15-20 rather than 10).")
                if input("     record it? [Y/n]: ").strip().lower() == "n":
                    continue
            if args.replace:
                n_old = drop_word(word, existing, rows)
                if n_old:
                    existing = existing[existing.sign != word].reset_index(drop=True)
                    print(f"  replaced: archived {n_old} old '{word}' samples")
                else:
                    print(f"  no existing '{word}' samples to replace")
            print(f"  recording '{word}' -- sign it {args.reps} times, "
                  f"pausing between each")
            # next_id() rescans disk, and record_word writes as it goes,
            # so adding len(rows) here would double-count.
            n, aborted = record_word(cap, extractor, word, args.reps, args,
                                     next_id(), rows)
            print(f"  '{word}': {n} reps captured")
            # Flush after every word. Previously the index was written only at
            # the end, so a crash or a force-quit left the .npy files on disk
            # with no index rows -- recorded but invisible to training.
            if rows:
                pd.concat([existing, pd.DataFrame(rows)],
                          ignore_index=True).to_parquet(INDEX)
            if aborted:
                break
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted")
    finally:
        cap.release()
        try:
            extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    if rows:
        df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        # keep only rows whose .npy still exists -- --replace moved some away
        alive = {int(f.stem) for f in MINE.glob("*.npy")}
        df = df[df.sequence_id.isin(alive)].reset_index(drop=True)
        df.to_parquet(INDEX)
        print(f"\n{'='*60}")
        print(f"  this session : {len(rows)} samples")
        print(f"  TOTAL        : {len(df)} samples, {df.sign.nunique()} words, "
              f"signers {sorted(df.participant_id.unique())}")
        print(f"  index        : {INDEX}")
        print(f"{'='*60}")
        print("\n  per word:")
        for w, c in df.sign.value_counts().items():
            sub = df[df.sign == w]
            who = sorted(sub.participant_id.unique())
            print(f"    {w:<16} {c:>3} samples across {len(who)} signer(s) "
                  f"{[p - MINE_SIGNER_BASE for p in who]}")
    else:
        print("\nnothing recorded")


if __name__ == "__main__":
    log = config.OUTPUTS_DIR / "v016_mediapipe.log"
    quiet = "--quiet" in sys.argv
    try:
        with quiet_stderr(log) if quiet else contextlib.nullcontext():
            main()
    except SystemExit as e:
        # A bare sys.exit("message") writes to stderr, which is redirected --
        # so the message would disappear into the log and the script would look
        # like it silently did nothing.
        if isinstance(getattr(e, "code", None), str):
            print(e.code)
        raise
    except BaseException:
        # stderr is redirected, so surface the traceback on stdout or it
        # vanishes into the log file.
        print("\n--- error ---")
        traceback.print_exc(file=sys.stdout)
        print(f"(MediaPipe output went to {log})")
        sys.exit(1)
    print(f"\n(MediaPipe chatter was written to {log})")
