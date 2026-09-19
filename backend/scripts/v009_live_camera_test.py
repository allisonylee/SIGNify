"""
v009 -- live camera sanity check. NOT SHIPPED.

Sign into the webcam and watch what the encoder predicts, in real time. This
drives the SAME recognizer app/main.py uses (app/inference.ModelRecognizer),
so what you see here is what the app does.

    python backend/scripts/v009_live_camera_test.py
    python backend/scripts/v009_live_camera_test.py --vocab flower,clown,store,horse
    python backend/scripts/v009_live_camera_test.py --min-confidence 0.5

Keys:  q / ESC quit    r reset    SPACE force a prediction now

WHAT TO EXPECT
--------------
The encoder was trained on ISOLATED, pre-segmented signs. A webcam is a
CONTINUOUS stream, so the recognizer has to find sign boundaries itself:

  live top-5   ungated sliding window. Flickers constantly, including when you
               are not signing. That is the train/deploy gap, not a bug.
  COMMITTED    gated path -- motion onset, then stillness, then confidence and
               margin checks and a debounce. These are the real outputs.

Until a REST class is trained (scripts/v010_record_rest.py), the confidence and
margin floors are the only thing suppressing false positives, because the model
has never seen "nothing is happening".

Best-known signs:  flower .95  clown .92  store .90  horse .89  airplane .88
Worst:             chin .05  owie .05  every .08  go .14   -- do not judge on these

MIRRORING (plan R9): the PREVIEW is mirrored so it feels like a mirror, but the
landmarks fed to the model are UN-MIRRORED. Flipping both would swap the hands.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

# Must precede any mediapipe import: silences the NORM_RECT warning and the
# "Failed to send to clearcut" telemetry errors.
# Quieten MediaPipe's C++ logging BEFORE anything imports it.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import features
from backend.app.inference import FrameBuffer, load_model_recognizer
from backend.app.ingest import LandmarkFrame
from backend.app.landmarks import (
    HolisticExtractor, SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)

WHITE, GREY, GREEN, RED, BLUE, YELLOW = (
    (255, 255, 255), (150, 150, 150), (90, 220, 90),
    (80, 80, 240), (240, 160, 60), (60, 220, 240))


def draw_landmarks(img, xy, w, h):
    for sl, col in ((SLICE_LEFT_HAND, BLUE), (SLICE_RIGHT_HAND, RED),
                    (SLICE_POSE, GREEN)):
        for x, y in xy[sl]:
            if x > 0 or y > 0:
                cv2.circle(img, (int(x * w), int(y * h)), 3, col, -1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--vocab", default="", help="comma-separated signs to restrict to")
    ap.add_argument("--motion-threshold", type=float, default=None,
                    help="override; default 0.008, calibrated on real GISLR motion")
    ap.add_argument("--min-sign-frames", type=int, default=8)
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--min-margin", type=float, default=0.10)
    ap.add_argument("--debounce", type=float, default=1.0)
    ap.add_argument("--no-landmarks", action="store_true")
    args = ap.parse_args()

    rec = load_model_recognizer(
        motion_threshold=args.motion_threshold,
        min_sign_frames=args.min_sign_frames,
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
        debounce_s=args.debounce)
    if rec is None:
        sys.exit("no checkpoint -- run `python -m backend.training.train` first")
    labels = rec.labels
    print(f"model   {len(labels)} classes on {rec.device}")
    print(f"rest class trained: {rec.rest_idx is not None}"
          + ("" if rec.rest_idx is not None
             else "   <- run v010_record_rest.py, then train --with-rest"))
    print(f"guards  conf>={rec.min_confidence}  margin>={rec.min_margin}  "
          f"motion>{rec.motion_threshold}  debounce={rec.debounce_s}s")

    if args.vocab:
        want = [v.strip() for v in args.vocab.split(",") if v.strip()]
        idx = [labels.index(v) for v in want if v in labels]
        if missing := [v for v in want if v not in labels]:
            print(f"not in vocabulary, ignored: {missing}")
        if idx:
            rec.allowed = torch.tensor(idx, device=rec.device)
            print(f"restricted to {len(idx)}: {[labels[i] for i in idx]}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("cannot open camera -- grant Camera permission to your terminal "
                 "in System Settings > Privacy & Security")
    extractor = HolisticExtractor()

    buf = FrameBuffer(max_frames=256)
    committed: list[tuple[float, str, float]] = []   # (t, gloss, conf)
    live: list[tuple[str, float]] = []
    fps_hist: deque[float] = deque(maxlen=30)
    t_start = time.perf_counter()
    interrupted = False

    print("\nwindow open. q/ESC quit, r reset, SPACE force a prediction.")
    print("(the OpenCV window needs focus for keys; Ctrl-C also exits cleanly)\n")
    try:
        while True:
            t_loop = time.perf_counter()
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape[1] > args.width:
                sc = args.width / bgr.shape[1]
                bgr = cv2.resize(bgr, (args.width, int(bgr.shape[0] * sc)),
                                 interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
            t = time.perf_counter() - t_start

            # UN-MIRRORED to MediaPipe; only the preview gets flipped.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hands, pose = extractor.extract(rgb, int(t * 1000))
            assembled = features.assemble(hands, pose)
            buf.add(LandmarkFrame(t=t, hands=hands, pose=pose))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                committed.clear()
                for k in rec.rejected:
                    rec.rejected[k] = 0
            force = key == ord(" ")

            # The SHIPPING path: gating -> confidence -> margin -> debounce.
            if (result := rec.observe(buf, t)) is not None:
                committed.append((t, result.gloss[0], result.confidence))
                print(f"  [{t:6.1f}s] {result.gloss[0]}  {result.confidence:.0%}",
                      flush=True)

            # Ungated readout, purely so you can see the model thinking.
            if len(buf) >= 4 and buf.total % 3 == 0:
                live = rec.score(buf.window())
            if force and live:
                committed.append((t, live[0][0], live[0][1]))

            # ---- draw ----
            if not args.no_landmarks:
                draw_landmarks(bgr, assembled[:, :2], w, h)
            view = cv2.flip(bgr, 1)
            view = cv2.copyMakeBorder(view, 0, 150, 0, 270,
                                      cv2.BORDER_CONSTANT, value=(24, 24, 24))
            H, W = view.shape[:2]
            signing = rec.state == "signing"
            lh = assembled[SLICE_LEFT_HAND, 2].mean()
            rh = assembled[SLICE_RIGHT_HAND, 2].mean()
            fps_hist.append(1.0 / max(time.perf_counter() - t_loop, 1e-6))

            cv2.rectangle(view, (w + 8, 8), (W - 8, 132), (40, 40, 40), -1)
            cv2.putText(view, "SIGNING" if signing else "idle", (w + 18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        GREEN if signing else GREY, 2)
            cv2.putText(view, f"motion {rec.last_energy:.3f}", (w + 18, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
            cv2.putText(view, f"L {lh:3.0%}   R {rh:3.0%}", (w + 18, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        WHITE if (lh > 0 or rh > 0) else RED, 1)
            cv2.putText(view, f"{np.mean(fps_hist):4.1f} fps   buf {len(buf)}",
                        (w + 18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)

            cv2.putText(view, "live top-5 (ungated, flickers)", (w + 18, 162),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
            for i, (name, prob) in enumerate(live[:5]):
                yy = 186 + i * 26
                cv2.rectangle(view, (w + 18, yy - 12),
                              (w + 18 + int(210 * prob), yy + 4),
                              (60, 90, 60) if i == 0 else (55, 55, 55), -1)
                cv2.putText(view, f"{name[:14]:<14} {prob:4.0%}", (w + 22, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            WHITE if i == 0 else GREY, 1)

            cv2.putText(view, "COMMITTED  (motion + confidence + margin + debounce)",
                        (16, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
            txt = "  ".join(f"{n}({p:.0%})" for _, n, p in committed[-6:]) or "-"
            cv2.putText(view, txt[:98], (16, h + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            cv2.putText(view, "q quit   r reset   SPACE force", (16, h + 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
            rj = rec.rejected
            cv2.putText(view, f"rejected: conf {rj['confidence']}  "
                              f"margin {rj['margin']}  short {rj['too_short']}  "
                              f"debounce {rj['debounce']}  rest {rj['rest']}",
                        (16, h + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)

            cv2.imshow("sign language sanity check (v009)", view)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        cap.release()
        try:
            extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    if interrupted:
        print("\ninterrupted -- camera released, window closed.")
    print(f"\ncommitted {len(committed)} signs, in order:")
    prev = 0.0
    for t, n, p in committed:
        print(f"  [{t:6.1f}s]  +{t-prev:5.1f}s   {n:<16} {p:.0%}")
        prev = t
    if committed:
        print(f"  (gaps show the rest between signs; a gap near 0 means two "
              f"emissions ran together)")
    print(f"rejected: {rec.rejected}")
    if not committed:
        print("  (none -- try --vocab flower,clown,store,horse "
              "or lower --min-confidence)")


if __name__ == "__main__":
    main()
