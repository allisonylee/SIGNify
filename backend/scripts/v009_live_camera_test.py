"""
v009 -- live camera sanity check. NOT SHIPPED.

Sign into the webcam and watch what the encoder predicts, in real time.

    python backend/scripts/v009_live_camera_test.py
    python backend/scripts/v009_live_camera_test.py --vocab flower,clown,store,horse
    python backend/scripts/v009_live_camera_test.py --camera 1 --no-landmarks

Keys:  q quit   r reset committed list   SPACE force a prediction now

WHAT TO EXPECT -- read this before judging the output
-----------------------------------------------------
The encoder was trained on ISOLATED, pre-segmented signs. A webcam is a
CONTINUOUS stream, so this tool has to guess where a sign starts and stops.
That segmentation is Stage 3 work and is only crudely approximated here
(motion energy on the wrists). Expect:

  * the live top-5 strip to flicker constantly -- it is scoring a sliding
    window whether or not you are signing. That is the train/deploy gap, not
    a broken model.
  * COMMITTED words (the gated path) to be the meaningful ones. Those come
    from a motion start/stop segment, which is closer to how the model was
    trained.
  * accuracy well below the 57% benchmark if you are not a fluent signer
    producing the sign the way 21 Deaf signers in GISLR produced it.

Signs the model knows best (try these first):
    flower .95  clown .92  store .90  horse .89  stuck .88  airplane .88
Signs it is bad at (short, low-motion -- do not judge it on these):
    chin .05  owie .05  every .08  go .14

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

# Quieten MediaPipe's C++ logging before anything imports it. These must be set
# pre-import to take effect. Suppresses the harmless NORM_RECT warning and the
# "Failed to send to clearcut" telemetry-upload errors.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features
from backend.app.landmarks import (
    HolisticExtractor, ROW_LEFT_WRIST, ROW_RIGHT_WRIST,
    SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)
from backend.app.model import SignClassifier

WHITE, GREY, GREEN, RED, BLUE, YELLOW = (
    (255, 255, 255), (150, 150, 150), (90, 220, 90),
    (80, 80, 240), (240, 160, 60), (60, 220, 240))


def load_model(dev):
    ck_path = config.MODELS_DIR / "encoder_ase.pt"
    if not ck_path.exists():
        sys.exit(f"no checkpoint at {ck_path} -- run backend.training.train first")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = SignClassifier({"ase": len(ck["labels"])})
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    return m, ck["labels"], ck.get("val_acc")


def predict(model, labels, window, dev, allowed=None, k=5):
    x = torch.from_numpy(window[None].astype(np.float32)).to(dev)
    with torch.no_grad():
        logits = model(x, "ase")[0]
    if allowed is not None:
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed] = 0.0
        logits = logits + mask
    p = torch.softmax(logits, -1).cpu().numpy()
    top = np.argsort(p)[::-1][:k]
    return [(labels[i], float(p[i])) for i in top]


def draw_landmarks(img, frame_xy, w, h):
    """frame_xy: raw (53,2) in [0,1] image coords, pre-normalisation."""
    for sl, col in ((SLICE_LEFT_HAND, BLUE), (SLICE_RIGHT_HAND, RED),
                    (SLICE_POSE, GREEN)):
        for x, y in frame_xy[sl]:
            if x > 0 or y > 0:
                cv2.circle(img, (int(x * w), int(y * h)), 3, col, -1)


def panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y + h, x:x + w]
    cv2.addWeighted(np.zeros_like(sub), alpha, sub, 1 - alpha, 0, sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--vocab", default="", help="comma-separated signs to restrict to")
    ap.add_argument("--motion-threshold", type=float, default=0.020,
                    help="wrist speed (shoulder-widths/frame) counting as 'signing'")
    ap.add_argument("--min-sign-frames", type=int, default=8)
    ap.add_argument("--no-landmarks", action="store_true")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, labels, val_acc = load_model(dev)
    print(f"model: 250 signs, signer-independent val {val_acc:.3f}, device {dev}")

    allowed = None
    if args.vocab:
        want = [v.strip() for v in args.vocab.split(",") if v.strip()]
        idx = [labels.index(v) for v in want if v in labels]
        missing = [v for v in want if v not in labels]
        if missing:
            print(f"not in vocabulary, ignored: {missing}")
        if idx:
            allowed = torch.tensor(idx, device=dev)
            print(f"restricted to {len(idx)} signs: {[labels[i] for i in idx]}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("cannot open camera -- grant Camera permission to your terminal "
                 "in System Settings > Privacy & Security")

    extractor = HolisticExtractor()
    times: deque[float] = deque(maxlen=200)
    frames: deque[np.ndarray] = deque(maxlen=200)
    raw_xy: np.ndarray | None = None
    committed: list[tuple[str, float]] = []
    live: list[tuple[str, float]] = []

    seg_active, seg_start = False, 0
    quiet = 0
    fps_hist: deque[float] = deque(maxlen=30)
    t_start = time.perf_counter()

    print("\nwindow open. q or ESC to quit, r reset, SPACE force a prediction.")
    print("(the OpenCV window must have focus for keys to register;")
    print(" Ctrl-C in the terminal also works and now exits cleanly)\n")
    interrupted = False
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

          # UN-MIRRORED frame goes to MediaPipe. Only the preview is flipped.
          rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
          hands, pose = extractor.extract(rgb, int(t * 1000))
          assembled = features.assemble(hands, pose)
          raw_xy = assembled[:, :2].copy()
          times.append(t)
          frames.append(features.normalise(assembled))

          # --- motion gating -------------------------------------------------
          energy = 0.0
          if len(frames) >= 3:
              recent = np.stack(list(frames)[-3:])
              energy = features.motion_energy(recent)
          moving = energy > args.motion_threshold

          force = False
          key = cv2.waitKey(1) & 0xFF
          if key == ord("q"):
              break
          if key == ord("r"):
              committed.clear()
          if key == ord(" "):
              force = True

          if moving and not seg_active:
              seg_active, seg_start, quiet = True, len(times) - 1, 0
          elif seg_active and not moving:
              quiet += 1
              if quiet >= 5:                      # ~0.3 s of stillness ends a sign
                  n = len(times) - seg_start
                  if n >= args.min_sign_frames:
                      ts = np.array(list(times)[seg_start:])
                      fs = np.stack(list(frames)[seg_start:])
                      span = max(ts[-1] - ts[0], 1e-3)
                      win = features.resample_by_time(
                          (ts - ts[0]) / span, fs, config.WINDOW_FRAMES, 1.0)
                      top = predict(model, labels, win, dev, allowed)
                      committed.append(top[0])
                  seg_active = False

          # --- continuous readout (noisy by design) --------------------------
          if len(times) >= 4 and (len(times) % 3 == 0 or force):
              ts = np.array(times); fs = np.stack(frames)
              span = max(ts[-1] - ts[0], 1e-3)
              recent = ts >= ts[-1] - min(span, config.WINDOW_SECONDS)
              tt = ts[recent]
              win = features.resample_by_time(
                  (tt - tt[0]) / max(tt[-1] - tt[0], 1e-3),
                  fs[recent], config.WINDOW_FRAMES, 1.0)
              live = predict(model, labels, win, dev, allowed)
              if force and live:
                  committed.append(live[0])

          # --- draw ----------------------------------------------------------
          if not args.no_landmarks and raw_xy is not None:
              draw_landmarks(bgr, raw_xy, w, h)
          view = cv2.flip(bgr, 1)                 # mirror ONLY for display
          view = cv2.copyMakeBorder(view, 0, 150, 0, 260, cv2.BORDER_CONSTANT,
                                    value=(24, 24, 24))
          H, W = view.shape[:2]

          lh = assembled[SLICE_LEFT_HAND, 2].mean()
          rh = assembled[SLICE_RIGHT_HAND, 2].mean()
          fps_hist.append(1.0 / max(time.perf_counter() - t_loop, 1e-6))

          cv2.rectangle(view, (w + 8, 8), (W - 8, 132), (40, 40, 40), -1)
          cv2.putText(view, "SIGNING" if moving else "idle", (w + 18, 38),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN if moving else GREY, 2)
          cv2.putText(view, f"motion {energy:.3f}", (w + 18, 64),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
          cv2.putText(view, f"L hand {lh:3.0%}   R hand {rh:3.0%}", (w + 18, 88),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                      WHITE if (lh > 0 or rh > 0) else RED, 1)
          cv2.putText(view, f"{np.mean(fps_hist):4.1f} fps", (w + 18, 112),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)

          cv2.putText(view, "live top-5  (flickers: sliding window)", (w + 18, 162),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
          for i, (name, p) in enumerate(live[:5]):
              yy = 186 + i * 26
              cv2.rectangle(view, (w + 18, yy - 12), (w + 18 + int(200 * p), yy + 4),
                            (60, 90, 60) if i == 0 else (55, 55, 55), -1)
              cv2.putText(view, f"{name[:14]:<14} {p:4.0%}", (w + 22, yy),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                          WHITE if i == 0 else GREY, 1)

          cv2.putText(view, "COMMITTED  (on motion stop)", (16, h + 28),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
          txt = "  ".join(f"{n}({p:.0%})" for n, p in committed[-6:]) or "-"
          cv2.putText(view, txt[:96], (16, h + 58), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, WHITE, 2)
          cv2.putText(view, "q quit    r reset    SPACE force prediction",
                      (16, h + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
          cv2.putText(view,
                      "trained on isolated signs; live segmentation is Stage 3",
                      (16, h + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)

          cv2.imshow("sign language sanity check (v009)", view)

    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Ctrl-C can land inside MediaPipe's blocking dispatcher, so cleanup has
        # to be in a finally: otherwise the camera stays held and the window
        # never closes.
        cap.release()
        try:
            extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)          # let macOS actually dismiss the window

    if interrupted:
        print("\ninterrupted -- camera released, window closed.")
    print(f"\ncommitted {len(committed)} signs this session:")
    for n, p in committed:
        print(f"  {n}  {p:.0%}")
    if not committed:
        print("  (none -- try --vocab flower,clown,store,horse to narrow it down)")


if __name__ == "__main__":
    main()
