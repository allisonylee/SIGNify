"""
v004 — standalone webcam check for encoder_ins.pt.

NOT the shipping app. Does not import FastAPI or change backend/scripts/v009.

    python backend/scriptsISL/v004_live_ins.py
    python backend/scriptsISL/v004_live_ins.py --vocab "Thank you,Sorry,Hello"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features
from backend.app.landmarks import (
    HolisticExtractor, SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)
from backend.app.model import SignClassifier
from backend.trainingISL.paths import CK_INS, LANG, REST_SIGN

WHITE, GREY, GREEN, RED, BLUE, YELLOW = (
    (255, 255, 255), (150, 150, 150), (90, 220, 90),
    (80, 80, 240), (240, 160, 60), (60, 220, 240))


def load_model(dev):
    if not CK_INS.exists():
        sys.exit(f"no {CK_INS}")
    ck = torch.load(CK_INS, map_location="cpu", weights_only=False)
    m = SignClassifier({LANG: len(ck["labels"])})
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    return m, ck["labels"], ck.get("val_acc")


def predict(model, labels, window, dev, allowed=None, k=5):
    x = torch.from_numpy(window[None].astype(np.float32)).to(dev)
    with torch.no_grad():
        logits = model(x, LANG)[0]
    if allowed is not None:
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed] = 0.0
        logits = logits + mask
    p = torch.softmax(logits, -1).cpu().numpy()
    top = np.argsort(p)[::-1][:k]
    return [(labels[i], float(p[i])) for i in top]


def draw_landmarks(img, frame_xy, w, h):
    for sl, col in ((SLICE_LEFT_HAND, BLUE), (SLICE_RIGHT_HAND, RED),
                    (SLICE_POSE, GREEN)):
        for x, y in frame_xy[sl]:
            if x > 0 or y > 0:
                cv2.circle(img, (int(x * w), int(y * h)), 3, col, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--vocab", default="")
    ap.add_argument("--motion-threshold", type=float, default=0.020)
    ap.add_argument("--min-sign-frames", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, val_acc = load_model(dev)
    print(f"ISL head {len(labels)} classes, recorded val {val_acc}, {dev}")
    print("this process does not talk to the FastAPI websocket")

    allowed = None
    if args.vocab:
        want = [v.strip() for v in args.vocab.split(",") if v.strip()]
        idx = [labels.index(v) for v in want if v in labels]
        if idx:
            allowed = torch.tensor(idx, device=dev)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("cannot open camera")

    extractor = HolisticExtractor()
    times, frames = deque(maxlen=200), deque(maxlen=200)
    committed, live = [], []
    seg_active, seg_start, quiet = False, 0, 0
    t_start = time.perf_counter()
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape[1] > args.width:
                sc = args.width / bgr.shape[1]
                bgr = cv2.resize(bgr, (args.width, int(bgr.shape[0] * sc)))
            h, w = bgr.shape[:2]
            t = time.perf_counter() - t_start
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hands, pose = extractor.extract(rgb, int(t * 1000))
            assembled = features.assemble(hands, pose)
            times.append(t)
            frames.append(features.normalise(assembled))
            energy = features.motion_energy(np.stack(list(frames)[-3:])) if len(frames) >= 3 else 0.0
            moving = energy > args.motion_threshold
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if moving and not seg_active:
                seg_active, seg_start, quiet = True, len(times) - 1, 0
            elif seg_active and not moving:
                quiet += 1
                if quiet >= 5:
                    if len(times) - seg_start >= args.min_sign_frames:
                        ts = np.array(list(times)[seg_start:])
                        fs = np.stack(list(frames)[seg_start:])
                        span = max(ts[-1] - ts[0], 1e-3)
                        win = features.resample_by_time(
                            (ts - ts[0]) / span, fs, config.WINDOW_FRAMES, 1.0)
                        top = predict(model, labels, win, dev, allowed)
                        if top[0][0] != REST_SIGN:
                            committed.append(top[0])
                    seg_active = False
            if len(times) >= 4 and len(times) % 3 == 0:
                ts = np.array(times)
                fs = np.stack(frames)
                recent = ts >= ts[-1] - min(ts[-1] - ts[0], config.WINDOW_SECONDS)
                tt = ts[recent]
                live = predict(model, labels, features.resample_by_time(
                    (tt - tt[0]) / max(tt[-1] - tt[0], 1e-3),
                    fs[recent], config.WINDOW_FRAMES, 1.0), dev, allowed)

            draw_landmarks(bgr, assembled[:, :2], w, h)
            view = cv2.flip(bgr, 1)
            cv2.putText(view, "ISL standalone — not integrated", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            cv2.putText(view, "SIGNING" if moving else "idle", (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN if moving else GREY, 2)
            if live:
                cv2.putText(view, f"{live[0][0]} {live[0][1]:.0%}", (16, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            cv2.imshow("ISL live (scriptsISL)", view)
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()
    print("committed:", committed)


if __name__ == "__main__":
    main()
