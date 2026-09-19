"""
Video clip -> (32, 53, 3) using the live HolisticLandmarker + app.features.

Isolated signs are resampled over the WHOLE clip (span=1.0 on a 0..1 time
axis), matching GISLR convert_fast.py. Do not resample a trailing 1.5 s window
or the start of the sign is dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config, features  # noqa: E402
from backend.app.landmarks import (  # noqa: E402
    HolisticExtractor, SLICE_LEFT_HAND, SLICE_RIGHT_HAND,
)


def extract_clip(video_path: Path, extractor: HolisticExtractor,
                 timestamp_base_ms: int = 0) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times, frames = [], []
    i = 0
    hands_hit = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t_s = i / float(fps)
            hands, pose = extractor.extract(
                rgb, timestamp_base_ms + int(t_s * 1000))
            assembled = features.assemble(hands, pose)
            frames.append(features.normalise(assembled))
            times.append(t_s)
            if assembled[SLICE_LEFT_HAND, 2].mean() > 0.5 or assembled[SLICE_RIGHT_HAND, 2].mean() > 0.5:
                hands_hit += 1
            i += 1
    finally:
        cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"too few frames in {video_path}")

    stacked = np.stack(frames, axis=0)
    t = np.arange(len(stacked), dtype=np.float64) / max(len(stacked) - 1, 1)
    window = features.resample_by_time(t, stacked, config.WINDOW_FRAMES, 1.0)
    n = len(frames)
    return {
        "window": window.astype(np.float32),
        "n_frames": n,
        "fps": float(fps),
        "duration_s": float(times[-1]) if times else 0.0,
        "hands_detected_frac": hands_hit / max(n, 1),
        "motion_energy": float(features.motion_energy(window)),
        "raw": stacked,
        "raw_times": np.asarray(times, dtype=np.float64),
    }


def rest_window_from_padding(raw: np.ndarray, frac=0.2) -> np.ndarray | None:
    """Low-motion head (or tail) of an isolated clip, resampled to 32 frames."""
    n = len(raw)
    k = max(4, int(n * frac))
    for sl in (raw[:k], raw[-k:]):
        t = np.arange(len(sl), dtype=np.float64) / max(len(sl) - 1, 1)
        w = features.resample_by_time(t, sl, config.WINDOW_FRAMES, 1.0)
        if features.motion_energy(w) < 0.015:
            return w.astype(np.float32)
    return None
