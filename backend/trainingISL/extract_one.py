"""
Extract one INCLUDE clip in a child process.

MediaPipe can abort the process on a bad frame. Running this as `-m` keeps
the zip loop alive. Do not import this from app/.

    python -m backend.trainingISL.extract_one VIDEO OUT_NPY TIMESTAMP_MS REST_NPY
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from backend.app.landmarks import HolisticExtractor
from backend.trainingISL.extract_video import extract_clip, rest_window_from_padding


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: extract_one VIDEO OUT_NPY TIMESTAMP_MS [REST_NPY]")
    video = Path(sys.argv[1])
    out = Path(sys.argv[2])
    stamp = int(sys.argv[3])
    rest_out = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    extractor = HolisticExtractor()
    try:
        info = extract_clip(video, extractor, timestamp_base_ms=stamp)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, info["window"])
        rest_saved = False
        if rest_out is not None:
            rest = rest_window_from_padding(info["raw"])
            if rest is not None:
                np.save(rest_out, rest)
                rest_saved = True
        print(json.dumps({
            "ok": True,
            "n_frames": info["n_frames"],
            "duration_s": info["duration_s"],
            "hands_detected_frac": info["hands_detected_frac"],
            "motion_energy": info["motion_energy"],
            "rest_saved": rest_saved,
        }), flush=True)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
