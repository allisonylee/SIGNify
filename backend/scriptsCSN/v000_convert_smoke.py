"""Synthetic LSC50 CSV -> (32,53,3) without downloading Figshare."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.trainingCSN.convert_csv import to_window


POSE_IDX = [0, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]


def _csv(n_lm: int, t=40) -> bytes:
    cols = {}
    rng = np.random.default_rng(0)
    for i in range(n_lm):
        cols[f"landmark_{i}_x"] = rng.uniform(0.2, 0.8, t)
        cols[f"landmark_{i}_y"] = rng.uniform(0.2, 0.8, t)
        cols[f"landmark_{i}_z"] = rng.uniform(-0.1, 0.1, t)
    # shoulders far enough apart to pass MIN_SHOULDER_WIDTH
    if n_lm >= 33:
        cols["landmark_11_x"][:] = 0.35
        cols["landmark_12_x"][:] = 0.65
        cols["landmark_11_y"][:] = 0.4
        cols["landmark_12_y"][:] = 0.4
    buf = io.BytesIO()
    pd.DataFrame(cols).to_csv(buf, index=False)
    return buf.getvalue()


def main():
    info = to_window(_csv(33), _csv(42), POSE_IDX)
    w = info["window"]
    assert w.shape == (32, 53, 3), w.shape
    assert w.dtype == np.float32
    print(f"convert_csv smoke OK  window {w.shape}  hands={info['hands_detected_frac']:.0%}  E={info['motion_energy']:.3f}")


if __name__ == "__main__":
    main()
