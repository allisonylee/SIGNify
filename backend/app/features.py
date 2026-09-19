"""
Feature extraction: raw landmarks -> the tensor the model consumes.

*** IMPORTED BY BOTH app/ (inference) AND training/. ***

If training normalises differently from inference you get train/serve skew: it
does not raise, it does not show up in validation (train and val share the
bug), and it surfaces only as "the model is mysteriously bad live". So this is
the single source of truth. Never copy a constant out of here.

Pipeline:
    assemble   -> (53, 3)  rows in canonical order, channel 2 = detected mask
    normalise  -> translate by shoulder midpoint, scale by shoulder width
    resample   -> (32, 53, 3) interpolated ON TIMESTAMPS, not frame indices
"""
from __future__ import annotations

import numpy as np

from .landmarks import (
    N_CHANNELS, N_HAND, N_LANDMARKS, N_POSE_UPPER,
    ROW_LEFT_SHOULDER, ROW_RIGHT_SHOULDER,
    SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)

# Below this, shoulder width is too small to divide by safely (signer far away,
# or a bad detection). Value is in frame-normalised units.
MIN_SHOULDER_WIDTH = 0.02


def assemble(hands: dict, pose: np.ndarray) -> np.ndarray:
    """
    -> (53, 3) float32. Channels are (x, y, detected).

    Missing landmarks get detected=0 and coordinates 0. The mask exists so the
    model can tell "hand absent" from "hand at the origin" -- without it those
    are the same input.
    """
    out = np.zeros((N_LANDMARKS, N_CHANNELS), dtype=np.float32)
    for key, sl in (("left", SLICE_LEFT_HAND), ("right", SLICE_RIGHT_HAND)):
        arr = hands.get(key)
        if arr is not None and len(arr) == N_HAND:
            out[sl, :2] = arr
            out[sl, 2] = 1.0
    if pose is not None and len(pose) == N_POSE_UPPER:
        ok = ~np.isnan(pose).any(axis=1)
        out[SLICE_POSE, :2] = np.nan_to_num(pose)
        out[SLICE_POSE, 2] = ok.astype(np.float32)
    return out


def normalise(frame: np.ndarray) -> np.ndarray:
    """
    Make the frame invariant to where the signer is and how far away.

    Translate so the shoulder midpoint is the origin, then scale by shoulder
    width. Without this the model learns "sign X happens when the person sits
    there" and breaks the moment they shift.

    Returns a copy. Undetected rows stay at 0 with mask 0.
    """
    out = frame.copy()
    lsh, rsh = out[ROW_LEFT_SHOULDER], out[ROW_RIGHT_SHOULDER]
    if lsh[2] < 0.5 or rsh[2] < 0.5:
        return out                        # no anchor; leave raw, mask says so

    origin = (lsh[:2] + rsh[:2]) / 2.0
    width = float(np.abs(lsh[0] - rsh[0]))
    if width < MIN_SHOULDER_WIDTH:
        return out

    detected = out[:, 2] > 0.5
    out[detected, :2] = (out[detected, :2] - origin) / width
    return out


def resample_by_time(
    times: np.ndarray, frames: np.ndarray, n_out: int, span_s: float,
    end_t: float | None = None,
) -> np.ndarray:
    """
    Interpolate `frames` onto `n_out` points evenly spaced over the LAST
    `span_s` seconds ending at `end_t`.

    Resampling on timestamps rather than frame indices is what makes us immune
    to a variable frame rate. The browser delivers whatever a loaded laptop
    manages -- 22 fps here, 30 there -- and the harness delivers something
    else. Indexing by frame would make the model see signs at the wrong speed.

    times:  (T,) seconds, non-decreasing
    frames: (T, 53, 3)
    ->      (n_out, 53, 3)
    """
    times = np.asarray(times, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float32)
    if len(times) == 0:
        return np.zeros((n_out, N_LANDMARKS, N_CHANNELS), dtype=np.float32)
    if len(times) == 1:
        return np.repeat(frames[:1], n_out, axis=0)

    end = float(times[-1]) if end_t is None else float(end_t)
    grid = np.linspace(end - span_s, end, n_out)

    flat = frames.reshape(len(frames), -1)              # (T, 159)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(grid, times, flat[:, c])

    out = out.reshape(n_out, N_LANDMARKS, N_CHANNELS)
    # Interpolating a 0/1 mask yields fractions; re-threshold so "detected"
    # stays boolean and a half-interpolated coordinate is not trusted.
    out[:, :, 2] = (out[:, :, 2] >= 0.5).astype(np.float32)
    return out


def motion_energy(window: np.ndarray) -> float:
    """
    Mean wrist speed across the window, in shoulder-widths per frame.
    Stage 3 uses this for utterance boundaries; Stage 1 only logs it.
    """
    from .landmarks import ROW_LEFT_WRIST, ROW_RIGHT_WRIST
    total, n = 0.0, 0
    for row in (ROW_LEFT_WRIST, ROW_RIGHT_WRIST):
        pts, mask = window[:, row, :2], window[:, row, 2] > 0.5
        if mask.sum() < 2:
            continue
        d = np.linalg.norm(np.diff(pts[mask], axis=0), axis=1)
        total += float(d.sum()); n += len(d)
    return total / n if n else 0.0
