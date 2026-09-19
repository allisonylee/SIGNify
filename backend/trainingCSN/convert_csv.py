"""
LSC50 MediaPipe CSVs -> (32, 53, 3) using the same assemble/normalise/resample
as GISLR. Isolated; does not edit app/features.py.

LSC50 does not ship one holistic 543-row file. Pose (33) and hands (typically
42 = 21+21) live in separate CSVs, tagged SIGN_VOLUNTEER_REP.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app import config, features
from backend.app.features import MIN_SHOULDER_WIDTH
from backend.app.landmarks import (
    N_HAND,
    N_LANDMARKS,
    ROW_LEFT_SHOULDER,
    ROW_RIGHT_SHOULDER,
    SLICE_LEFT_HAND,
    SLICE_POSE,
    SLICE_RIGHT_HAND,
    pose_upper_indices,
)

TAG_RE = re.compile(r"(\d{4})_(\d{4})_(\d{4})")


def parse_tag(name: str) -> tuple[int, int, int] | None:
    m = TAG_RE.search(name.replace("\\", "/"))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def classify_member(name: str) -> str | None:
    """face | pose | hands | None."""
    p = name.replace("\\", "/").lower()
    if not p.endswith(".csv"):
        return None
    if parse_tag(p) is None:
        return None
    parts = p.split("/")
    joined = " ".join(parts)
    if "face" in joined or "rostro" in joined or "mesh" in joined:
        return "face"
    if "hand" in joined or "mano" in joined:
        return "hands"
    if "pose" in joined or "body" in joined or "cuerpo" in joined:
        return "pose"
    return "unknown"


def _xy_from_landmark_columns(df: pd.DataFrame) -> np.ndarray:
    xs = sorted(
        (c for c in df.columns if re.fullmatch(r"landmark_\d+_x", str(c).lower())
         or re.fullmatch(r"landmark_\d+_x", str(c))),
        key=lambda c: int(re.search(r"\d+", str(c)).group()),
    )
    # headers may keep original case
    if not xs:
        xs = sorted(
            (c for c in df.columns if re.search(r"landmark_\d+_x", str(c), re.I)),
            key=lambda c: int(re.search(r"\d+", str(c)).group()),
        )
    ys = []
    for xc in xs:
        yc = str(xc)[:-1] + "y" if str(xc).endswith("x") else str(xc).replace("_x", "_y")
        # try exact, then case-insensitive
        if yc in df.columns:
            ys.append(yc)
        else:
            match = next((c for c in df.columns if str(c).lower() == yc.lower()), None)
            if match is None:
                raise ValueError(f"no y column for {xc}")
            ys.append(match)
    x = df[xs].to_numpy(np.float32)
    y = df[ys].to_numpy(np.float32)
    return np.stack([x, y], axis=-1)  # (T, N, 2)


def csv_to_xy(raw: bytes) -> np.ndarray:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]
    if any(re.search(r"landmark_\d+_x", c, re.I) for c in df.columns):
        return _xy_from_landmark_columns(df)
    lower = {c.lower(): c for c in df.columns}
    if {"x", "y"}.issubset(lower) and "landmark_index" in lower:
        xcol, ycol = lower["x"], lower["y"]
        idx = lower["landmark_index"]
        frame_col = lower.get("frame")
        if frame_col:
            groups = list(df.groupby(df[frame_col], sort=True))
            n_lm = int(df[idx].max()) + 1
            out = np.full((len(groups), n_lm, 2), np.nan, np.float32)
            for i, (_, g) in enumerate(groups):
                out[i, g[idx].to_numpy(int), 0] = g[xcol].to_numpy(np.float32)
                out[i, g[idx].to_numpy(int), 1] = g[ycol].to_numpy(np.float32)
            return out
    numeric = df.select_dtypes(include=[np.number])
    arr = numeric.to_numpy(np.float32)
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        n = arr.shape[1] // 3
        xyz = arr.reshape(arr.shape[0], n, 3)
        return xyz[:, :, :2]
    if arr.ndim == 2 and arr.shape[1] % 2 == 0:
        n = arr.shape[1] // 2
        return arr.reshape(arr.shape[0], n, 2)
    raise ValueError(f"unrecognised CSV layout cols={list(df.columns)[:8]}")


def assemble_pose_hands(pose_xy: np.ndarray, hands_xy: np.ndarray,
                        pose_idx: list[int]) -> np.ndarray:
    """
    pose_xy: (T, 33, 2) typically
    hands_xy: (T, 42, 2) left then right, or (T, 21, 2) one hand
    -> (T, 53, 3)
    """
    t_p, t_h = len(pose_xy), len(hands_xy)
    t = min(t_p, t_h)
    pose_xy, hands_xy = pose_xy[:t], hands_xy[:t]
    out = np.zeros((t, N_LANDMARKS, 3), dtype=np.float32)

    n_h = hands_xy.shape[1]
    if n_h >= 42:
        left, right = hands_xy[:, :N_HAND], hands_xy[:, N_HAND:N_HAND * 2]
    elif n_h == 21:
        left, right = None, hands_xy[:, :N_HAND]
    else:
        raise ValueError(f"unexpected hand landmark count {n_h}")

    for block, sl in ((left, SLICE_LEFT_HAND), (right, SLICE_RIGHT_HAND)):
        if block is None:
            continue
        present = ~np.isnan(block).all(axis=(1, 2))
        near_zero = np.nan_to_num(np.abs(block)).sum(axis=(1, 2)) < 1e-6
        present = present & ~near_zero
        out[present, sl, :2] = np.nan_to_num(block[present])
        out[present, sl, 2] = 1.0

    if pose_xy.shape[1] < 33:
        raise ValueError(f"pose landmarks {pose_xy.shape[1]} < 33")
    pose = pose_xy[:, pose_idx]
    out[:, SLICE_POSE, :2] = np.nan_to_num(pose)
    out[:, SLICE_POSE, 2] = (~np.isnan(pose).any(axis=2)).astype(np.float32)
    return out


def normalise_vec(w: np.ndarray) -> np.ndarray:
    out = w.copy()
    lsh, rsh = out[:, ROW_LEFT_SHOULDER], out[:, ROW_RIGHT_SHOULDER]
    width = np.abs(lsh[:, 0] - rsh[:, 0])
    good = (lsh[:, 2] >= 0.5) & (rsh[:, 2] >= 0.5) & (width >= MIN_SHOULDER_WIDTH)
    if not good.any():
        return out
    origin = (lsh[good, :2] + rsh[good, :2]) / 2.0
    sub = out[good]
    det = sub[:, :, 2] > 0.5
    scaled = (sub[:, :, :2] - origin[:, None, :]) / width[good][:, None, None]
    sub[:, :, :2] = np.where(det[:, :, None], scaled, sub[:, :, :2])
    out[good] = sub
    return out


def to_window(pose_raw: bytes, hands_raw: bytes, pose_idx: list[int] | None = None):
    pose_idx = pose_idx if pose_idx is not None else pose_upper_indices()
    pose_xy = csv_to_xy(pose_raw)
    hands_xy = csv_to_xy(hands_raw)
    w = normalise_vec(assemble_pose_hands(pose_xy, hands_xy, pose_idx))
    if len(w) < 2:
        raise RuntimeError("too few frames")
    t = np.arange(len(w), dtype=np.float64) / max(len(w) - 1, 1)
    window = features.resample_by_time(t, w, config.WINDOW_FRAMES, 1.0)
    left = w[:, SLICE_LEFT_HAND, 2].mean(axis=1) > 0.5
    right = w[:, SLICE_RIGHT_HAND, 2].mean(axis=1) > 0.5
    hands_hit = float(np.mean(left | right))
    return {
        "window": window.astype(np.float32),
        "raw": w.astype(np.float32),
        "n_frames": int(len(w)),
        "hands_detected_frac": hands_hit,
        "motion_energy": float(features.motion_energy(window)),
    }


def rest_window_from_padding(raw: np.ndarray, frac=0.2) -> np.ndarray | None:
    n = len(raw)
    k = max(4, int(n * frac))
    for sl in (raw[:k], raw[-k:]):
        t = np.arange(len(sl), dtype=np.float64) / max(len(sl) - 1, 1)
        w = features.resample_by_time(t, sl, config.WINDOW_FRAMES, 1.0)
        if features.motion_energy(w) < 0.015:
            return w.astype(np.float32)
    return None


def sequence_id(sign: int, volunteer: int, rep: int) -> int:
    return sign * 100_000 + volunteer * 100 + rep


def rest_sequence_id(sid: int) -> int:
    """Offset must sit above every sign id (max ~4_900_403). 1e6 collided with sign 10."""
    return 10_000_000 + int(sid)
