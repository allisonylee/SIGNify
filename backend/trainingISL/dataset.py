"""INCLUDE-50 dataset + the same augmentations GISLR uses. Local copy (do not edit training/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.landmarks import (  # noqa: E402
    POSE_UPPER_NAMES, SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)
from backend.trainingISL.paths import FEAT, INDEX, LABELS, REST_SIGN  # noqa: E402

_MIRROR_PAIRS = [
    (POSE_UPPER_NAMES.index(a), POSE_UPPER_NAMES.index(b))
    for a, b in [("MOUTH_LEFT", "MOUTH_RIGHT"), ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
                 ("LEFT_ELBOW", "RIGHT_ELBOW"), ("LEFT_WRIST", "RIGHT_WRIST"),
                 ("LEFT_HIP", "RIGHT_HIP")]
]


def horizontal_flip(w: np.ndarray) -> np.ndarray:
    out = w.copy()
    out[..., 0] *= -1
    lh = out[:, SLICE_LEFT_HAND].copy()
    out[:, SLICE_LEFT_HAND] = out[:, SLICE_RIGHT_HAND]
    out[:, SLICE_RIGHT_HAND] = lh
    p0 = SLICE_POSE.start
    for a, b in _MIRROR_PAIRS:
        tmp = out[:, p0 + a].copy()
        out[:, p0 + a] = out[:, p0 + b]
        out[:, p0 + b] = tmp
    return out


def time_warp(w: np.ndarray, factor: float) -> np.ndarray:
    t = np.arange(len(w))
    new = np.clip(np.linspace(0, (len(w) - 1) * factor, len(w)), 0, len(w) - 1)
    out = np.empty_like(w)
    flat = w.reshape(len(w), -1)
    of = out.reshape(len(w), -1)
    for c in range(flat.shape[1]):
        of[:, c] = np.interp(new, t, flat[:, c])
    out[..., 2] = (out[..., 2] >= 0.5).astype(np.float32)
    return out


def affine(w: np.ndarray, deg: float, scale: float, shift: np.ndarray) -> np.ndarray:
    out = w.copy()
    r = np.deg2rad(deg)
    R = np.array([[np.cos(r), -np.sin(r)],
                  [np.sin(r),  np.cos(r)]], dtype=np.float32)
    m = out[..., 2] > 0.5
    out[..., :2][m] = (out[..., :2][m] @ R.T) * scale + shift
    return out


def drop_hand(w: np.ndarray, side: str) -> np.ndarray:
    out = w.copy()
    sl = SLICE_LEFT_HAND if side == "left" else SLICE_RIGHT_HAND
    out[:, sl, :2] = 0.0
    out[:, sl, 2] = 0.0
    return out


class IsolatedSignDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, label_map: dict[str, int], train=True, seed=0):
        self.rows = rows.reset_index(drop=True)
        self.label_map = label_map
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.labels = np.array([label_map[s] for s in self.rows.sign], dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        sid = int(self.rows.sequence_id.iloc[i])
        w = np.load(FEAT / f"{sid}.npy").astype(np.float32)
        if self.train:
            g = self.rng
            if g.random() < 0.5:
                w = horizontal_flip(w)
            if g.random() < 0.5:
                w = time_warp(w, float(g.uniform(0.8, 1.2)))
            if g.random() < 0.5:
                w = affine(w, float(g.uniform(-10, 10)), float(g.uniform(0.9, 1.1)),
                           g.uniform(-0.05, 0.05, 2).astype(np.float32))
            if g.random() < 0.15:
                w = drop_hand(w, "left" if g.random() < 0.5 else "right")
            if g.random() < 0.25:
                k = g.integers(0, len(w), max(1, len(w) // 10))
                w[k] = w[np.clip(k - 1, 0, len(w) - 1)]
        return torch.from_numpy(np.ascontiguousarray(w)), int(self.labels[i])


def signer_independent_split(rows: pd.DataFrame, n_val_signers=2, seed=0):
    signers = np.sort(rows.participant_id.unique())
    rng = np.random.default_rng(seed)
    n_hold = min(n_val_signers, max(len(signers) - 1, 1))
    val = set(rng.choice(signers, size=n_hold, replace=False).tolist())
    tr = rows[~rows.participant_id.isin(val)]
    va = rows[rows.participant_id.isin(val)]
    return tr.reset_index(drop=True), va.reset_index(drop=True), sorted(val)


def load_rows() -> pd.DataFrame:
    from backend.trainingISL.download_include import (
        build_index, fetch_metadata, merge_rest_into_index,
    )
    index = merge_rest_into_index(build_index(fetch_metadata()))
    have = {int(p.stem) for p in FEAT.glob("*.npy")}
    keep = index[index.sequence_id.isin(have)].reset_index(drop=True)
    keep.to_parquet(INDEX)
    return keep


def load_label_list() -> list[str]:
    if LABELS.exists():
        return json.loads(LABELS.read_text(encoding="utf-8"))
    rows = load_rows()
    signs = sorted(s for s in rows.sign.unique() if s != REST_SIGN)
    return signs + [REST_SIGN]
