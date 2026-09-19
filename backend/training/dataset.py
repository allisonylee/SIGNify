"""
Dataset + augmentation. Offline; does not ship.

SIGNER-INDEPENDENT SPLITS ARE NOT OPTIONAL. The 21 GISLR signers are held out
whole. A random split leaks signer identity: the model memorises a person's
style, validation accuracy comes out inflated, and the demo then fails on a new
face. The signer-independent number is the only one that predicts the demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config                                # noqa: E402
from backend.app.landmarks import (                           # noqa: E402
    POSE_UPPER_NAMES, SLICE_LEFT_HAND, SLICE_POSE, SLICE_RIGHT_HAND,
)

DATA = config.ROOT / "data" / "gislr"
FEAT = DATA / "features"
MINE_DIR = DATA / "mine"

# sequence_id namespaces, mirroring the participant_id ones below:
#   < 800,000,000   GISLR sequences (inside the packed array)
#   800,000,000+    signs WE recorded         -> MINE_DIR
#   900,000,000+    rest windows WE recorded  -> REST_DIR
MINE_ID_BASE = 800_000_000
REST_ID_BASE = 900_000_000


def source_dir(sequence_id):
    """Which directory holds this sample's .npy file."""
    sid = int(sequence_id)
    if sid >= REST_ID_BASE:
        return DATA / "rest"
    if sid >= MINE_ID_BASE:
        return MINE_DIR
    return FEAT


# Pose rows whose meaning mirrors when the image is flipped.
_MIRROR_PAIRS = [
    (POSE_UPPER_NAMES.index(a), POSE_UPPER_NAMES.index(b))
    for a, b in [("MOUTH_LEFT", "MOUTH_RIGHT"), ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
                 ("LEFT_ELBOW", "RIGHT_ELBOW"), ("LEFT_WRIST", "RIGHT_WRIST"),
                 ("LEFT_HIP", "RIGHT_HIP")]
]


def horizontal_flip(w: np.ndarray) -> np.ndarray:
    """
    Mirror, AND swap left/right hands and the paired pose joints.

    Flipping x without swapping the labels would teach the model that a
    left-handed signer's dominant hand is the right one. Handles left-handed
    signers; also the augmentation that partially masks a real handedness bug,
    which is why plan R9 is checked separately.
    """
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
    """Resample the window to simulate a faster/slower signer."""
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
    """Zero a hand and clear its mask -> one-handed-signing robustness."""
    out = w.copy()
    sl = SLICE_LEFT_HAND if side == "left" else SLICE_RIGHT_HAND
    out[:, sl, :2] = 0.0
    out[:, sl, 2] = 0.0
    return out


class GISLRDataset(Dataset):
    """
    Features are preloaded into one contiguous array by default.

    94k individual .npy reads per epoch would dominate training time; the whole
    set is only ~1.9 GB (94,477 x 32 x 53 x 3 x 4 B), so it fits in RAM and the
    disk is touched exactly once. Pass cache=False for the lazy path.
    """

    def __init__(self, rows: pd.DataFrame, label_map: dict[str, int], train=True,
                 seed=0, cache=True, shared=None):
        self.rows = rows.reset_index(drop=True)
        self.label_map = label_map
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.labels = np.array([label_map[s] for s in self.rows.sign], dtype=np.int64)
        if shared is not None:
            self.data = shared
        elif cache:
            self.data = load_packed(self.rows.sequence_id.to_numpy())
        else:
            self.data = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.data is not None:
            w = self.data[i].copy()
        else:
            sid = int(self.rows.sequence_id.iloc[i])
            w = np.load(source_dir(sid) / f"{sid}.npy").astype(np.float32)
        if self.train:
            g = self.rng
            if g.random() < 0.5:
                w = horizontal_flip(w)
            if g.random() < 0.5:
                w = time_warp(w, float(g.uniform(0.8, 1.2)))
            if g.random() < 0.5:
                w = affine(w, float(g.uniform(-10, 10)), float(g.uniform(0.9, 1.1)),
                           g.uniform(-0.05, 0.05, 2).astype(np.float32))
            if g.random() < 0.15:                      # one-handed robustness
                w = drop_hand(w, "left" if g.random() < 0.5 else "right")
            if g.random() < 0.25:                      # frame dropout
                k = g.integers(0, len(w), max(1, len(w) // 10))
                w[k] = w[np.clip(k - 1, 0, len(w) - 1)]
        return torch.from_numpy(np.ascontiguousarray(w)), int(self.labels[i])


def signer_independent_split(rows: pd.DataFrame, n_val_signers=4, seed=0):
    """Hold out whole signers. See the module docstring."""
    real = rows[rows.participant_id >= 0]
    rest = rows[rows.participant_id < 0]          # recorded rest, no signer
    signers = np.sort(real.participant_id.unique())
    rng = np.random.default_rng(seed)
    val = set(rng.choice(signers, size=min(n_val_signers, len(signers) - 1),
                         replace=False).tolist())
    tr = real[~real.participant_id.isin(val)]
    va = real[real.participant_id.isin(val)]
    if len(rest):
        # Rest windows all come from one recording session, so they cannot be
        # split by signer. Split them randomly at the same ratio instead, and
        # keep them out of the signer-independence claim.
        frac = len(va) / max(len(va) + len(tr), 1)
        shuf = rest.sample(frac=1.0, random_state=seed)
        k = int(len(shuf) * frac)
        va = pd.concat([va, shuf.iloc[:k]], ignore_index=True)
        tr = pd.concat([tr, shuf.iloc[k:]], ignore_index=True)
    return tr, va, sorted(val)


PACK = DATA / "features_all.npy"
IDS = DATA / "features_ids.npy"


def load_packed(sequence_ids: np.ndarray) -> np.ndarray:
    """
    Rows for `sequence_ids` from the consolidated pack (see consolidate.py).
    Falls back to per-file loads if the pack has not been built.
    """
    if not (PACK.exists() and IDS.exists()):
        first = np.load(source_dir(sequence_ids[0]) / f"{int(sequence_ids[0])}.npy")
        out = np.empty((len(sequence_ids), *first.shape), dtype=np.float32)
        for k, sid in enumerate(sequence_ids):
            out[k] = np.load(source_dir(sid) / f"{int(sid)}.npy")
        return out
    all_ids = np.load(IDS)
    pos = {int(v): i for i, v in enumerate(all_ids)}
    mm = np.load(PACK, mmap_mode="r")
    out = np.empty((len(sequence_ids), *mm.shape[1:]), dtype=np.float32)
    for k, sid in enumerate(sequence_ids):
        sid = int(sid)
        if sid in pos:
            out[k] = mm[pos[sid]]
        else:                       # something WE recorded, outside the pack
            out[k] = np.load(source_dir(sid) / f"{sid}.npy")
    return out


REST_LABEL = "__REST__"

# participant_id namespaces, so any row says where it came from:
#   2044 .. 62590   the 21 GISLR signers
#   -1              recorded rest windows (no signer)
#   700000+         US -- people NOT in the GISLR corpus
# GISLR's highest id is 62,590, so this cannot collide.
MINE_SIGNER_BASE = 700_000


def is_mine(participant_id) -> bool:
    """True for signers we recorded ourselves, i.e. not in GISLR."""
    return int(participant_id) >= MINE_SIGNER_BASE


def mine_signer_id(n: int) -> int:
    return MINE_SIGNER_BASE + int(n)
REST_DIR = DATA / "rest"


def load_rest_rows() -> pd.DataFrame:
    """Recorded rest windows, if any (scripts/v010_record_rest.py)."""
    idx = DATA / "rest_index.parquet"
    if not idx.exists():
        return pd.DataFrame(columns=["sequence_id", "sign", "participant_id"])
    df = pd.read_parquet(idx)
    have = {int(p.stem) for p in REST_DIR.glob("*.npy")}
    return df[df.sequence_id.isin(have)].reset_index(drop=True)


def load_rows() -> pd.DataFrame:
    """Index rows whose features actually exist on disk."""
    idx = pd.read_parquet(DATA / "subset_index.parquet")
    have = {int(p.stem) for p in FEAT.glob("*.npy")}
    return idx[idx.sequence_id.isin(have)].reset_index(drop=True)
