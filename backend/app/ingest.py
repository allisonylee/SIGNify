"""
The adapter boundary between transport and everything downstream.

Mode A: the browser runs MediaPipe and sends landmarks (~106 floats/frame).
Mode B: the client sends JPEG frames and WE run MediaPipe here.

Both produce a LandmarkFrame. NOTHING downstream of this module may know which
mode produced the data -- that is what makes mode A a drop-in later rather
than a rewrite. See plan section 14.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

import numpy as np

from .landmarks import N_HAND, N_POSE_UPPER


@dataclass
class LandmarkFrame:
    """One frame of landmarks, in the canonical convention."""
    t: float                                    # seconds, client clock
    hands: dict[str, np.ndarray | None]         # 'left'/'right' -> (21,2)
    pose: np.ndarray                            # (11,2)
    seq: int = -1
    source: str = "?"                           # 'landmarks' | 'frames'
    extract_ms: float = 0.0                     # 0 in mode A


class IngestError(ValueError):
    pass


def _to_xy(v, n, what):
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32)
    if a.shape != (n, 2):
        raise IngestError(f"{what}: expected ({n},2), got {a.shape}")
    return a


def ingest_landmarks(msg: dict) -> LandmarkFrame:
    """Mode A. Parse and validate landmarks the client already computed."""
    try:
        t = float(msg["t"])
    except (KeyError, TypeError, ValueError) as e:
        raise IngestError(f"missing/!bad t: {e}") from e

    hands_in = msg.get("hands") or {}
    hands = {k: _to_xy(hands_in.get(k), N_HAND, f"hands.{k}")
             for k in ("left", "right")}
    pose = _to_xy(msg.get("pose"), N_POSE_UPPER, "pose")
    if pose is None:
        pose = np.full((N_POSE_UPPER, 2), np.nan, dtype=np.float32)

    return LandmarkFrame(t=t, hands=hands, pose=pose,
                         seq=int(msg.get("seq", -1)), source="landmarks")


def ingest_frames(msg: dict, extractor) -> LandmarkFrame:
    """Mode B. Decode a JPEG and run MediaPipe here."""
    import time
    import cv2

    try:
        t = float(msg["t"])
        raw = base64.b64decode(msg["jpeg"])
    except (KeyError, TypeError, ValueError) as e:
        raise IngestError(f"bad frame message: {e}") from e

    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IngestError("jpeg failed to decode")

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hands, pose = extractor.extract(rgb, int(t * 1000))
    dt = (time.perf_counter() - t0) * 1000

    return LandmarkFrame(t=t, hands=hands, pose=pose,
                         seq=int(msg.get("seq", -1)), source="frames",
                         extract_ms=dt)


def ingest(msg: dict, extractor=None) -> LandmarkFrame:
    """Dispatch on message type. The only place mode is decided."""
    kind = msg.get("type")
    if kind == "landmarks":
        return ingest_landmarks(msg)
    if kind == "frame":
        if extractor is None:
            raise IngestError("mode B message but no extractor available")
        return ingest_frames(msg, extractor)
    raise IngestError(f"not an ingest message: {kind!r}")
