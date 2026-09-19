"""
The landmark specification, and the MediaPipe wrapper that produces it.

THIS IS THE SINGLE SOURCE OF TRUTH for which landmarks we use and in what
order. Both app/ (inference) and training/ import from here. Never hardcode
an index anywhere else.

Verified by backend/scripts/v003_stage0_holistic_check.py:
  MediaPipe 0.10.35, HolisticLandmarker, CPU delegate.
  Blocks: face 478 (UNUSED), pose 33, left_hand 21, right_hand 21.
  Coordinates are normalised to [0,1] of the source frame.

DO NOT upgrade mediapipe past 0.10.35 -- 1.0.x aborts on macOS arm64.
See backend/requirements.txt and backend/outputs/stage0_api_matrix.txt.
"""
from __future__ import annotations

import numpy as np

# 11 upper-body pose landmarks, by NAME. Indices come from
# mediapipe.tasks.python.vision.PoseLandmark, resolved at import time so a
# reordering upstream cannot silently shift us. (Plan risk R1.)
POSE_UPPER_NAMES = (
    "NOSE", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
)

N_HAND = 21
N_POSE_UPPER = len(POSE_UPPER_NAMES)
N_LANDMARKS = N_HAND * 2 + N_POSE_UPPER          # 53
N_CHANNELS = 3                                    # x, y, detected-mask
N_FEATURES = N_LANDMARKS * N_CHANNELS             # 159

# Canonical ordering of the 53 rows. Anything that builds or reads a feature
# vector must use these slices.
SLICE_LEFT_HAND = slice(0, N_HAND)                # 0..20
SLICE_RIGHT_HAND = slice(N_HAND, N_HAND * 2)      # 21..41
SLICE_POSE = slice(N_HAND * 2, N_LANDMARKS)       # 42..52

# Row offsets within SLICE_POSE, for normalisation.
_POSE_IDX = {name: i for i, name in enumerate(POSE_UPPER_NAMES)}
ROW_LEFT_SHOULDER = SLICE_POSE.start + _POSE_IDX["LEFT_SHOULDER"]
ROW_RIGHT_SHOULDER = SLICE_POSE.start + _POSE_IDX["RIGHT_SHOULDER"]
ROW_NOSE = SLICE_POSE.start + _POSE_IDX["NOSE"]
ROW_LEFT_WRIST = SLICE_POSE.start + _POSE_IDX["LEFT_WRIST"]
ROW_RIGHT_WRIST = SLICE_POSE.start + _POSE_IDX["RIGHT_WRIST"]


def pose_upper_indices() -> list[int]:
    """MediaPipe pose indices for POSE_UPPER_NAMES, resolved by name."""
    from mediapipe.tasks.python.vision import PoseLandmark
    return [int(getattr(PoseLandmark, n)) for n in POSE_UPPER_NAMES]


class HolisticExtractor:
    """
    Wraps MediaPipe HolisticLandmarker in VIDEO mode.

    VIDEO mode (not IMAGE) because it applies temporal tracking, which is what
    a live stream wants. Timestamps must be non-decreasing integers in ms.
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
        "holistic_landmarker/float16/latest/holistic_landmarker.task"
    )

    def __init__(self, model_path=None):
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        from .config import MODELS_DIR

        self._mp = mp
        path = model_path or (MODELS_DIR / "holistic_landmarker.task")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.MODEL_URL, path)

        self._landmarker = vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
            )
        )
        self._pose_idx = pose_upper_indices()
        self._last_ts = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._landmarker.close()

    def extract(self, rgb: np.ndarray, timestamp_ms: int):
        """
        rgb: HxWx3 uint8, contiguous. Returns (hands, pose) where hands is
        {'left': (21,2)|None, 'right': (21,2)|None} and pose is (11,2).
        Coordinates stay in MediaPipe's [0,1] frame-normalised space.
        """
        # MediaPipe requires strictly increasing timestamps in VIDEO mode.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        res = self._landmarker.detect_for_video(img, timestamp_ms)

        def xy(block):
            if not block:
                return None
            return np.array([[p.x, p.y] for p in block], dtype=np.float32)

        pose_all = res.pose_landmarks
        if pose_all:
            pose = np.array(
                [[pose_all[i].x, pose_all[i].y] for i in self._pose_idx],
                dtype=np.float32,
            )
        else:
            pose = np.full((N_POSE_UPPER, 2), np.nan, dtype=np.float32)

        return {"left": xy(res.left_hand_landmarks),
                "right": xy(res.right_hand_landmarks)}, pose
