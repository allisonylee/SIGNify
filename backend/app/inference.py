"""
Frame buffering and recognition.

Stage 1 ships StubRecognizer, which emits a fixed gloss on a timer. It exists
so the rest of the pipeline can be built and measured before a model exists.
Stage 2 supplies the trained encoder, Stage 3 replaces the timer with real
motion gating -- both behind the Recognizer interface, so main.py never changes.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config, features
from .ingest import LandmarkFrame


@dataclass
class Recognition:
    gloss: list[str]
    confidence: float
    window_ms: float          # time to build the window
    infer_ms: float           # time in the model


class FrameBuffer:
    """
    Rolling buffer of raw frames, resampled to a fixed window ON TIMESTAMPS.

    Holds ~2x the window span so a variable frame rate still fills the window.
    """

    def __init__(self, span_s=None, n_out=None, max_frames=256):
        self.span_s = span_s if span_s is not None else config.WINDOW_SECONDS
        self.n_out = n_out if n_out is not None else config.WINDOW_FRAMES
        self._t: deque[float] = deque(maxlen=max_frames)
        self._f: deque[np.ndarray] = deque(maxlen=max_frames)
        self.dropped_out_of_order = 0

    def add(self, frame: LandmarkFrame) -> None:
        # Network reordering: a late frame would break np.interp's requirement
        # that x be increasing, so drop it rather than corrupt the window.
        if self._t and frame.t < self._t[-1]:
            self.dropped_out_of_order += 1
            return
        self._t.append(frame.t)
        self._f.append(features.normalise(features.assemble(frame.hands, frame.pose)))

    def __len__(self):
        return len(self._t)

    def span(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    def ready(self) -> bool:
        return len(self._t) >= 2 and self.span() >= self.span_s * 0.5

    def window(self) -> np.ndarray:
        """-> (n_out, 53, 3), interpolated over the last span_s seconds."""
        return features.resample_by_time(
            np.array(self._t), np.stack(self._f), self.n_out, self.span_s)


class Recognizer:
    def infer(self, window: np.ndarray) -> Recognition | None:
        raise NotImplementedError


class StubRecognizer(Recognizer):
    """
    STAGE 1 PLACEHOLDER. Emits a fixed gloss every `period_s`.

    Does no recognition whatsoever. Reports real window/motion numbers so the
    latency instrumentation is exercised with honest values.
    """

    def __init__(self, gloss="HELLO", period_s=2.0):
        self.gloss, self.period_s = gloss, period_s
        self._last = 0.0

    def infer(self, window: np.ndarray) -> Recognition | None:
        now = time.perf_counter()
        if now - self._last < self.period_s:
            return None
        self._last = now
        t0 = time.perf_counter()
        energy = features.motion_energy(window)          # real number, unused
        return Recognition(
            gloss=[self.gloss],
            confidence=round(min(0.99, 0.5 + energy * 10), 3),
            window_ms=0.0,
            infer_ms=(time.perf_counter() - t0) * 1000,
        )


def glosses_to_text(gloss: list[str]) -> str:
    """
    STAGE 1 PLACEHOLDER for the Gemini layer (Stage 4).
    Real version prompts an LLM to turn glosses into a fluent sentence in the
    target spoken language.
    """
    return " ".join(g.capitalize() for g in gloss) + "."
