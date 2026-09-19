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
        # Absolute count of frames ever added. The deque is bounded, so local
        # indices shift left as it rolls; callers hold ABSOLUTE indices and
        # segment() converts. Without this, a segment that spans an eviction
        # silently reads the wrong frames.
        self.total = 0

    def add(self, frame: LandmarkFrame) -> None:
        # Network reordering: a late frame would break np.interp's requirement
        # that x be increasing, so drop it rather than corrupt the window.
        if self._t and frame.t < self._t[-1]:
            self.dropped_out_of_order += 1
            return
        self._t.append(frame.t)
        self._f.append(features.normalise(features.assemble(frame.hands, frame.pose)))
        self.total += 1

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

    def segment(self, start_abs: int) -> np.ndarray:
        """
        Resample frames from ABSOLUTE index `start_abs` onward to n_out, over
        THEIR OWN duration.

        Training resampled each isolated sign across its full length, so a live
        segment must be normalised the same way -- not over a fixed wall-clock
        span. Getting this wrong makes the model see signs at the wrong speed.
        """
        base = self.total - len(self._t)          # absolute index of _t[0]
        start = max(0, start_abs - base)          # clamp if it was evicted
        t = np.array(list(self._t)[start:])
        f = np.stack(list(self._f)[start:])
        span = max(t[-1] - t[0], 1e-3)
        return features.resample_by_time((t - t[0]) / span, f, self.n_out, 1.0)


class Recognizer:
    def infer(self, window: np.ndarray) -> Recognition | None:
        raise NotImplementedError


class StubRecognizer(Recognizer):
    """
    STAGE 1 PLACEHOLDER, kept for tests and for running the pipe with no model.
    Emits a fixed gloss every `period_s`. Does no recognition.
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
        energy = features.motion_energy(window)
        return Recognition(
            gloss=[self.gloss],
            confidence=round(min(0.99, 0.5 + energy * 10), 3),
            window_ms=0.0,
            infer_ms=(time.perf_counter() - t0) * 1000,
        )


class SegmentState:
    """Where we are in the sign/rest cycle."""
    IDLE = "idle"
    SIGNING = "signing"


class ModelRecognizer(Recognizer):
    """
    STAGE 3. Wraps the trained encoder with the machinery that turns an
    ISOLATED-SIGN CLASSIFIER into something usable on a CONTINUOUS stream.

    The model only ever answers "which sign is this 32-frame window". It has no
    concept of where one sign ends and the next begins, and it has never seen
    "nothing is happening" -- so without the guards below it classifies idle
    hands several times a second, confidently and wrongly.

    Four guards, in order of how much they matter:

      1. motion gating      classify only between a motion onset and the
                            stillness that follows it. Also the utterance
                            boundary for the speech path (plan section 8).
      2. confidence floor   drop anything under `min_confidence`. Without a
                            rest class this is the main thing standing between
                            you and constant spurious output.
      3. margin check       require the top class to beat the runner-up by
                            `min_margin`; a flat distribution means "unsure",
                            which is different from "confidently wrong".
      4. debounce           suppress the same gloss repeating within
                            `debounce_s`, so one held sign fires once.

    A trained REST class would subsume 2 and 3 and is the principled fix; it
    needs recorded idle footage (see scripts/v010_record_rest.py).
    """

    def __init__(self, model, labels, device, lang="ase",
                 motion_threshold=None, min_sign_frames=8, quiet_seconds=None,
                 max_segment_s=None, min_confidence=None, min_margin=None,
                 debounce_s=None, rest_label="__REST__", allowed=None):
        motion_threshold = (config.REC_MOTION_THRESHOLD
                            if motion_threshold is None else motion_threshold)
        min_confidence = (config.REC_MIN_CONFIDENCE
                          if min_confidence is None else min_confidence)
        min_margin = config.REC_MIN_MARGIN if min_margin is None else min_margin
        debounce_s = config.REC_DEBOUNCE_S if debounce_s is None else debounce_s
        quiet_seconds = (config.REC_QUIET_SECONDS
                         if quiet_seconds is None else quiet_seconds)
        max_segment_s = (config.REC_MAX_SEGMENT_S
                         if max_segment_s is None else max_segment_s)
        self.model, self.labels, self.device, self.lang = model, labels, device, lang
        self.motion_threshold = motion_threshold
        self.min_sign_frames = min_sign_frames
        # SECONDS, not frames. A frame count silently changes the required
        # pause with frame rate -- 5 frames is 0.50 s at 10 fps but 0.17 s at
        # 30 fps -- so the same signing works on one machine and merges signs
        # on another.
        self.quiet_seconds = quiet_seconds
        # Hard ceiling on one segment. Real signs are short; continuous motion
        # past this means the pause was missed and two signs are merging, so
        # cut rather than emit one wrong prediction for both.
        self.max_segment_s = max_segment_s
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.debounce_s = debounce_s
        self.rest_label = rest_label
        self.rest_idx = labels.index(rest_label) if rest_label in labels else None
        if config.REC_IGNORE_REST:
            print("[rec] SIGN_IGNORE_REST=1 -- rest veto DISABLED (test only)")
            self.rest_idx = None
        self.allowed = allowed

        self.state = SegmentState.IDLE
        self._seg_start = 0
        self._seg_start_t = 0.0
        self._quiet_since = None
        self._last_gloss = None
        self._last_emit_t = -1e9
        self.last_energy = 0.0
        self.last_scores: list[tuple[str, float]] = []
        self.rejected = {"confidence": 0, "margin": 0, "rest": 0,
                         "debounce": 0, "too_short": 0}

    # -- scoring ---------------------------------------------------------
    def score(self, window: np.ndarray) -> list[tuple[str, float]]:
        import torch
        x = torch.from_numpy(window[None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(x, self.lang)[0]
        if self.allowed is not None:
            mask = torch.full_like(logits, float("-inf"))
            mask[self.allowed] = 0.0
            logits = logits + mask
        p = torch.softmax(logits, -1).cpu().numpy()
        order = np.argsort(p)[::-1]
        return [(self.labels[i], float(p[i])) for i in order[:5]]

    # -- continuous stream ------------------------------------------------
    def observe(self, buf, now: float) -> Recognition | None:
        """
        Feed the buffer after every frame. Returns a Recognition only at a
        completed, accepted utterance boundary; None the rest of the time.
        """
        if len(buf) < 3:
            return None
        recent = np.stack(list(buf._f)[-3:])
        self.last_energy = features.motion_energy(recent)
        moving = self.last_energy > self.motion_threshold

        if self.state == SegmentState.IDLE:
            if moving:
                self.state = SegmentState.SIGNING
                self._seg_start = buf.total - 1        # ABSOLUTE index
                self._seg_start_t = now
                self._quiet_since = None
            return None

        # SIGNING
        too_long = (now - self._seg_start_t) >= self.max_segment_s
        if moving and not too_long:
            self._quiet_since = None
            return None
        if not too_long:
            if self._quiet_since is None:
                self._quiet_since = now
            if (now - self._quiet_since) < self.quiet_seconds:
                return None

        self.state = SegmentState.IDLE
        n = buf.total - self._seg_start
        if n < self.min_sign_frames:
            self.rejected["too_short"] += 1
            return None

        t0 = time.perf_counter()
        window = buf.segment(self._seg_start)
        window_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        scores = self.score(window)
        infer_ms = (time.perf_counter() - t1) * 1000
        self.last_scores = scores

        (top, p), (_, p2) = scores[0], scores[1]
        if self.rest_idx is not None and top == self.rest_label:
            self.rejected["rest"] += 1
            return None
        if p < self.min_confidence:
            self.rejected["confidence"] += 1
            return None
        if p - p2 < self.min_margin:
            self.rejected["margin"] += 1
            return None
        if top == self._last_gloss and (now - self._last_emit_t) < self.debounce_s:
            self.rejected["debounce"] += 1
            return None

        self._last_gloss, self._last_emit_t = top, now
        return Recognition(gloss=[top], confidence=round(p, 3),
                           window_ms=window_ms, infer_ms=infer_ms)

    def infer(self, window: np.ndarray) -> Recognition | None:
        """One-shot scoring of a supplied window; bypasses the state machine."""
        t0 = time.perf_counter()
        scores = self.score(window)
        self.last_scores = scores
        top, p = scores[0]
        if top == self.rest_label or p < self.min_confidence:
            return None
        return Recognition(gloss=[top], confidence=round(p, 3), window_ms=0.0,
                           infer_ms=(time.perf_counter() - t0) * 1000)


class UtteranceBuffer:
    """
    Accumulates committed glosses into an utterance, and decides when it ended.

    THE RECOGNIZER EMITS ONE GLOSS AT A TIME. Without this, every sign is
    independently turned into text and spoken on its own -- "Mother." "Happy."
    "Visit." -- and the LLM is never given a sequence to make grammatical,
    because a one-item list always hits the bypass.

    Endpointing is by TIMEOUT, the same idea speech recognisers use: a gloss
    arrives, the clock resets; when no new gloss has arrived for
    `timeout_s`, the utterance is finished and goes to the LLM.

    The deliberate trade: speech now waits `timeout_s` after your last sign.
    Text does NOT -- partials are emitted per gloss and appear instantly, so
    the screen keeps up while only the audio waits (plan section 8.2).

    Two safety valves stop it accumulating forever if the timeout never trips:
    `max_glosses` and `max_duration_s`.
    """

    def __init__(self, timeout_s=None, max_glosses=12, max_duration_s=15.0):
        self.timeout_s = (config.UTTERANCE_TIMEOUT_S
                          if timeout_s is None else timeout_s)
        self.max_glosses = max_glosses
        self.max_duration_s = max_duration_s
        self.glosses: list[str] = []
        self.start_t = 0.0
        self.last_t = 0.0

    def __len__(self):
        return len(self.glosses)

    def add(self, gloss: str, now: float) -> None:
        if not self.glosses:
            self.start_t = now
        self.glosses.append(gloss)
        self.last_t = now

    def flush_reason(self, now: float) -> str | None:
        """Why the utterance should end now, or None to keep waiting."""
        if not self.glosses:
            return None
        if now - self.last_t >= self.timeout_s:
            return "timeout"
        if len(self.glosses) >= self.max_glosses:
            return "max_glosses"
        if now - self.start_t >= self.max_duration_s:
            return "max_duration"
        return None

    def take(self) -> list[str]:
        out, self.glosses = self.glosses, []
        return out

    def seconds_until_flush(self, now: float) -> float:
        if not self.glosses:
            return float("inf")
        return max(0.0, self.timeout_s - (now - self.last_t))


def load_model_recognizer(device=None, **kw):
    """Build a ModelRecognizer from the trained checkpoint."""
    import torch
    from . import config
    from .model import SignClassifier

    ck_path = config.MODELS_DIR / "encoder_ase.pt"
    if not ck_path.exists():
        return None
    dev = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = SignClassifier({"ase": len(ck["labels"])})
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    return ModelRecognizer(m, ck["labels"], dev, **kw)


def glosses_to_text(gloss: list[str]) -> str:
    """
    STAGE 1 PLACEHOLDER for the Gemini layer (Stage 4).
    Real version prompts an LLM to turn glosses into a fluent sentence in the
    target spoken language.
    """
    return " ".join(g.capitalize() for g in gloss) + "."
