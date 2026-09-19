"""
Pitch x age -> voice_id, resolved from a table baked offline.

WHY A TABLE. ElevenLabs Voice Design takes ~7 s per call and its output is a
PERMANENT voice that consumes an account slot, so it cannot run when a slider
moves or when the app starts -- either would put seconds of API time in front of
the user. `backend/scripts/v022_design_voice_grid.py` does that work ahead of
time; this module only reads the result.

Cost at request time: one dict lookup. The translation and speech paths never
call the Voice Design API.
"""
from __future__ import annotations

import json
import pathlib

from . import config

GRID_PATH = pathlib.Path(__file__).with_name("voice_grid.json")

_GRID: dict[str, dict] = {}
_LOADED = False


def _load() -> dict[str, dict]:
    """Read the table once. A missing or broken file is not fatal -- the app
    falls back to the configured default voice and still speaks."""
    global _GRID, _LOADED
    if _LOADED:
        return _GRID
    _LOADED = True
    try:
        _GRID = json.loads(GRID_PATH.read_text()).get("cells", {})
    except FileNotFoundError:
        print(f"[voices] no {GRID_PATH.name}; using the default voice for every "
              f"slider position. Run backend/scripts/v022_design_voice_grid.py")
    except Exception as e:                               # noqa: BLE001
        print(f"[voices] {GRID_PATH.name} unreadable ({e}); using the default voice")
    return _GRID


def _clamp_step(v, lo=1, hi=5, default=3) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def resolve(pitch, age) -> tuple[str, str]:
    """
    -> (voice_id, how). `how` says where the id came from, so a demo that
    quietly fell back to the default voice is visible in the log instead of
    looking like the sliders did nothing.
    """
    grid = _load()
    p, a = _clamp_step(pitch), _clamp_step(age)
    cell = grid.get(f"{p},{a}")
    if cell and cell.get("voice_id"):
        return cell["voice_id"], f"grid[{p},{a}]"
    # Nearest generated cell beats the unrelated default voice: a grid that is
    # only partly baked should still move when the sliders move.
    if grid:
        near = min(grid.values(),
                   key=lambda c: (c["pitch"] - p) ** 2 + (c["age"] - a) ** 2)
        return near["voice_id"], f"nearest[{near['pitch']},{near['age']}] for [{p},{a}]"
    return config.ELEVENLABS_VOICE_ID, "default (grid empty)"


def available() -> int:
    return len(_load())
