"""
v011 -- FrameBuffer.segment() must stay correct across deque eviction.

The buffer is bounded, so local indices shift left once it is full. A segment
spanning an eviction would silently read the WRONG frames -- no exception, just
a wrong answer. Verify absolute indexing handles it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.inference import FrameBuffer
from backend.app.ingest import LandmarkFrame
from backend.app.landmarks import N_HAND, N_POSE_UPPER

ok = []
def check(n, c, d=""):
    ok.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}   {d}")

from backend.app.landmarks import POSE_UPPER_NAMES

L_SH = POSE_UPPER_NAMES.index("LEFT_SHOULDER")
R_SH = POSE_UPPER_NAMES.index("RIGHT_SHOULDER")
ORIGIN_X, HALF_W = 0.5, 0.1          # shoulders at 0.6 / 0.4 -> width 0.2


def mk(i):
    """Frame whose left-hand x encodes i, so frames are distinguishable."""
    pose = np.tile([[ORIGIN_X, 0.3]], (N_POSE_UPPER, 1)).astype(np.float32)
    pose[L_SH] = [ORIGIN_X + HALF_W, 0.3]
    pose[R_SH] = [ORIGIN_X - HALF_W, 0.3]
    return LandmarkFrame(
        t=i * 0.05,
        hands={"left": np.full((N_HAND, 2), float(i), np.float32), "right": None},
        pose=pose,
    )


def decode(x_norm):
    """Undo normalise(): x_norm = (i - origin) / width."""
    return x_norm * (2 * HALF_W) + ORIGIN_X

MAX = 32
buf = FrameBuffer(span_s=1.5, n_out=8, max_frames=MAX)
for i in range(20):
    buf.add(mk(i))
check("total counts every add", buf.total == 20, f"total={buf.total} len={len(buf)}")

seg = buf.segment(10)
check("segment before rollover works", seg.shape == (8, 53, 3), str(seg.shape))

for i in range(20, 60):          # force eviction: 60 adds into a 32 buffer
    buf.add(mk(i))
check("deque is full", len(buf) == MAX, f"len={len(buf)} total={buf.total}")
check("total kept growing", buf.total == 60, f"total={buf.total}")

seg = buf.segment(50)            # absolute index, still resident
check("segment after rollover works", seg.shape == (8, 53, 3), str(seg.shape))

# frames 0..27 were evicted; asking for an evicted index must clamp, not crash
seg_old = buf.segment(5)
check("evicted start clamps, no crash", seg_old.shape == (8, 53, 3), str(seg_old.shape))

# the decisive one: does an absolute index return the RIGHT frames?
buf2 = FrameBuffer(span_s=1.5, n_out=4, max_frames=16)
for i in range(40):
    buf2.add(mk(i))
s = buf2.segment(36)             # frames 36..39
vals = np.unique(np.round(decode(s[:, 0, 0])).astype(int))
check("absolute index selects the right frames", vals.min() >= 36 and vals.max() <= 39,
      f"encoded frame ids in segment: {vals.tolist()}")

s_wrong = buf2.segment(24)       # older but still resident
v2 = np.unique(np.round(decode(s_wrong[:, 0, 0])).astype(int))
check("older absolute index reaches further back", v2.min() < vals.min(),
      f"ids {v2.min()}..{v2.max()} vs {vals.min()}..{vals.max()}")

print(f"\n{'='*58}\n  {sum(ok)}/{len(ok)} checks passed\n{'='*58}")
sys.exit(0 if all(ok) else 1)
