"""
v006 -- sanity-check features produced from REAL GISLR parquet.

A clean exit from download.py only proves it ran. This looks at the numbers:
shapes, ranges, NaNs, detection rates, and whether normalisation actually
anchored the shoulders where it should.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import config
from backend.app.landmarks import (
    ROW_LEFT_SHOULDER, ROW_RIGHT_SHOULDER, SLICE_LEFT_HAND, SLICE_POSE,
    SLICE_RIGHT_HAND,
)

FEAT = config.ROOT / "data" / "gislr" / "features"
idx = pd.read_parquet(config.ROOT / "data" / "gislr" / "subset_index.parquet")
import random
_all = sorted(FEAT.glob("*.npy"))
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 400
files = _all if len(_all) <= SAMPLE else random.Random(0).sample(_all, SAMPLE)
print(f"{len(_all):,} feature files; sanity-checking a random {len(files)}\n")

ok = []
def check(n, c, d=""):
    ok.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}   {d}")

arrs = {int(p.stem): np.load(p) for p in files}
meta = idx.set_index("sequence_id")

print("[1] shape and finiteness")
shapes = {a.shape for a in arrs.values()}
check("all (32,53,3)", shapes == {(config.WINDOW_FRAMES, 53, 3)}, str(shapes))
check("no NaN", not any(np.isnan(a).any() for a in arrs.values()))
check("no Inf", not any(np.isinf(a).any() for a in arrs.values()))

print("\n[2] mask channel is strictly binary")
vals = set()
for a in arrs.values():
    vals |= set(np.unique(a[..., 2]).tolist())
check("mask in {0,1}", vals <= {0.0, 1.0}, str(sorted(vals)))

print("\n[3] normalisation anchored the shoulders")
lx, rx = [], []
for a in arrs.values():
    m = a[:, ROW_LEFT_SHOULDER, 2] > 0.5
    if m.any():
        lx += a[m, ROW_LEFT_SHOULDER, 0].tolist()
        rx += a[m, ROW_RIGHT_SHOULDER, 0].tolist()
lx, rx = np.array(lx), np.array(rx)
check("LEFT_SHOULDER.x ~ +0.5", abs(lx.mean() - 0.5) < 0.05, f"mean {lx.mean():+.4f}")
check("RIGHT_SHOULDER.x ~ -0.5", abs(rx.mean() + 0.5) < 0.05, f"mean {rx.mean():+.4f}")
check("LEFT > RIGHT (un-mirrored)", lx.mean() > rx.mean(),
      "signer's left appears at larger x")

print("\n[4] coordinate ranges (shoulder-width units)")
allxy = np.concatenate([a[a[..., 2] > 0.5][:, :2] for a in arrs.values()])
p1, p99 = np.percentile(allxy, [1, 99], axis=0)
print(f"      x  1st/99th pct  {p1[0]:+.2f} / {p99[0]:+.2f}")
print(f"      y  1st/99th pct  {p1[1]:+.2f} / {p99[1]:+.2f}")
check("x within +-5 shoulder widths", abs(p1[0]) < 5 and abs(p99[0]) < 5)
check("y within +-5 shoulder widths", abs(p1[1]) < 5 and abs(p99[1]) < 5)

print("\n[5] detection rates (GISLR legitimately has missing hands)")
for nm, sl in [("left hand", SLICE_LEFT_HAND), ("right hand", SLICE_RIGHT_HAND),
               ("pose", SLICE_POSE)]:
    r = np.mean([a[:, sl, 2].mean() for a in arrs.values()])
    print(f"      {nm:11s} detected in {r:6.1%} of frames")
pose_rate = np.mean([a[:, SLICE_POSE, 2].mean() for a in arrs.values()])
hand_rate = np.mean([max(a[:, SLICE_LEFT_HAND, 2].mean(),
                         a[:, SLICE_RIGHT_HAND, 2].mean()) for a in arrs.values()])
check("pose almost always present", pose_rate > 0.9, f"{pose_rate:.1%}")
check("at least one hand usually present", hand_rate > 0.5, f"{hand_rate:.1%}")

print("\n[6] different sequences are actually different")
ids = list(arrs)
d = [float(np.abs(arrs[ids[i]] - arrs[ids[j]]).mean())
     for i in range(len(ids)) for j in range(i + 1, len(ids))]
check("no two files identical", min(d) > 1e-6, f"min mean|diff| = {min(d):.4f}")

print("\n[7] per-sequence detail (first 8)")
for sid, a in list(arrs.items())[:8]:
    sign = meta.loc[sid, "sign"] if sid in meta.index else "?"
    pid = meta.loc[sid, "participant_id"] if sid in meta.index else "?"
    print(f"      {sid:<12} sign={str(sign):<12} signer={pid}  "
          f"lh={a[:,SLICE_LEFT_HAND,2].mean():4.0%} "
          f"rh={a[:,SLICE_RIGHT_HAND,2].mean():4.0%} "
          f"|xy|max={np.abs(a[...,:2]).max():5.2f}")

print(f"\n{'='*62}\n  {sum(ok)}/{len(ok)} checks passed\n{'='*62}")
sys.exit(0 if all(ok) else 1)
