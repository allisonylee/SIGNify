"""
v007 -- prove the vectorised converter equals the per-frame one.

A silent numerical difference between the two paths would poison the training
set in a way nothing downstream would catch, so compare them element-wise on
real files before trusting the fast path on 91,000 sequences.
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.training.convert_fast import (
    to_features_fast, to_features_slow,
)
from backend.app.landmarks import pose_upper_indices

N = 40
Z = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home()/"Downloads"/"asl-signs.zip")
pose_idx = pose_upper_indices()

with zipfile.ZipFile(Z) as z:
    members = [n for n in z.namelist() if n.endswith(".parquet")]
    rng = np.random.default_rng(0)
    pick = [members[i] for i in rng.choice(len(members), N, replace=False)]
    raws = [z.read(m) for m in pick]

print(f"comparing {N} randomly chosen sequences\n")
worst = 0.0
fell_back = 0
t_fast = t_slow = 0.0
mismatch = []

for m, raw in zip(pick, raws):
    t0 = time.perf_counter(); a = to_features_fast(raw, pose_idx); t_fast += time.perf_counter()-t0
    t0 = time.perf_counter(); b = to_features_slow(raw, pose_idx); t_slow += time.perf_counter()-t0
    if a is None:
        fell_back += 1
        continue
    if a.shape != b.shape:
        mismatch.append((m, f"shape {a.shape} vs {b.shape}")); continue
    d = float(np.abs(a - b).max())
    worst = max(worst, d)
    if d > 1e-5:
        mismatch.append((m, f"max|diff| = {d:.3e}"))
    # mask channel must agree EXACTLY, not approximately
    if not np.array_equal(a[..., 2], b[..., 2]):
        n = int((a[..., 2] != b[..., 2]).sum())
        mismatch.append((m, f"{n} mask cells differ"))

print(f"  worst element-wise |fast - slow| : {worst:.3e}")
print(f"  layout fallbacks                 : {fell_back}/{N}")
print(f"  mismatches                       : {len(mismatch)}")
for m, why in mismatch[:5]:
    print(f"      {Path(m).name}: {why}")
print(f"\n  fast {t_fast*1000/N:7.1f} ms/seq")
print(f"  slow {t_slow*1000/N:7.1f} ms/seq")
print(f"  speedup {t_slow/max(t_fast,1e-9):.0f}x")

ok = not mismatch and worst < 1e-5
print(f"\n{'='*58}\n  EQUIVALENT: {ok}\n{'='*58}")
sys.exit(0 if ok else 1)
