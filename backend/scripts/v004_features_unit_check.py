"""
v004 -- unit checks for app/features.py.

Not a test framework, just assertions with printed numbers, per the project
rule that outputs are verified by looking at the values.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import features as F
from backend.app import landmarks as L

ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")


def make_frame(shoulder_y=0.3, half_w=0.1, cx=0.5, hands=True):
    h = {}
    if hands:
        h["left"] = np.full((21, 2), [cx + 0.2, 0.5], dtype=np.float32)
        h["right"] = np.full((21, 2), [cx - 0.2, 0.5], dtype=np.float32)
    else:
        h["left"] = h["right"] = None
    pose = np.zeros((11, 2), dtype=np.float32)
    for i, nm in enumerate(L.POSE_UPPER_NAMES):
        pose[i] = [cx, 0.2]
    pose[L.POSE_UPPER_NAMES.index("LEFT_SHOULDER")] = [cx + half_w, shoulder_y]
    pose[L.POSE_UPPER_NAMES.index("RIGHT_SHOULDER")] = [cx - half_w, shoulder_y]
    return F.assemble(h, pose)

print("\n[1] assemble shape + mask semantics")
f = make_frame()
check("shape is (53,3)", f.shape == (53, 3), str(f.shape))
check("all detected when present", f[:, 2].sum() == 53, f"{f[:,2].sum():.0f}/53")
f_nohands = make_frame(hands=False)
check("absent hands -> mask 0", f_nohands[L.SLICE_LEFT_HAND, 2].sum() == 0)
check("absent hands -> coords 0", np.all(f_nohands[L.SLICE_LEFT_HAND, :2] == 0))
check("pose still detected", f_nohands[L.SLICE_POSE, 2].sum() == 11)

print("\n[2] normalise: translation invariance")
a = F.normalise(make_frame(cx=0.5))
b = F.normalise(make_frame(cx=0.8))          # same pose, shifted right
d = np.abs(a[:, :2] - b[:, :2]).max()
check("shift of +0.3 cancels out", d < 1e-5, f"max|diff| = {d:.2e}")

print("\n[3] normalise: scale invariance")
a = F.normalise(make_frame(half_w=0.10))
b = F.normalise(make_frame(half_w=0.20))     # twice as close to camera
# hands sit at a fixed ABSOLUTE offset in make_frame, so they should NOT match;
# the shoulders themselves must land at +-0.5 in both.
lsh_a, lsh_b = a[L.ROW_LEFT_SHOULDER, 0], b[L.ROW_LEFT_SHOULDER, 0]
check("left shoulder -> +0.5 regardless of scale",
      abs(lsh_a - 0.5) < 1e-5 and abs(lsh_b - 0.5) < 1e-5,
      f"{lsh_a:.4f} / {lsh_b:.4f}")
check("shoulder midpoint -> origin",
      abs(a[L.ROW_LEFT_SHOULDER, 0] + a[L.ROW_RIGHT_SHOULDER, 0]) < 1e-5)

print("\n[4] normalise: refuses to divide by a tiny shoulder width")
tiny = F.normalise(make_frame(half_w=0.001))
check("degenerate width left raw", np.isfinite(tiny).all() and tiny[:, :2].max() <= 1.0,
      f"max = {tiny[:,:2].max():.3f}")

print("\n[5] resample_by_time: variable frame rate")
frames = np.stack([make_frame(cx=0.5 + 0.01 * i) for i in range(30)])
t_even = np.linspace(0.0, 1.0, 30)
rng = np.random.default_rng(0)
t_jitter = np.sort(rng.uniform(0, 1, 30)); t_jitter[0], t_jitter[-1] = 0.0, 1.0
frames_j = np.stack([make_frame(cx=0.5 + 0.01 * (29 * t)) for t in t_jitter])
r_even = F.resample_by_time(t_even, frames, 32, 1.0)
r_jit = F.resample_by_time(t_jitter, frames_j, 32, 1.0)
check("output shape (32,53,3)", r_even.shape == (32, 53, 3), str(r_even.shape))
d = np.abs(r_even[:, :, :2] - r_jit[:, :, :2]).max()
check("30fps even vs jittered agree", d < 0.02, f"max|diff| = {d:.4f}")

print("\n[6] resample_by_time: mask stays binary after interpolation")
uniq = np.unique(r_even[:, :, 2])
check("mask values are 0/1 only", set(uniq.tolist()) <= {0.0, 1.0}, str(uniq.tolist()))

print("\n[7] resample_by_time: degenerate inputs")
check("empty -> zeros", F.resample_by_time([], np.zeros((0, 53, 3)), 32, 1.0).shape == (32, 53, 3))
check("single frame -> repeated",
      np.allclose(F.resample_by_time([0.5], frames[:1], 32, 1.0), frames[0]))

print("\n[8] motion_energy")
still = np.repeat(make_frame()[None], 32, axis=0)
moving = np.stack([make_frame(cx=0.3 + 0.01 * i) for i in range(32)])
e_still, e_move = F.motion_energy(F_still := still), F.motion_energy(moving)
check("still < moving", e_still < e_move, f"{e_still:.5f} vs {e_move:.5f}")
check("still is ~zero", e_still < 1e-6, f"{e_still:.2e}")

print(f"\n{'=' * 60}\n  {sum(ok)}/{len(ok)} checks passed\n{'=' * 60}")
sys.exit(0 if all(ok) else 1)
