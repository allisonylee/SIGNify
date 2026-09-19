"""
v019 -- state-machine checks for RestGatedRecognizer.

Feeds controlled score sequences instead of real video, so the segmentation
logic can be verified without a camera: does it stay silent on rest, fire once
per sign, pick the PEAK rather than the last frame, and separate two signs?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.inference import RestGatedRecognizer

ok = []
def check(n, c, d=""):
    ok.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}   {d}")


class FakeBuf:
    """Minimal stand-in: the recognizer only needs total/len/window/segment."""
    def __init__(self): self.total = 0
    def add(self): self.total += 1
    def __len__(self): return max(self.total, 10)
    def window(self): return np.zeros((32, 53, 3), np.float32)
    def segment(self, start): return np.zeros((32, 53, 3), np.float32)


def make(scripted):
    """A recognizer whose score() returns the next scripted result."""
    r = RestGatedRecognizer.__new__(RestGatedRecognizer)
    r.labels = ["__REST__", "hello", "thankyou"]
    r.rest_label = "__REST__"; r.rest_idx = 0
    r.min_confidence = 0.18; r.min_margin = 0.0; r.debounce_s = 1.0
    r.stride = 1; r.enter_rest_below = 0.55; r.enter_conf = 0.18
    r.exit_rest_above = 0.70; r.exit_confirm = 2; r.max_sign_s = 3.0
    r.state = "idle"; r._tick = 0; r._rest_streak = 0; r._best = None
    r._seg_start = 0; r._seg_start_t = 0.0; r._last_gloss = None
    r._last_emit_t = -1e9; r.last_rest_p = 1.0; r.last_scores = []
    r.last_reject = ""
    r.rejected = {"confidence":0,"margin":0,"rest":0,"debounce":0,"too_short":0}
    it = iter(scripted)
    r.score = lambda w: next(it)
    return r


def run(r, n, fps=25.0):
    buf = FakeBuf(); out = []
    for i in range(n):
        buf.add()
        res = r.observe(buf, i / fps)
        if res: out.append((round(i/fps, 2), res.gloss[0], res.confidence))
    return out

REST = [("__REST__", 0.93), ("hello", 0.02), ("thankyou", 0.01)]
def sign(w, p, rest=0.05):
    other = "thankyou" if w == "hello" else "hello"
    return sorted([(w, p), ("__REST__", rest), (other, 0.02)],
                  key=lambda x: -x[1])

print("\n[1] pure rest -> silence")
r = make([REST]*30)
check("no emissions", run(r, 30) == [], str(run.__name__))

print("\n[2] rest -> one sign -> rest  => exactly one emission, at the PEAK")
script = [REST]*5 + [sign("hello",0.30), sign("hello",0.62), sign("hello",0.88),
                     sign("hello",0.55)] + [REST]*6
r = make(script); out = run(r, len(script))
check("exactly one emission", len(out) == 1, str(out))
check("correct gloss", out and out[0][1] == "hello")
check("PEAK confidence kept, not the last", out and abs(out[0][2]-0.88) < 1e-6,
      f"got {out[0][2] if out else None}, peak was 0.88, last was 0.55")

print("\n[3] two signs separated by rest => two emissions")
script = ([REST]*4 + [sign("hello",0.7)]*3 + [REST]*4
          + [sign("thankyou",0.8)]*3 + [REST]*4)
r = make(script); out = run(r, len(script))
check("two emissions", len(out) == 2, str([(g,c) for _,g,c in out]))
check("in the right order", [g for _,g,_ in out] == ["hello","thankyou"])

print("\n[4] a weak sign is dropped by the confidence floor")
script = [REST]*4 + [sign("hello",0.10)]*3 + [REST]*4
r = make(script); out = run(r, len(script))
check("no emission below 0.18", out == [], str(out))

print("\n[5] brief rest dips do NOT split one sign")
script = ([REST]*4 + [sign("hello",0.6), sign("hello",0.7),
                      [("__REST__",0.72),("hello",0.2),("thankyou",0.01)],
                      sign("hello",0.85), sign("hello",0.6)] + [REST]*6)
r = make(script); out = run(r, len(script))
check("one emission, not two", len(out) == 1, str(out))
check("kept the later peak 0.85", out and abs(out[0][2]-0.85) < 1e-6)

print(f"\n{'='*58}\n  {sum(ok)}/{len(ok)} checks passed\n{'='*58}")
sys.exit(0 if all(ok) else 1)
