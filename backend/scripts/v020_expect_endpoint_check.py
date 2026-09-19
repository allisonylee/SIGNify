"""
v020 -- does count-based endpointing actually fire where we claim?

NOT SHIPPED. Pure logic check on UtteranceBuffer.flush_reason: no camera, no
model, no server. Feeds it synthetic gloss timings and asserts the reason and
the moment it flushes.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from backend.app.inference import UtteranceBuffer

FAILED = []

def check(name, got, want):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, want {want!r}")


print("\n1. five signs arrive -> fires on the 5th, not on the clock")
u = UtteranceBuffer(timeout_s=0.8, expect=5)
t = 0.0
for i, w in enumerate(["hello", "me", "happy", "see", "you"]):
    t += 2.0                       # 2s apart: way past the 0.8s timeout
    u.add(w, t)
    r = u.flush_reason(t)
    check(f"   after {i+1} sign(s)", r, "expect" if i == 4 else None)
check("   flushes immediately, no wait", u.seconds_until_flush(t), 0.0)
check("   contents", u.take(), ["hello", "me", "happy", "see", "you"])

print("\n2. a long mid-sentence pause must NOT flush a half sentence")
u = UtteranceBuffer(timeout_s=0.8, expect=5)
u.add("hello", 0.0); u.add("me", 1.0)
check("   +0.9s after last gloss (past timeout_s)", u.flush_reason(1.9), None)
check("   +3.9s (past max_duration 6.0 from start? no, still under)",
      u.flush_reason(4.9), None)
check("   +3.99s, just under the 4.0s backstop", u.flush_reason(4.99), None)

print("\n3. but a MISSED sign must not hang forever")
check("   +4.0s of stillness -> speak what we have",
      u.flush_reason(5.0), "expect_backstop")
check("   contents", u.take(), ["hello", "me"])

print("\n4. max_duration must NOT cut off a slow 5-sign sentence")
u = UtteranceBuffer(timeout_s=0.8, max_duration_s=6.0, expect=5)
t = 0.0
for w in ["hello", "me", "happy", "see"]:
    t += 2.5
    u.add(w, t)                    # t = 10.0s, well past max_duration_s
check("   4 signs over 10s, still waiting", u.flush_reason(t), None)
u.add("you", t + 2.5)
check("   5th arrives at 12.5s -> fires", u.flush_reason(t + 2.5), "expect")

print("\n5. max_glosses is still a hard cap (model double-firing)")
u = UtteranceBuffer(timeout_s=0.8, max_glosses=12, expect=99)
for i in range(12):
    u.add("hello", float(i))
check("   12 glosses with expect=99", u.flush_reason(11.0), "max_glosses")

print("\n6. expect=0 leaves the existing timeout behaviour untouched")
u = UtteranceBuffer(timeout_s=0.8, expect=0)
u.add("hello", 0.0)
check("   +0.5s", u.flush_reason(0.5), None)
check("   +0.8s", u.flush_reason(0.8), "timeout")

print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
