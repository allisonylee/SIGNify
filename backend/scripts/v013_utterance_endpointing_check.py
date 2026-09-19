"""
v013 -- utterance endpointing: when does a sentence end?

The recognizer emits ONE gloss at a time. Something has to decide which glosses
belong to the same sentence, or every sign gets spoken alone and the LLM never
sees a sequence to make grammatical. UtteranceBuffer does that by timeout.

These checks pin the behaviour that decides it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.inference import UtteranceBuffer
from backend.app.llm import should_bypass

ok = []
def check(n, c, d=""):
    ok.append(bool(c)); print(f"  [{'PASS' if c else 'FAIL'}] {n}   {d}")

TO = 2.0

print("\n[1] empty buffer never flushes")
u = UtteranceBuffer(timeout_s=TO)
check("silent at t=0", u.flush_reason(0.0) is None)
check("silent at t=100", u.flush_reason(100.0) is None, "no signs -> no sentence")

print("\n[2] one gloss, then silence -> flush after the timeout")
u = UtteranceBuffer(timeout_s=TO)
u.add("HELLO", 10.0)
check("no flush at +0.5s", u.flush_reason(10.5) is None)
check("no flush at +1.9s", u.flush_reason(11.9) is None)
check("FLUSH at +2.0s", u.flush_reason(12.0) == "timeout")
check("single gloss bypasses the LLM", should_bypass(u.take()))

print("\n[3] three glosses close together -> ONE sentence, LLM fires")
u = UtteranceBuffer(timeout_s=TO)
for g, t in [("MOTHER", 10.0), ("HAPPY", 11.0), ("VISIT", 12.2)]:
    u.add(g, t)
    check(f"  still open after {g} @{t}", u.flush_reason(t) is None)
check("no flush 1.5s after last", u.flush_reason(13.7) is None,
      "clock resets on each gloss")
check("FLUSH 2.0s after last", u.flush_reason(14.2) == "timeout")
gl = u.take()
check("3 glosses in one utterance", gl == ["MOTHER", "HAPPY", "VISIT"], str(gl))
check("does NOT bypass -> LLM gets the sequence", not should_bypass(gl))

print("\n[4] a long gap splits into two sentences")
u = UtteranceBuffer(timeout_s=TO)
u.add("HELLO", 0.0); u.add("NAME", 0.8)
first = None
if u.flush_reason(3.0):
    first = u.take()
u.add("ALLISON", 4.0); u.add("NICE", 4.9)
second = u.take() if u.flush_reason(7.5) else None
check("first sentence", first == ["HELLO", "NAME"], str(first))
check("second sentence", second == ["ALLISON", "NICE"], str(second))

print("\n[5] safety valves stop unbounded growth")
u = UtteranceBuffer(timeout_s=TO, max_glosses=4)
for i in range(4):
    u.add(f"W{i}", 10.0 + i * 0.3)
check("max_glosses fires", u.flush_reason(11.2) == "max_glosses",
      "even though the timeout has not elapsed")
u = UtteranceBuffer(timeout_s=TO, max_glosses=99, max_duration_s=5.0)
for i in range(10):
    u.add(f"W{i}", i * 0.6)
check("max_duration fires", u.flush_reason(5.5) == "max_duration")

print("\n[6] the timing the user actually feels")
u = UtteranceBuffer(timeout_s=TO)
u.add("HELLO", 100.0)
check("countdown reported", abs(u.seconds_until_flush(100.5) - 1.5) < 1e-6,
      f"{u.seconds_until_flush(100.5):.2f}s until speech")
print(f"      -> text appears instantly; speech waits {TO}s after the LAST sign")

print(f"\n{'='*60}\n  {sum(ok)}/{len(ok)} checks passed\n{'='*60}")
sys.exit(0 if all(ok) else 1)
