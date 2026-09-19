"""
v023 -- do the sliders actually change the voice, and do they cost anything?

NOT SHIPPED. Sends the config sign.js sends, with different slider positions,
and checks the audio that comes back really is different -- and that resolving
a voice adds no measurable time to the speech path.
"""
import asyncio, base64, hashlib, json, statistics, sys, time
import websockets

V = ["hello", "me", "happy", "see", "you"]
FAILED = []
def check(n, ok, d=""):
    if not ok: FAILED.append(n)
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}{('  ' + d) if d else ''}")

async def utter(voice):
    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=16*1024*1024) as ws:
        cfg = {"type": "config", "sign_language": "ase", "output_language": "en",
               "mode": "speech", "expect": 5}
        if voice is not None:
            cfg["voice"] = voice
        await ws.send(json.dumps(cfg))
        await asyncio.sleep(0.3)
        while True:
            try: await asyncio.wait_for(ws.recv(), 0.2)
            except asyncio.TimeoutError: break
        for g in V:
            await ws.send(json.dumps({"type": "gloss", "value": g}))
            await asyncio.sleep(0.02)
        t5 = time.perf_counter()
        audio = tts_ms = None
        end = time.perf_counter() + 30
        while time.perf_counter() < end:
            try: m = json.loads(await asyncio.wait_for(ws.recv(), 6))
            except asyncio.TimeoutError: break
            if m.get("type") == "audio": audio = base64.b64decode(m["chunk"])
            if m.get("type") == "timing":
                tts_ms = m.get("tts_ms"); break
        return audio, tts_ms, (time.perf_counter() - t5) * 1000

async def main():
    print("\n1. different slider positions -> genuinely different audio")
    picks = [(1,1),(1,5),(3,3),(5,1),(5,5)]
    sigs, times = {}, []
    for p,a in picks:
        audio, tts_ms, total = await utter({"pitch": p, "age": a})
        if audio is None:
            check(f"   pitch={p} age={a}", False, "no audio"); continue
        sigs[(p,a)] = hashlib.sha256(audio).hexdigest()[:12]
        times.append(tts_ms)
        print(f"   pitch={p} age={a}  {len(audio):6,} B  {len(audio)/32000:.2f}s  "
              f"sha={sigs[(p,a)]}  tts={tts_ms} ms")
        await asyncio.sleep(0.3)
    check("   all five are distinct recordings", len(set(sigs.values())) == len(sigs),
          f"{len(set(sigs.values()))} distinct of {len(sigs)}")

    print("\n2. resolving a voice must not cost speech latency")
    # INTERLEAVED and n=8 each. The first version of this compared 5 grid runs
    # against 3 default runs taken at a different moment, and "passed" a +98 ms
    # gap with a tolerance loose enough to be meaningless. Network TTS varies by
    # over 100 ms run to run, so the comparison has to be paired and the verdict
    # has to be against the observed spread, not an invented threshold.
    base, grid = [], []
    for _ in range(8):
        base.append((await utter(None))[1])
        await asyncio.sleep(0.25)
        grid.append((await utter({"pitch": 3, "age": 3}))[1])
        await asyncio.sleep(0.25)
    bm, gm = statistics.median(base), statistics.median(grid)
    spread = statistics.pstdev(base + grid)
    print(f"   default premade  n={len(base)}  p50 {bm:6.1f} ms  "
          f"min {min(base):.1f}  max {max(base):.1f}")
    print(f"   grid voice [3,3] n={len(grid)}  p50 {gm:6.1f} ms  "
          f"min {min(grid):.1f}  max {max(grid):.1f}")
    check("   grid voice is indistinguishable from the default",
          abs(gm - bm) < spread, f"{gm - bm:+.1f} ms vs spread {spread:.1f} ms")

    print("\n3. out-of-range slider values must not break anything")
    audio, _, _ = await utter({"pitch": 99, "age": -4})
    check("   clamped and still spoke", audio is not None and len(audio) > 0,
          f"{len(audio) if audio else 0:,} B")

    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0

sys.exit(asyncio.run(main()))
