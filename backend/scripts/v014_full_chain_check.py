"""
v014 -- does the FULL server chain fire? NOT SHIPPED.

Replays synthetic LANDMARK frames (mode A) over the websocket that are shaped
to trigger the motion gate, so the whole server path runs without a camera:

    landmarks -> recognizer -> UtteranceBuffer -> Gemini -> ElevenLabs -> audio

This exists because --source synthetic in harness_client sends JPEGs of a stick
figure that MediaPipe does not detect at all (0% pose), so it exercises
transport only. Sending landmarks directly skips MediaPipe and reaches the
recognizer.

    python -m uvicorn backend.app.main:app --port 8000     # terminal 1
    python backend/scripts/v014_full_chain_check.py        # terminal 2
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.landmarks import N_HAND, N_POSE_UPPER, POSE_UPPER_NAMES

IDX = {n: i for i, n in enumerate(POSE_UPPER_NAMES)}
FPS = 20


def landmarks(i: int, moving: bool):
    pose = np.tile([[0.5, 0.3]], (N_POSE_UPPER, 1)).astype(np.float32)
    pose[IDX["LEFT_SHOULDER"]] = [0.60, 0.30]
    pose[IDX["RIGHT_SHOULDER"]] = [0.40, 0.30]
    wx = 0.55 + (0.06 * np.sin(i * 0.9) if moving else 0.0)
    wy = 0.45 + (0.06 * np.cos(i * 0.9) if moving else 0.0)
    pose[IDX["LEFT_WRIST"]] = [wx, wy]
    pose[IDX["RIGHT_WRIST"]] = [0.45, 0.45]
    return ({"left": np.full((N_HAND, 2), wx, np.float32).tolist(), "right": None},
            pose.tolist())


async def main():
    got = {"partial": [], "text": [], "audio": 0, "timing": [], "error": []}
    url = "ws://127.0.0.1:8000/ws"
    print(f"connecting {url}")
    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "config", "sign_language": "ase",
                                  "output_language": "en", "mode": "speech"}))

        async def rx():
            async for raw in ws:
                m = json.loads(raw)
                k = m.get("type")
                if k == "partial":
                    got["partial"].append(m["gloss"])
                    print(f"  <- partial  utterance so far: {m['gloss']}")
                elif k == "text":
                    got["text"].append(m)
                    print(f"  <- TEXT     {m['text']!r}")
                    print(f"               from glosses {m.get('glosses')} "
                          f"(ended: {m.get('end_reason')})")
                elif k == "audio":
                    got["audio"] += 1
                    n = len(base64.b64decode(m["chunk"]))
                    print(f"  <- AUDIO    {n:,} B = {n/2/m['sample_rate']:.2f}s")
                elif k == "timing":
                    got["timing"].append(m)
                    if "llm_ms" in m:
                        print(f"  <- timing   {({a:b for a,b in m.items() if a!='type'})}")
                elif k == "error":
                    got["error"].append(m); print(f"  <- ERROR    {m.get('detail')}")

        task = asyncio.create_task(rx())
        i = 0
        # 3 signs close together -> should form ONE utterance, then a long
        # silence to trip the 2 s endpoint timeout.
        plan = [("rest", False, 1.5)]
        for _ in range(3):
            plan += [("SIGN", True, 1.0), ("gap", False, 0.6)]
        plan += [("silence", False, 4.0)]

        t0 = time.perf_counter()
        for name, moving, dur in plan:
            print(f"  -> {name} {dur}s")
            for _ in range(int(dur * FPS)):
                h, p = landmarks(i, moving)
                await ws.send(json.dumps({"type": "landmarks", "seq": i,
                                          "t": time.perf_counter() - t0,
                                          "hands": h, "pose": p}))
                i += 1
                await asyncio.sleep(1 / FPS)
        await asyncio.sleep(6)          # let LLM + TTS finish
        task.cancel()

    print(f"\n{'='*62}")
    n_utt = len(got["text"])
    print(f"  glosses committed : {len(got['partial'])}")
    print(f"  utterances (text) : {n_utt}")
    print(f"  audio clips       : {got['audio']}")
    print(f"  errors            : {len(got['error'])}")
    llm = [t for t in got["timing"] if "llm_ms" in t]
    for t in llm:
        print(f"    llm={t.get('llm')} {t.get('llm_ms')}ms  "
              f"tts={t.get('tts_ms')}ms  glosses={t.get('glosses')}")
    ok = (n_utt >= 1 and got["audio"] >= 1 and not got["error"])
    multi = any(t.get("glosses", 0) >= 2 for t in llm)
    print(f"\n  FULL CHAIN (text + audio): {'PASS' if ok else 'FAIL'}")
    print(f"  MULTI-GLOSS UTTERANCE    : {'PASS' if multi else 'FAIL'}"
          f"   <- proves the LLM gets a sequence, not one word")
    print(f"{'='*62}")
    return 0 if (ok and multi) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
