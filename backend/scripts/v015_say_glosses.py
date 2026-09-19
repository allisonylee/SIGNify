"""
v015 -- push glosses straight through the LLM + speech chain. NOT SHIPPED.

Bypasses the camera AND the model, so it tests only the Stage 4 half:

    glosses -> UtteranceBuffer -> Gemini -> ElevenLabs -> audio out your speakers

Use it to confirm the speech chain works before blaming recognition.

    python -m uvicorn backend.app.main:app --port 8000        # terminal 1
    python backend/scripts/v015_say_glosses.py clown flower store
    python backend/scripts/v015_say_glosses.py --lang es dad find flower
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

import numpy as np
import websockets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glosses", nargs="+")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    import sounddevice as sd
    url = f"ws://{a.host}:{a.port}/ws"
    print(f"connecting {url}")
    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "config", "output_language": a.lang,
                                  "mode": "text" if a.no_audio else "speech"}))
        await ws.recv()
        for i, g in enumerate(a.glosses):
            last = i == len(a.glosses) - 1
            await ws.send(json.dumps({"type": "gloss", "value": g.upper(),
                                      "t": i * 0.6, "flush": last}))
            print(f"  -> {g.upper()}" + ("   [flush]" if last else ""))
            await asyncio.sleep(0.25)

        heard = False
        try:
            async with asyncio.timeout(30):
                async for raw in ws:
                    m = json.loads(raw)
                    k = m.get("type")
                    if k == "partial":
                        print(f"  <- utterance so far: {m['gloss']}")
                    elif k == "text":
                        print(f"  <- SENTENCE {m['text']!r} [{m['lang']}]")
                    elif k == "timing" and "llm_ms" in m:
                        print(f"  <- timing   {({x:y for x,y in m.items() if x!='type'})}")
                    elif k == "error":
                        print(f"  <- ERROR    {m.get('detail')}")
                    elif k == "audio":
                        pcm = base64.b64decode(m["chunk"])
                        sr = m["sample_rate"]
                        print(f"  <- AUDIO    {len(pcm):,} B = {len(pcm)/2/sr:.2f}s "
                              f"-- playing")
                        if not a.no_audio:
                            sd.play(np.frombuffer(pcm, np.int16), sr); sd.wait()
                        heard = True
                        break
        except (asyncio.TimeoutError, TimeoutError):
            print("  !! timed out waiting for audio")
    print(f"\n  SPEECH CHAIN: {'PASS' if (heard or a.no_audio) else 'FAIL'}")
    return 0 if (heard or a.no_audio) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
