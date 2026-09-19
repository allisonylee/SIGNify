"""
FastAPI app: one websocket endpoint, plus health.

SERVES NO HTML. The page belongs to the frontend partner; this process only
speaks the wire contract in plan section 14.1.
"""
from __future__ import annotations

import asyncio
import base64
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import config
from .ingest import IngestError, ingest
from .inference import FrameBuffer, StubRecognizer, glosses_to_text
from .landmarks import HolisticExtractor
from .tts import get_provider

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["tts"] = get_provider()
    print(f"[startup] tts provider : {STATE['tts'].name}")
    print(f"[startup] audio        : {config.AUDIO_FORMAT}")
    print("[startup] mode B extractor is created lazily, per connection")
    yield
    STATE.clear()


app = FastAPI(title="Sign Language Backend", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "tts": STATE["tts"].name,
        "audio_format": config.AUDIO_FORMAT,
        "sample_rate": config.AUDIO_SAMPLE_RATE,
        "window": {"frames": config.WINDOW_FRAMES, "seconds": config.WINDOW_SECONDS},
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    buf = FrameBuffer()
    rec = StubRecognizer()
    tts = STATE["tts"]
    extractor = None                      # mode B only, created on first frame
    cfg = {"sign_language": "ase", "output_language": "en", "mode": "speech"}
    n_in = 0
    t_connect = time.perf_counter()

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "config":
                cfg.update({k: v for k, v in msg.items() if k != "type"})
                await ws.send_json({"type": "state", "signing": False,
                                    "config": cfg})
                continue

            if kind == "frame" and extractor is None:
                extractor = HolisticExtractor()
                print("[ws] mode B: created HolisticExtractor")

            try:
                if kind == "frame":
                    # MediaPipe is ~30ms of CPU. Running it inline would block
                    # the event loop, stalling this connection's reads and any
                    # other. Safe in a thread because calls stay sequential per
                    # connection (VIDEO mode needs monotonic timestamps).
                    frame = await asyncio.to_thread(ingest, msg, extractor)
                else:
                    frame = ingest(msg, extractor)
            except IngestError as e:
                await ws.send_json({"type": "error", "detail": str(e)})
                continue

            n_in += 1
            buf.add(frame)
            if not buf.ready():
                continue

            t0 = time.perf_counter()
            window = buf.window()
            window_ms = (time.perf_counter() - t0) * 1000

            result = rec.infer(window)
            if result is None:
                continue
            result.window_ms = window_ms

            # Text goes out IMMEDIATELY -- before any LLM or TTS work. This is
            # the sub-100ms path from plan section 8.
            await ws.send_json({"type": "partial", "gloss": result.gloss,
                                "conf": result.confidence})
            t_partial = time.perf_counter()

            text = glosses_to_text(result.gloss)
            await ws.send_json({"type": "text", "text": text,
                                "lang": cfg["output_language"]})

            timings = {
                "extract_ms": round(frame.extract_ms, 1),
                "window_ms": round(result.window_ms, 1),
                "infer_ms": round(result.infer_ms, 2),
            }

            if cfg.get("mode") == "speech":
                t1 = time.perf_counter()
                try:
                    pcm = await tts.synthesize(text, cfg["output_language"])
                except Exception as e:                       # noqa: BLE001
                    await ws.send_json({"type": "error",
                                        "detail": f"tts: {e}"})
                    continue
                timings["tts_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                await ws.send_json({
                    "type": "audio", "seq": n_in, "fmt": config.AUDIO_FORMAT,
                    "sample_rate": config.AUDIO_SAMPLE_RATE,
                    "chunk": base64.b64encode(pcm).decode(),
                })

            timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            timings["text_first_ms"] = round((t_partial - t0) * 1000, 1)
            await ws.send_json({"type": "timing", **timings})
            print(f"[ws] {frame.source:9s} frames={n_in:4d} "
                  f"buf={len(buf):3d} span={buf.span():.2f}s  {timings}")

    except WebSocketDisconnect:
        dur = time.perf_counter() - t_connect
        print(f"[ws] closed after {dur:.1f}s, {n_in} frames "
              f"({n_in/dur:.1f}/s), {buf.dropped_out_of_order} out-of-order")
    finally:
        if extractor is not None:
            extractor.close()
