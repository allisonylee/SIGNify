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
from .inference import (FrameBuffer, StubRecognizer, UtteranceBuffer,
                        load_model_recognizer)
from .llm import get_translator, should_bypass
from .tts import VoiceSettings, PRESETS
from .landmarks import HolisticExtractor
from .tts import get_provider

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["tts"] = get_provider()
    STATE["llm"] = get_translator()
    rec = load_model_recognizer()
    STATE["recognizer_factory"] = load_model_recognizer if rec else None
    if rec is None:
        print("[startup] recognizer   : STUB (no encoder_ase.pt found)")
    else:
        print(f"[startup] recognizer   : ModelRecognizer, {len(rec.labels)} classes, "
              f"{rec.device}")
        print(f"[startup] rest class   : {rec.rest_idx is not None}"
              + ("" if rec.rest_idx is not None
                 else "  <- no REST class; confidence/margin floors are the "
                      "only false-positive guard"))
    llm = STATE["llm"]
    print(f"[startup] llm          : {llm.model if llm.available else 'DISABLED (no key) -> naive join'}")
    print(f"[startup] tts provider : {STATE['tts'].name}")
    print(f"[startup] audio        : {config.AUDIO_FORMAT}")
    print("[startup] mode B extractor is created lazily, per connection")
    yield
    STATE.clear()


app = FastAPI(title="Sign Language Backend", lifespan=lifespan)


@app.get("/voices")
async def voices():
    """Voice library for the UI picker. The frontend never sees the API key."""
    tts = STATE["tts"]
    items = await tts.list_voices()
    return {"voices": items, "presets": PRESETS,
            "limited": getattr(tts, "voices_limited", False),
            "note": getattr(tts, "voices_error", "") or None}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "tts": STATE["tts"].name,
        "audio_format": config.AUDIO_FORMAT,
        "sample_rate": config.AUDIO_SAMPLE_RATE,
        "window": {"frames": config.WINDOW_FRAMES, "seconds": config.WINDOW_SECONDS},
        "llm": STATE["llm"].model if STATE["llm"].available else None,
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    buf = FrameBuffer()
    factory = STATE.get("recognizer_factory")
    rec = factory() if factory else StubRecognizer()
    utt = UtteranceBuffer()
    # SPECULATIVE PREFETCH (plan 8.1). After each gloss we start translating
    # the utterance-so-far in the background. If another gloss arrives we throw
    # that away and start again; if none does, the result is already in hand
    # when the timeout fires, so the LLM's ~570 ms leaves the path the user
    # feels. Cost is modest: 1-2 gloss utterances bypass the model anyway.
    prefetch = None
    prefetch_for: list[str] = []

    def start_prefetch():
        """Kick off a background translation of the utterance so far."""
        nonlocal prefetch, prefetch_for
        if prefetch is not None and not prefetch.done():
            prefetch.cancel()
        prefetch_for = list(utt.glosses)
        prefetch = asyncio.create_task(
            llm.translate(prefetch_for, cfg["output_language"]))
    tts = STATE["tts"]
    llm = STATE["llm"]
    extractor = None                      # mode B only, created on first frame
    cfg = {"sign_language": "ase", "output_language": "en", "mode": "speech"}
    voice = VoiceSettings.from_client(None)
    n_in = 0
    t_connect = time.perf_counter()

    try:
        while True:
            # Wake up regularly even with no traffic. Endpointing is driven by
            # the PASSAGE OF TIME, not by messages -- if it only ran on arrival,
            # a stalled camera or a client that stops sending would leave a
            # half-finished utterance pending forever.
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.15)
            except (asyncio.TimeoutError, TimeoutError):
                msg, kind = None, None
            else:
                kind = msg.get("type")

            if msg is None:
                # idle tick: only the endpoint check below matters
                now = time.perf_counter() - t_connect
                frame = None
            else:
                kind = msg.get("type")

            if kind == "config":
                cfg.update({k: v for k, v in msg.items()
                            if k not in ("type", "voice")})
                if "voice" in msg:
                    # Clamped server-side: a client sending speed=50 is capped,
                    # not passed through (plan 15.5).
                    voice = VoiceSettings.from_client(msg["voice"])
                await ws.send_json({"type": "state", "signing": False,
                                    "config": cfg, "voice": voice.as_dict()})
                continue

            if kind == "gloss":
                # MANUAL GLOSS INJECTION. Feeds the utterance buffer directly,
                # bypassing the camera and the recognizer entirely.
                #
                # This exists so the downstream chain -- utterance grouping,
                # endpointing, the LLM, voice settings, TTS -- can be tested
                # and demoed WITHOUT depending on recognition accuracy. The two
                # halves of the system are independent, and this keeps them
                # independently debuggable.
                g = str(msg.get("value", "")).strip()
                if not g:
                    await ws.send_json({"type": "error",
                                        "detail": "gloss message needs 'value'"})
                    continue
                utt.add(g, msg.get("t", time.perf_counter() - t_connect))
                await ws.send_json({"type": "partial",
                                    "gloss": list(utt.glosses),
                                    "conf": 1.0, "source": "manual"})
                print(f"[ws] MANUAL gloss {g!r} utterance={len(utt)}")
                start_prefetch()
                if not msg.get("flush"):
                    continue
                # flush=true ends the utterance immediately, so a test does not
                # have to wait out the endpoint timeout.
                utt.last_t = -1e9
                frame = None

            if kind == "frame" and extractor is None:
                extractor = HolisticExtractor()
                print("[ws] mode B: created HolisticExtractor")

            if kind in ("gloss", None):
                pass          # idle tick, or gloss already handled: fall through
            else:
              try:
                if kind == "frame":
                    # MediaPipe is ~40 ms of CPU; inline it would block the
                    # event loop and stall this connection's reads.
                    frame = await asyncio.to_thread(ingest, msg, extractor)
                else:
                    frame = ingest(msg, extractor)
              except IngestError as e:
                await ws.send_json({"type": "error", "detail": str(e)})
                continue

            if msg is not None:
                now = (time.perf_counter() - t_connect) if frame is None else frame.t
            if frame is not None:
                n_in += 1
                buf.add(frame)

            # ---- 1. recognition: may or may not commit a gloss ----
            if frame is not None and buf.ready():
                t0 = time.perf_counter()
                if hasattr(rec, "observe"):
                    before = sum(rec.rejected.values())
                    result = rec.observe(buf, frame.t)
                    window_ms = 0.0
                    if result is None and sum(rec.rejected.values()) > before:
                        # A segment WAS classified but a guard threw it away.
                        # Surface it -- otherwise "nothing happens" is
                        # indistinguishable from "not detected at all".
                        why = max(rec.rejected, key=lambda k: rec.rejected[k]
                                  if rec.rejected[k] else -1)
                        why = next((k for k in rec.rejected
                                    if rec.rejected[k] and k == why), why)
                        await ws.send_json({
                            "type": "rejected", "reason": why,
                            "top": [[n, round(p, 3)] for n, p in rec.last_scores[:3]],
                            "counts": dict(rec.rejected)})
                        print(f"[ws] rejected ({why}): "
                              f"{[(n, round(p,2)) for n,p in rec.last_scores[:3]]}")
                else:
                    window = buf.window()
                    window_ms = (time.perf_counter() - t0) * 1000
                    result = rec.infer(window)

                if result is not None:
                    result.window_ms = result.window_ms or window_ms
                    # Text goes out IMMEDIATELY, per gloss, before any LLM or
                    # TTS work. The screen keeps up; only audio waits.
                    utt.add(result.gloss[0], now)
                    await ws.send_json({"type": "partial",
                                        "gloss": list(utt.glosses),
                                        "conf": result.confidence})
                    await ws.send_json({"type": "timing",
                        "extract_ms": round(frame.extract_ms, 1),
                        "window_ms": round(result.window_ms, 1),
                        "infer_ms": round(result.infer_ms, 2),
                        "text_first_ms": round(
                            (time.perf_counter() - t0) * 1000, 1),
                        "utterance": len(utt)})
                    print(f"[ws] gloss {result.gloss[0]!r} "
                          f"conf={result.confidence} utterance={len(utt)}")

                    start_prefetch()

            # Mid-sign? Then the sentence is not over, whatever the clock
            # says. Without this the utterance ends while the user is still
            # signing (see UtteranceBuffer.keep_alive).
            if getattr(rec, "state", None) == "signing":
                utt.keep_alive(now)

            # ---- 2. endpointing: checked on EVERY frame ----
            # An utterance ends because no new sign arrived, so this cannot sit
            # behind the "a gloss was committed" branch -- it is precisely the
            # absence of a gloss that ends it.
            reason = utt.flush_reason(now)
            if reason is None:
                continue

            glosses = utt.take()
            t_utt = time.perf_counter()
            if prefetch is not None and prefetch_for == glosses:
                try:
                    text = await prefetch          # usually already finished
                    llm_source = "prefetch"
                except asyncio.CancelledError:
                    text = await llm.translate(glosses, cfg["output_language"])
                    llm_source = "gemini"
            else:
                if prefetch is not None and not prefetch.done():
                    prefetch.cancel()
                text = await llm.translate(glosses, cfg["output_language"])
                llm_source = "bypass" if should_bypass(glosses) else "gemini"
            prefetch, prefetch_for = None, []
            llm_ms = (time.perf_counter() - t_utt) * 1000
            await ws.send_json({"type": "text", "text": text,
                                "lang": cfg["output_language"],
                                "glosses": glosses, "end_reason": reason})

            timings = {"llm_ms": round(llm_ms, 1), "llm": llm_source,
                       "glosses": len(glosses), "end_reason": reason}

            if cfg.get("mode") == "speech":
                t1 = time.perf_counter()
                try:
                    pcm = await tts.synthesize(text, cfg["output_language"], voice)
                except Exception as e:                       # noqa: BLE001
                    await ws.send_json({"type": "error", "detail": f"tts: {e}"})
                    continue
                timings["tts_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                await ws.send_json({
                    "type": "audio", "seq": n_in, "fmt": config.AUDIO_FORMAT,
                    "sample_rate": config.AUDIO_SAMPLE_RATE,
                    "chunk": base64.b64encode(pcm).decode(),
                })

            timings["total_ms"] = round((time.perf_counter() - t_utt) * 1000, 1)
            await ws.send_json({"type": "timing", **timings})
            print(f"[ws] UTTERANCE {glosses} -> {text!r}  {timings}")

    except WebSocketDisconnect:
        dur = time.perf_counter() - t_connect
        print(f"[ws] closed after {dur:.1f}s, {n_in} frames "
              f"({n_in/dur:.1f}/s), {buf.dropped_out_of_order} out-of-order")
    finally:
        if prefetch is not None and not prefetch.done():
            prefetch.cancel()
        if extractor is not None:
            extractor.close()
