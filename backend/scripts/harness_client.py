"""
Python stand-in for the browser. NOT SHIPPED -- a test tool.

Drives the backend end to end with no frontend in existence:

    webcam/synthetic -> [mode A: MediaPipe here] -> WS -> backend -> audio out

Keep it for the whole hackathon: when integration breaks at hour 30, this tells
you in ten seconds whether the bug is ours or the frontend's.

    python backend/scripts/harness_client.py --source synthetic --mode B
    python backend/scripts/harness_client.py --source webcam    --mode A
    python backend/scripts/harness_client.py --source webcam --no-audio

MIRRORING (plan section 14.2): the wire format is UN-MIRRORED. cv2 gives us
un-mirrored frames already, so we send them as-is. A client that mirrors for
display must un-mirror before sending.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app.landmarks import N_POSE_UPPER  # noqa: E402


def synthetic_frame(i: int, w=640, h=480) -> np.ndarray:
    """
    A crude moving figure. Enough for MediaPipe to usually find *something*,
    and enough to exercise the pipe when the camera is unavailable.
    """
    import cv2
    img = np.full((h, w, 3), 210, np.uint8)
    cx, cy = w // 2, h // 2
    cv2.circle(img, (cx, cy - 110), 55, (170, 150, 140), -1)          # head
    cv2.rectangle(img, (cx - 75, cy - 55), (cx + 75, cy + 120),
                  (90, 90, 160), -1)                                   # torso
    a = i * 0.18
    for sign in (-1, 1):
        hx = int(cx + sign * (110 + 42 * np.cos(a)))
        hy = int(cy - 18 + 42 * np.sin(a))
        cv2.line(img, (cx + sign * 70, cy - 35), (hx, hy), (90, 90, 160), 26)
        cv2.circle(img, (hx, hy), 27, (170, 150, 140), -1)             # hands
    return img


async def run(args):
    import cv2
    import websockets

    play = None
    if not args.no_audio:
        import sounddevice as sd
        play = sd

    cap = None
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("!! cannot open camera. On macOS the terminal needs camera\n"
                  "   permission: System Settings > Privacy & Security > Camera.\n"
                  "   Falling back to --source synthetic.")
            cap, args.source = None, "synthetic"

    extractor = None
    if args.mode == "A":
        from backend.app.landmarks import HolisticExtractor
        extractor = HolisticExtractor()
        print("[harness] mode A: running MediaPipe locally, sending landmarks")
    else:
        print("[harness] mode B: sending JPEG frames, backend extracts")

    url = f"ws://{args.host}:{args.port}/ws"
    print(f"[harness] connecting {url}")
    t_start = time.perf_counter()
    t_sent_done = None
    sent = 0
    ui = {"utterance": [], "sentence": "", "audio": "", "err": "",
          "fps": 0.0, "reject": ""}
    show = (not args.no_preview) and args.source == "webcam"
    got = {"partial": 0, "text": 0, "audio": 0, "timing": 0, "error": 0}

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        cfg = {"type": "config", "sign_language": args.sign_language,
               "output_language": args.output_language,
               "mode": "text" if args.no_audio else "speech"}
        expect = max(0, args.expect)
        if expect:
            cfg["expect"] = expect
            print(f"  waiting for {expect} signs before sending to the LLM "
                  f"(backstop: speaks anyway after ~4s of stillness)")
        await ws.send(json.dumps(cfg))

        async def receive():
            async for raw in ws:
                m = json.loads(raw)
                k = m.get("type")
                got[k] = got.get(k, 0) + 1
                if k == "partial":
                    # 'gloss' is the utterance SO FAR, not a single word.
                    ui["utterance"] = m["gloss"]
                    print(f"  <- gloss   utterance so far: {m['gloss']}  "
                          f"conf={m['conf']}")
                elif k == "text":
                    ui["sentence"] = m["text"]
                    ui["utterance"] = []
                    print(f"  <- SENTENCE {m['text']!r} [{m['lang']}]")
                    if m.get("glosses"):
                        print(f"               from {m['glosses']} "
                              f"(ended: {m.get('end_reason')})")
                elif k == "timing":
                    print(f"  <- timing  {({kk: vv for kk, vv in m.items() if kk != 'type'})}")
                elif k == "audio":
                    pcm = base64.b64decode(m["chunk"])
                    secs = len(pcm) / 2 / m["sample_rate"]
                    ui["audio"] = f"{secs:.1f}s"
                    print(f"  <- audio   {len(pcm):,} B = {secs:.2f}s @{m['sample_rate']}")
                    if play is not None:
                        # Off the event loop: play.wait() blocks, which would
                        # stall the send loop and corrupt the fps measurement.
                        a = np.frombuffer(pcm, dtype=np.int16)
                        sr = m["sample_rate"]
                        def _spk(buf=a, rate=sr):
                            play.play(buf, rate); play.wait()
                        asyncio.create_task(asyncio.to_thread(_spk))
                elif k == "rejected":
                    top = m.get("top") or []
                    best = f"{top[0][0]} {top[0][1]:.2f}" if top else "?"
                    ui["reject"] = f"{m['reason']}: {best}"
                    print(f"  <- rejected ({m['reason']}) best={best}")
                elif k == "error":
                    ui["err"] = str(m.get("detail"))[:60]
                    print(f"  <- ERROR   {m.get('detail')}")

        rx = asyncio.create_task(receive())
        interrupted = False
        try:
            limit = args.frames if args.frames else int(args.seconds * args.fps)
            next_due = time.perf_counter()
            while sent < limit:
                if cap is not None:
                    ok, bgr = await asyncio.to_thread(cap.read)
                    if not ok:
                        print("!! camera read failed"); break
                    # 1080p is far more than MediaPipe needs and costs real
                    # time in both modes. Downscale to a fixed working width.
                    if bgr.shape[1] > args.width:
                        sc = args.width / bgr.shape[1]
                        bgr = cv2.resize(bgr, (args.width, int(bgr.shape[0] * sc)),
                                         interpolation=cv2.INTER_AREA)
                else:
                    bgr = synthetic_frame(sent)

                t = time.perf_counter() - t_start
                if args.mode == "A":
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    hands, pose = await asyncio.to_thread(
                        extractor.extract, rgb, int(t * 1000))
                    msg = {"type": "landmarks", "seq": sent, "t": t,
                           "hands": {k: (v.tolist() if v is not None else None)
                                     for k, v in hands.items()},
                           "pose": np.nan_to_num(pose).tolist()
                           if pose is not None else [[0, 0]] * N_POSE_UPPER}
                else:
                    ok, enc = cv2.imencode(".jpg", bgr,
                                           [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    msg = {"type": "frame", "seq": sent, "t": t,
                           "jpeg": base64.b64encode(enc.tobytes()).decode()}

                await ws.send(json.dumps(msg))
                sent += 1

                if show:
                    el = time.perf_counter() - t_start
                    ui["fps"] = sent / max(el, 1e-6)
                    view = cv2.flip(bgr, 1)          # mirror for display ONLY
                    view = cv2.copyMakeBorder(view, 0, 116, 0, 0,
                                              cv2.BORDER_CONSTANT, value=(24, 24, 24))
                    hh = view.shape[0] - 116
                    cv2.putText(view, f"mode {args.mode}  {ui['fps']:4.1f} fps  "
                                      f"sent {sent}", (12, hh + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    u = " ".join(ui["utterance"]) or "-"
                    cv2.putText(view, f"utterance: {u[:52]}", (12, hh + 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 240), 2)
                    cv2.putText(view, (ui["sentence"] or "(waiting)")[:58],
                                (12, hh + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (90, 220, 90), 2)
                    tail = ui["err"] or ui["reject"] or (
                        f"audio {ui['audio']}" if ui["audio"] else "")
                    cv2.putText(view, f"q to quit    {tail}", (12, hh + 106),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (80, 80, 240) if ui["err"] else (150, 150, 150), 1)
                    cv2.imshow("harness -- full pipeline (LLM + speech)", view)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        print("\n  quit requested")
                        break
                # Pace to a DEADLINE, not "sleep 1/fps after the work".
                # The old form added the full interval on top of capture and
                # encode, so a 15 fps request delivered ~10 fps -- and frame
                # rate is the dominant factor in segmentation quality.
                next_due += 1.0 / args.fps
                delay = next_due - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_due = time.perf_counter()   # behind: do not accrue debt
            t_sent_done = time.perf_counter()
            await asyncio.sleep(args.linger)
        except KeyboardInterrupt:
            interrupted = True
            print("\n  interrupted -- waiting briefly for any pending audio")
            await asyncio.sleep(min(args.linger, 3.0))
        finally:
            rx.cancel()

    dur = time.perf_counter() - t_start
    if show:
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)
    if cap is not None:
        cap.release()
    if extractor is not None:
        extractor.close()

    print(f"\n{'=' * 58}")
    print(f"  source {args.source}  mode {args.mode}")
    send_dur = (t_sent_done - t_start) if t_sent_done else dur
    print(f"  sent {sent} frames in {send_dur:.1f}s  ({sent/send_dur:.1f} fps "
          f"sending; requested {args.fps:.0f})")
    print(f"  total wall clock {dur:.1f}s (includes {args.linger:.0f}s linger)")
    print(f"  received {got}")
    emitted = got.get("text", 0) > 0
    good = (not emitted) or (args.no_audio or got.get("audio", 0) > 0)
    if emitted:
        print(f"  END-TO-END: {'PASS' if good else 'FAIL'} "
              f"(recognizer emitted and the full chain ran)")
    else:
        print("  END-TO-END: TRANSPORT OK, no signs emitted.")
        print("    Expected for --source synthetic: the gated recognizer only "
              "fires on a\n    confident sign. Set SIGN_MIN_CONFIDENCE=0 on the "
              "server to force emissions.")
    print(f"{'=' * 58}")
    return 0 if good else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["webcam", "synthetic"], default="webcam")
    p.add_argument("--mode", choices=["A", "B"], default="B",
                   help="A = landmarks on client, B = frames to backend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0,
                   help="target capture rate; real rate is capped by "
                        "camera + MediaPipe (~25 fps)")
    p.add_argument("--frames", type=int, default=0,
                   help="stop after N frames; 0 = use --seconds")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="how long to capture (default 60s)")
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--width", type=int, default=640,
                   help="downscale camera frames to this width before processing")
    p.add_argument("--linger", type=float, default=3.0,
                   help="seconds to keep receiving after the last frame")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--expect", type=int, default=0, metavar="N",
                   help="hold the utterance until N signs are recognised, then "
                        "send all of them to the LLM at once, instead of "
                        "ending the sentence on a timeout. 0 (default) = "
                        "timeout mode")
    p.add_argument("--no-preview", action="store_true",
                   help="run headless (no camera window)")
    p.add_argument("--sign-language", default="ase")
    p.add_argument("--output-language", default="en")
    sys.exit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
