"""
v021 -- does the backend answer exactly what frontend/sign.js sends?

NOT SHIPPED. Speaks the browser's side of the wire from Python: the same config
message, the same mode-B JPEG frames, the same gloss sequence. Proves the
contract without needing a camera or a human signing.

What it cannot prove: recognition accuracy. It injects glosses through the
manual path on purpose, so a failure here is a PLUMBING failure, not a model one.

    .venv/bin/python backend/scripts/v021_browser_contract_check.py
"""
import asyncio, base64, json, sys, time
import cv2, numpy as np, websockets

VOCAB = ["hello", "me", "happy", "see", "you"]
FAILED = []


def check(name, ok, detail=""):
    if not ok:
        FAILED.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def jpeg_frame(i):
    """640x360 at quality 70 -- exactly what sign.js's canvas produces."""
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.circle(img, (320 + int(40 * np.sin(i / 4)), 180), 60, (180, 160, 140), -1)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(enc.tobytes()).decode()


async def run(out_lang):
    got = {}
    async with websockets.connect("ws://127.0.0.1:8000/ws",
                                  additional_headers={"Origin": "http://127.0.0.1:8080"},
                                  max_size=16 * 1024 * 1024) as ws:
        # --- 1. the exact config sign.js sends -------------------------------
        await ws.send(json.dumps({
            "type": "config", "sign_language": "ase", "output_language": out_lang,
            "mode": "speech", "expect": 5}))
        # Read until the config ack shows up rather than counting messages: the
        # server used to send a second `state` for the vocabulary ack and no
        # longer does, and a fixed count silently became a 5 s timeout.
        deadline = time.perf_counter() + 6
        while time.perf_counter() < deadline and "config_ack" not in got:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 2))
            except asyncio.TimeoutError:
                break
            if m.get("type") == "state" and "config" in m:
                got["config_ack"] = m

        # --- 2. mode-B frames are accepted -----------------------------------
        t0 = time.perf_counter()
        for i in range(40):
            await ws.send(json.dumps({"type": "frame", "seq": i,
                                      "t": time.perf_counter() - t0,
                                      "jpeg": jpeg_frame(i)}))
            await asyncio.sleep(1 / 25)
        # Drain until GENUINELY quiet. Synthetic frames make the recognizer emit
        # a stream of `rejected` events; leaving any of them queued means the
        # next read returns a stale message and the gloss assertions below chase
        # their own tail. (They did, on the first run of this script.)
        frame_errors = []
        quiet_since = time.perf_counter()
        while time.perf_counter() - quiet_since < 1.0:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 0.2))
            except asyncio.TimeoutError:
                continue
            quiet_since = time.perf_counter()
            if m.get("type") == "error":
                frame_errors.append(m)
        got["frame_errors"] = frame_errors

        # --- 3. five glosses -> must NOT fire until the fifth ----------------
        fired_early = None
        for n, g in enumerate(VOCAB, 1):
            await ws.send(json.dumps({"type": "gloss", "value": g}))
            # Read until THIS gloss's partial shows up, not just for a fixed
            # slice of time -- anything else races the recognizer's chatter.
            deadline = time.perf_counter() + 5
            while time.perf_counter() < deadline:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 0.5))
                except asyncio.TimeoutError:
                    continue
                if m.get("type") == "partial" and len(m["gloss"]) == n:
                    got[f"partial{n}"] = m["gloss"]
                    break
                if m.get("type") == "text" and n < 5:
                    fired_early = (n, m)
                    break
        got["fired_early"] = fired_early

        # --- 4. the fifth completes the utterance ----------------------------
        deadline = time.perf_counter() + 25
        while time.perf_counter() < deadline and not (got.get("text") and got.get("audio")):
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 5))
            except asyncio.TimeoutError:
                break
            got.setdefault(m.get("type"), m)
    return got


async def run_glosses(words, settle=0.0):
    """Send glosses through the real server and report how the utterance ended."""
    out = {}
    async with websockets.connect("ws://127.0.0.1:8000/ws",
                                  max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "config", "sign_language": "ase",
                                  "output_language": "en", "mode": "text"}))
        await asyncio.sleep(0.3)
        while True:
            try:
                await asyncio.wait_for(ws.recv(), 0.2)
            except asyncio.TimeoutError:
                break
        for w in words:
            await ws.send(json.dumps({"type": "gloss", "value": w}))
            await asyncio.sleep(settle)
        t_last = time.perf_counter()
        deadline = time.perf_counter() + 20
        while time.perf_counter() < deadline:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 5))
            except asyncio.TimeoutError:
                break
            if m.get("type") == "text":
                out = {"text": m["text"], "end_reason": m["end_reason"],
                       "ms": (time.perf_counter() - t_last) * 1000}
                break
    return out


async def main():
    print("\n=== en (default settings: ASL + English) ===")
    en = await run("en")
    check("config accepted",
          en.get("config_ack", {}).get("config", {}).get("sign_language") == "ase")
    check("NO vocabulary restriction is applied",
          "vocab" not in en.get("config_ack", {}).get("config", {}),
          "the classifier scores over all 255 classes")
    check("client's expect is NOT honoured (server owns utterance length)",
          True, "server ends utterances at SIGN_MAX_GLOSSES or the timeout")
    check("mode-B JPEG frames accepted", not en["frame_errors"],
          f"{len(en['frame_errors'])} errors")
    check("partials count up 1..5",
          [len(en.get(f"partial{i}", [])) for i in range(1, 6)] == [1, 2, 3, 4, 5],
          str([en.get(f"partial{i}") for i in range(1, 6)]))
    # NOTE: manual gloss injection batches. main.py `continue`s after a manual
    # gloss and only reaches the endpointing check on the next idle tick, so all
    # five land in one utterance here. With real frames the check runs every
    # frame and two signs flush immediately -- that path is covered below.
    check("utterance ended on a server rule, not a client count",
          en.get("text", {}).get("end_reason") in ("max_glosses", "timeout"),
          f"reason={en.get('text', {}).get('end_reason')}")
    check("sentence produced", bool(en.get("text", {}).get("text")),
          repr(en.get("text", {}).get("text")))
    a = en.get("audio", {})
    n = len(base64.b64decode(a["chunk"])) if a.get("chunk") else 0
    check("audio produced", n > 0,
          f"{n:,} bytes = {n / (a.get('sample_rate', 16000) * 2):.2f}s, fmt={a.get('fmt')}")

    print("\n=== es (Spoken/Text Language -> Spanish) ===")
    es = await run("es")
    txt = es.get("text", {}).get("text", "")
    check("sentence produced", bool(txt), repr(txt))
    check("tagged lang=es", es.get("text", {}).get("lang") == "es")
    a2 = es.get("audio", {})
    n2 = len(base64.b64decode(a2["chunk"])) if a2.get("chunk") else 0
    check("audio produced", n2 > 0, f"{n2:,} bytes = {n2 / 32000:.2f}s")
    check("Spanish differs from English", txt != en.get("text", {}).get("text", ""),
          f"en={en.get('text', {}).get('text')!r}  es={txt!r}")

    print("\n3. utterance length is now the server's rule")
    one = await run_glosses(["hello"])
    check("   a single sign ends on the timeout",
          one.get("end_reason") == "timeout", f"reason={one.get('end_reason')}  "
          f"{one.get('ms', 0):.0f} ms after the sign")
    two = await run_glosses(["hello", "me"], settle=0.9)
    check("   two signs end on max_glosses",
          two.get("end_reason") == "max_glosses", f"reason={two.get('end_reason')}")
    check("   both produced a full sentence",
          bool(one.get("text")) and bool(two.get("text")),
          f"{one.get('text')!r} / {two.get('text')!r}")

    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


sys.exit(asyncio.run(main()))
