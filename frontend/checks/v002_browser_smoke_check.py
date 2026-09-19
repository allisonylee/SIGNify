"""
v002 -- does the sign page actually work in a real browser?

NOT SHIPPED. Drives frontend/index.html in headless Chrome over the DevTools
protocol, with Chrome's fake camera (--use-fake-device-for-media-stream) so the
whole capture path runs without a human or a webcam.

Deliberately does NOT pass --autoplay-policy=no-user-gesture-required. That flag
hides the real defect this script exists to catch: an AudioContext created
outside a user gesture starts SUSPENDED and plays nothing, silently. The click
below is sent with userGesture=true, which is what a real click is.

Needs both servers up:
    python frontend/serve.py                      # :8080
    uvicorn backend.app.main:app --port 8000      # :8000

    .venv/bin/python frontend/checks/v002_browser_smoke_check.py
"""
import asyncio, json, subprocess, tempfile, time, urllib.request, sys, shutil
import websockets

PORT = 9222
PROFILE = tempfile.mkdtemp(prefix="signagram-chrome-")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL = "http://127.0.0.1:8080/index.html"

proc = subprocess.Popen([
    CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
    f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check",
    "--use-fake-ui-for-media-stream",        # auto-grant the camera prompt
    "--use-fake-device-for-media-stream",    # synthetic 640x480 video source
    "--disable-gpu", "about:blank",
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ws_url():
    for _ in range(60):
        try:
            d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
            for t in d:
                if t.get("type") == "page":
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("chrome did not expose a debugger target")

FAILED, LOGS = [], []
def check(name, ok, detail=""):
    if not ok: FAILED.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

async def main():
    url = ws_url()
    async with websockets.connect(url, max_size=32*1024*1024) as ws:
        n = [0]
        async def send(method, params=None, wait=True):
            n[0] += 1; mid = n[0]
            await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            if not wait: return None
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 25))
                if m.get("id") == mid: return m
                collect(m)
        def collect(m):
            meth = m.get("method")
            if meth == "Runtime.consoleAPICalled":
                args = " ".join(str(a.get("value", a.get("description",""))) for a in m["params"]["args"])
                LOGS.append((m["params"]["type"], args))
            elif meth == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                LOGS.append(("exception", d.get("text","") + " " + str(d.get("exception",{}).get("description",""))[:300]))
            elif meth == "Log.entryAdded":
                e = m["params"]["entry"]
                LOGS.append((e["level"], e.get("text","")))

        await send("Runtime.enable"); await send("Log.enable"); await send("Page.enable")
        await send("Page.addScriptToEvaluateOnNewDocument", {"source":
            "const _AC = window.AudioContext;"
            "window.AudioContext = function(...a){const c=new _AC(...a);"
            "window.AudioContext.__last=c;return c;};"
            "window.AudioContext.prototype=_AC.prototype;"
            "const _WS = window.WebSocket;"
            "window.WebSocket = function(...a){const s=new _WS(...a);"
            "window.WebSocket.__last=s;return s;};"
            "window.WebSocket.prototype=_WS.prototype;"
            "Object.assign(window.WebSocket,{OPEN:1,CONNECTING:0,CLOSING:2,CLOSED:3});"})
        await send("Page.navigate", {"url": URL})
        await asyncio.sleep(3.0)
        # drain events
        try:
            while True: collect(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
        except asyncio.TimeoutError: pass

        async def ev(expr, gesture=False):
            r = await send("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                                "awaitPromise": True, "userGesture": gesture})
            return r["result"]["result"].get("value")

        live = ("(()=>{const v=document.querySelector('.sign-video');"
                "return !!(v&&v.srcObject&&v.srcObject.getVideoTracks()[0]"
                "&&v.srcObject.getVideoTracks()[0].readyState==='live')})()")
        pressed = "document.querySelector('.toggle').getAttribute('aria-pressed')"
        actx = "(()=>{const c=window.AudioContext.__last;return c?c.state:'none';})()"
        stat = "document.querySelector('.sign-status').textContent"

        # --- NO CLICK YET: the camera is supposed to come up by itself --------
        check("sign.js ran (video injected)", await ev("!!document.querySelector('.sign-video')"))
        check("status line injected", await ev("!!document.querySelector('.sign-status')"))
        ph = await ev("document.getElementById('placeholder').textContent")
        check("placeholder just says text will appear",
              "appear here" in (ph or "") and "hello me happy" not in (ph or ""), repr(ph))
        check("CAMERA LIVE WITH NO CLICK", await ev(live))
        dims = await ev("(()=>{const v=document.querySelector('.sign-video');return v?`${v.videoWidth}x${v.videoHeight}`:''})()")
        check("video has real dimensions", bool(dims) and dims != "0x0", dims)
        check("stage marked live", await ev("document.querySelector('.stage').classList.contains('live')"))
        check("Speak starts OFF", await ev(pressed) == "false", f"aria-pressed={await ev(pressed)!r}")
        check("NO caption under the camera", (await ev(stat) or "") == "", repr(await ev(stat)))
        check("empty caption takes no vertical space",
              await ev("(()=>{const e=document.querySelector('.sign-status');"
                       "return e.getBoundingClientRect().height;})()") == 0)
        check("no AudioContext before any gesture", await ev(actx) == "none",
              f"state={await ev(actx)!r}")
        check("no exception on load", not [l for l in LOGS if l[0] == "exception"],
              str([l[1][:140] for l in LOGS if l[0]=='exception']))

        # --- Speak ON: audio only. The camera must not be touched. ------------
        await ev("document.querySelector('.toggle').click()", gesture=True)
        await asyncio.sleep(2.0)
        try:
            while True: collect(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
        except asyncio.TimeoutError: pass

        check("Speak reads pressed", await ev(pressed) == "true")
        check("AudioContext RUNNING after the gesture", await ev(actx) == "running",
              f"state={await ev(actx)!r}")
        check("CAMERA STILL LIVE after Speak ON", await ev(live))
        check("still no caption after Speak ON", (await ev(stat) or "") == "", repr(await ev(stat)))

        # --- Speak OFF: still must not touch the camera -----------------------
        await ev("document.querySelector('.toggle').click()", gesture=True)
        await asyncio.sleep(2.0)
        check("Speak reads unpressed", await ev(pressed) == "false")
        check("CAMERA STILL LIVE after Speak OFF", await ev(live))
        check("still no caption after Speak OFF", (await ev(stat) or "") == "", repr(await ev(stat)))

        # --- sentences must ACCUMULATE, not replace --------------------------
        # Driven through the page's own message handler with two synthetic
        # `text` frames, because the fake camera cannot sign.
        await ev("window.dispatchEvent(new Event('noop'))")
        before = await ev("document.getElementById('committed').textContent")
        check("transcript starts empty", (before or "") == "", repr(before))

        # Feed the page's OWN socket two server `text` frames. This exercises the
        # real handleMessage path -- no test hook in shipping code -- because the
        # fake camera cannot actually sign.
        feed = ("(t)=>window.WebSocket.__last.dispatchEvent("
                "new MessageEvent('message',{data:JSON.stringify({type:'text',text:t})}))")
        await ev(f"({feed})('Hello, I am happy to see you.')")
        first = await ev("document.getElementById('committed').textContent")
        check("first sentence lands", first == "Hello, I am happy to see you.", repr(first))
        check("placeholder hidden once there is text",
              await ev("document.getElementById('placeholder').hidden"))

        await ev(f"({feed})('Nice to meet you.')")
        both = await ev("document.getElementById('committed').textContent")
        check("SECOND SENTENCE APPENDS, first not wiped",
              both == "Hello, I am happy to see you. Nice to meet you.", repr(both))

        await ev(f"({feed})('')")
        check("an empty sentence changes nothing",
              await ev("document.getElementById('committed').textContent") == both)
        check("no exception through both toggles", not [l for l in LOGS if l[0] == "exception"],
              str([l[1][:140] for l in LOGS if l[0]=='exception']))

    print("\n  --- browser console ---")
    for lvl, txt in LOGS[:25]:
        print(f"    {lvl:9s} {txt[:150]}")
    if not LOGS: print("    (empty)")
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0

try:
    code = asyncio.run(main())
finally:
    proc.terminate()
    shutil.rmtree(PROFILE, ignore_errors=True)
sys.exit(code)
