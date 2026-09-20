"""
Static server for the frontend, plus the one endpoint the browser cannot do
itself.

Two reasons this exists instead of opening the HTML off disk:

  1. getUserMedia needs a secure context, and file:// is not one -- the mic is
     simply unavailable without http://localhost.
  2. The ElevenLabs realtime WebSocket authenticates with either an API key
     header or a single-use token in the query string. Browsers cannot set
     WebSocket headers, and the key must never reach the page, so the token has
     to be minted here.

Audio itself does NOT pass through this process: the page opens its own
WebSocket straight to ElevenLabs once it holds a token.

Stdlib only.

    python frontend/serve.py          # then open http://127.0.0.1:8000/
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
TOKEN_URL = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"
TOKEN_PATH = "/api/scribe-token"


def read_api_key() -> str | None:
    """Environment first, then the gitignored .env at the project root."""
    if key := os.environ.get("ELEVENLABS_API_KEY"):
        return key

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, _, value = line.strip().partition("=")
        if name == "ELEVENLABS_API_KEY":
            return value.strip().strip("\"'") or None
    return None


class Handler(SimpleHTTPRequestHandler):
    # Windows resolves .js from the registry, where it is often text/plain --
    # which browsers refuse to execute as an ES module. Pin the types instead.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        # Allow the token to be fetched by a page served from a DIFFERENT local
        # origin. In practice that is VS Code Live Preview on :3000, which
        # people keep opening the pages from; without this the fallback in
        # speech.js is blocked by CORS and the 404 just becomes a CORS error.
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path.partition("?")[0] != TOKEN_PATH:
            self.send_error(404, "no such endpoint")
            return

        key = read_api_key()
        if not key:
            self.reply(503, {"error": "ELEVENLABS_API_KEY is not set -- see frontend/README.md"})
            return

        request = urllib.request.Request(
            TOKEN_URL, method="POST", data=b"", headers={"xi-api-key": key})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            self.reply(e.code, {"error": f"ElevenLabs returned {e.code}: {detail}"})
            return
        except OSError as e:
            self.reply(502, {"error": f"could not reach ElevenLabs: {e}"})
            return

        if not (token := payload.get("token")):
            self.reply(502, {"error": "ElevenLabs response contained no token"})
            return

        self.reply(200, {"token": token})

    def reply(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


# NOT 8000. The sign-language backend (backend/app/main.py, uvicorn) owns 8000,
# and when both wanted it this server lost the race and never started -- which
# showed up in the browser as "token request failed (404)", because the page was
# then being served by something else (VS Code Live Preview) that has no
# /api/scribe-token route. The two servers must not collide.
DEFAULT_PORT = 8080


def main() -> None:
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    try:
        server = ThreadingHTTPServer(
            ("127.0.0.1", port), partial(Handler, directory=str(FRONTEND_DIR)))
    except OSError as e:
        # A bare traceback here is what let the collision go unnoticed.
        print(f"[serve] cannot bind 127.0.0.1:{port} -- {e}")
        print(f"[serve] something else is already on that port. Free it, or run")
        print(f"[serve]   PORT=8081 python frontend/serve.py")
        raise SystemExit(1)

    found = "found" if read_api_key() else "MISSING (see frontend/README.md)"
    print(f"[serve] root           : {FRONTEND_DIR}")
    print(f"[serve] elevenlabs key : {found}")
    print(f"[serve] open           : http://127.0.0.1:{port}/speech.html")
    print(f"[serve] NOTE           : open it from HERE, not from VS Code Live "
          f"Preview -- a static server has no {TOKEN_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
