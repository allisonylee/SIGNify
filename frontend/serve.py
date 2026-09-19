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
        super().end_headers()

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


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(Handler, directory=str(FRONTEND_DIR)))

    found = "found" if read_api_key() else "MISSING (see frontend/README.md)"
    print(f"[serve] root           : {FRONTEND_DIR}")
    print(f"[serve] elevenlabs key : {found}")
    print(f"[serve] open           : http://127.0.0.1:{port}/speech.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
