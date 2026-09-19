#!/usr/bin/env bash
# Start both servers for the demo and wait. Ctrl-C stops both.
#
#   ./run.sh              the app
#   ./run.sh --harness    the app, then the Python harness instead of the browser
#
# Two servers, two ports, on purpose:
#   :8000  FastAPI  -- sign recognition websocket, Gemini, ElevenLabs TTS
#   :8080  serve.py -- the static pages, plus POST /api/scribe-token for the
#                      Speech tab (a plain static server cannot do that route,
#                      which is what "token request failed (404)" meant)
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || { echo "no .venv here -- run from the project root"; exit 1; }

for port in 8000 8080; do
  if lsof -tiTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "!! port $port is already in use. Stop it first:"
    echo "     kill \$(lsof -tiTCP:$port -sTCP:LISTEN)"
    exit 1
  fi
done

mkdir -p backend/outputs/logs
pids=()
cleanup() { echo; echo "[run] stopping"; for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "[run] backend  :8000 ..."
SIGN_UTTERANCE_TIMEOUT_S=0.8 $PY -u -m uvicorn backend.app.main:app \
  --port 8000 --log-level warning > backend/outputs/logs/server.log 2>&1 &
pids+=($!)

echo "[run] frontend :8080 ..."
$PY -u frontend/serve.py > backend/outputs/logs/frontend.log 2>&1 &
pids+=($!)

for i in $(seq 1 90); do
  curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8000/health 2>/dev/null \
    && curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/index.html 2>/dev/null \
    && break
  sleep 0.5
done

if ! curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8000/health 2>/dev/null; then
  echo "!! backend never came up. Last lines:"; tail -20 backend/outputs/logs/server.log; exit 1
fi
if ! curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/index.html 2>/dev/null; then
  echo "!! frontend never came up. Last lines:"; tail -20 backend/outputs/logs/frontend.log; exit 1
fi

echo
echo "  Sign tab     http://127.0.0.1:8080/index.html     <- the demo"
echo "  Speech tab   http://127.0.0.1:8080/speech.html"
echo "  Settings     http://127.0.0.1:8080/settings.html"
echo
echo "  Sign tab: the camera starts by itself. Sign  hello me happy see you"
echo "           Speak toggles the voice on/off; text appears either way."
echo "  logs: backend/outputs/logs/{server,frontend}.log"
echo

if [ "${1:-}" = "--harness" ]; then
  echo "[run] harness instead of the browser (Ctrl-C to stop everything)"
  $PY backend/scripts/harness_client.py --source webcam --mode B --seconds 90 \
    --vocab hello,me,happy,see,you --expect 5 || true
else
  echo "[run] Ctrl-C to stop both."
  wait
fi
