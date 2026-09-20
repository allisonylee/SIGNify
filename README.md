# Signify

Live sign language to text and speech. A webcam feed of someone signing becomes
a grammatical sentence, spoken aloud in the language the other person speaks.

```
camera -> MediaPipe landmarks -> encoder -> glosses -> Gemini -> ElevenLabs
```

## Setup on a new machine

**1. Python 3.14 and a virtualenv.**

```
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows: `py -3.14 -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`.

**2. API keys.** `.env` is gitignored and holds secrets, so it is NOT in the
repo. Create it at the project root:

```
GEMINI_API_KEY=...
ELEVENLABS_API_KEY=...
```

Without `GEMINI_API_KEY` the app still runs but joins glosses naively instead of
writing a sentence. Without `ELEVENLABS_API_KEY` it falls back to the macOS
`say` command, which does not exist on Windows or Linux.

**3. Run it.**

```
./run.sh
```

Then open <http://127.0.0.1:8080/index.html>. Ctrl-C stops both servers.

Windows has no `./run.sh`; use WSL or Git Bash, or start the two servers by hand:

```
.venv/bin/python -m uvicorn backend.app.main:app --port 8000 --log-level warning
.venv/bin/python frontend/serve.py
```

## Ports

| Port | Server | Why it exists |
| --- | --- | --- |
| 8000 | `backend/app/main.py` (FastAPI) | recognition websocket, Gemini, ElevenLabs |
| 8080 | `frontend/serve.py` | the static pages, plus `POST /api/scribe-token` |

They must not share a port. The Speech tab posts to the relative path
`/api/scribe-token`, which only `serve.py` implements — open the pages from
`serve.py`, **not** from VS Code Live Preview or any other static server, or the
Listen button reports `token request failed (404)`.

## Requirements that are not negotiable

- **`mediapipe==0.10.35`.** 1.0.x aborts with SIGABRT on macOS arm64 for every
  vision task; 0.9.x fails silently. Every model in `backend/models` was trained
  and fine-tuned against 0.10.35, so changing it risks train/serve skew as well
  as crashes.
- **Not an Intel Mac.** mediapipe 0.10.35 publishes wheels for macOS arm64,
  Linux x86_64, and Windows — but not macOS x86_64.
- **Chrome**, for the Speech tab. Its translation uses Chrome's on-device
  Translator API; other browsers transcribe without translating and say so.

## Custom voices

The Settings sliders (pitch, age) select from 25 voices designed ahead of time
with the ElevenLabs Voice Design API, listed in `backend/app/voice_grid.json`.

**Those voices live in one ElevenLabs account.** The ids in that file only
resolve for whoever owns the key that created them — a different key sees
unknown ids. To rebuild them against another account:

```
.venv/bin/python backend/scripts/v022_design_voice_grid.py            # dry run
.venv/bin/python backend/scripts/v022_design_voice_grid.py --apply    # ~4 min, 25 slots
```

Nothing is generated at request time; a slider change is a dict lookup.

## Layout

| Path | What it is |
| --- | --- |
| `backend/app/` | the shipping server |
| `backend/training/` | ASL training and fine-tuning |
| `backend/trainingISL/`, `backend/trainingCSN/` | the other two languages |
| `backend/scripts/` | verification scripts and the Python test harness — not shipped |
| `backend/models/` | trained checkpoints |
| `frontend/` | the pages |
| `frontend/checks/` | browser tests — not shipped |
| `claude/docs/` | the plan and per-stage write-ups |
