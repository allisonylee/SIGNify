"""Settings, read from the environment (or backend/.env)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Load .env. Checks backend/ first, then the repo root -- the keys actually
# live at the repo root, and nothing should depend on remembering which.
REPO_ROOT = ROOT.parent
for _envfile in (ROOT / ".env", REPO_ROOT / ".env"):
    if not _envfile.exists():
        continue
    for _line in _envfile.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# The Kaggle client reads KAGGLE_KEY; the .env calls it KAGGLE_API_TOKEN.
# Bridge it here so nothing downstream has to know about the discrepancy.
if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
    os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Audio wire format. PCM keeps the harness dependency-free; the browser can
# play it through WebAudio just as easily as mp3.
AUDIO_SAMPLE_RATE = 16000
AUDIO_FORMAT = "pcm_s16le_16000"

# Feature spec -- see app/landmarks.py. Single source of truth lives there.
WINDOW_FRAMES = 32          # frames per classified window
WINDOW_SECONDS = 1.5        # wall-clock span the window covers
