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
# Model is benchmarked, not assumed -- see backend/outputs/llm_latency.txt
# MEASURED, not assumed (backend/outputs/llm_latency.txt):
#   gemini-3.5-flash-lite  566 ms median  <- chosen; rejects thinking_config
#   gemini-3.1-flash-lite  659 ms median     accepts thinking_budget=0
#   gemini-2.5-flash-lite  404 -- retired for new projects
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "60"))
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Audio wire format. PCM keeps the harness dependency-free; the browser can
# play it through WebAudio just as easily as mp3.
AUDIO_SAMPLE_RATE = 16000
AUDIO_FORMAT = "pcm_s16le_16000"

# Recognizer guards (Stage 3). Tunable without a code edit -- useful for
# testing the path end to end and for tuning against a real signer.
# CALIBRATED against 24,000 three-frame windows drawn from real GISLR signing
# (backend/outputs/motion_calibration.txt). Percentiles of signing motion:
#   p5 0.0048   p10 0.0069   p25 0.0139   p50 0.0316
# A threshold of 0.020 -- the arbitrary value used before measuring -- marks
# 35.6% of genuine signing frames as "idle", which chops segments mid-sign.
# 0.008 misses ~12%. The false-positive side needs recorded REST footage to
# tune (scripts/v010_record_rest.py); raise this if idle motion trips the gate.
REC_MOTION_THRESHOLD = float(os.environ.get("SIGN_MOTION_THRESHOLD", "0.008"))
REC_MIN_CONFIDENCE = float(os.environ.get("SIGN_MIN_CONFIDENCE", "0.35"))
REC_MIN_MARGIN = float(os.environ.get("SIGN_MIN_MARGIN", "0.10"))
REC_DEBOUNCE_S = float(os.environ.get("SIGN_DEBOUNCE_S", "1.0"))
# Stillness that ends a sign, in SECONDS (not frames -- a frame count makes the
# required pause depend on machine speed). Swept in backend/outputs/
# pause_tolerance.txt against VARIABLE pauses, which is what a human produces.
# Swept against VARIABLE pauses (0.2-1.0 s), fresh recognizer per trial,
# 6 signs x 8 seeds. Detected out of 6:
#        10fps  15fps  30fps
#   0.10   4.6    5.1    5.9
#   0.20   4.5    5.1    5.5
#   0.50   4.5    4.5    4.8
# FRAME RATE DOMINATES the threshold. Keep this low and push fps up.
REC_QUIET_SECONDS = float(os.environ.get("SIGN_QUIET_SECONDS", "0.15"))
# Force a segment boundary after this long. Real signs are short; sustained
# motion past it means a pause was missed and signs are merging.
REC_MAX_SEGMENT_S = float(os.environ.get("SIGN_MAX_SEGMENT_S", "2.5"))
# TEST ONLY. The __REST__ veto is categorical, so no confidence threshold can
# bypass it -- which makes the downstream chain (utterance -> LLM -> TTS)
# untestable with synthetic input, because synthetic motion correctly
# classifies as rest. Never set this in a real run.
REC_IGNORE_REST = os.environ.get("SIGN_IGNORE_REST", "") == "1"

# How long with no new sign before the utterance is considered finished and
# sent to the LLM. This is the ONLY thing that decides where a sentence ends.
# Longer  -> more words per sentence, better grammar, later speech.
# Shorter -> snappier speech, more fragments.
UTTERANCE_TIMEOUT_S = float(os.environ.get("SIGN_UTTERANCE_TIMEOUT_S", "2.0"))

# Feature spec -- see app/landmarks.py. Single source of truth lives there.
WINDOW_FRAMES = 32          # frames per classified window
WINDOW_SECONDS = 1.5        # wall-clock span the window covers
