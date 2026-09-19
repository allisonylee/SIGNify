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
# RE-MEASURED against real idle footage (the half that was unmeasured before).
# Idle and signing motion overlap almost entirely:
#          IDLE   SIGNING
#   p50  0.0275    0.0387
#   p90  0.1532    0.1877
# At 0.008 idle tripped the gate 89% of the time, so it NEVER closed and every
# segment ran to max_segment_s -- a fixed 2.5 s wait before anything was
# classified. 0.025 is the best available separation (53% / 36%), which is
# still poor: motion energy cannot reliably distinguish these. The real fix is
# to gate on the model's own P(__REST__), which separates them cleanly.
REC_MOTION_THRESHOLD = float(os.environ.get("SIGN_MOTION_THRESHOLD", "0.025"))
# Lowered from 0.35 once the __REST__ class existed. The floor was originally
# the ONLY thing suppressing false positives; now the rest class does that job
# categorically, so the floor can be permissive. 0.35 was rejecting most real
# signs from a non-fluent signer.
REC_MIN_CONFIDENCE = float(os.environ.get("SIGN_MIN_CONFIDENCE", "0.20"))
REC_MIN_MARGIN = float(os.environ.get("SIGN_MIN_MARGIN", "0.05"))
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
# Because the gate closes unreliably, this cap is what actually ends most
# segments -- so it sets the felt latency. 2.5 s meant a 2.5 s wait before
# EVERY word. 1.2 s is about one sign's length.
REC_MAX_SEGMENT_S = float(os.environ.get("SIGN_MAX_SEGMENT_S", "1.2"))
# Reject a segment as rest only when the model is CONFIDENT it is rest.
# The veto used to be categorical -- argmax == __REST__ threw the segment away
# even at P=0.43, which silently ate real signs whenever rest merely edged out
# the right answer. Observed live: rejections at 0.43/0.49/0.66 alongside
# genuine ones at 0.95.
REC_REST_THRESHOLD = float(os.environ.get("SIGN_REST_THRESHOLD", "0.75"))
# "rest"   -- segment using the model's P(__REST__)   (default; see
#             RestGatedRecognizer for why motion gating was abandoned)
# "motion" -- the old motion-energy gate, kept for comparison
# HOW LONG AFTER A SIGN ENDS BEFORE ITS WORD APPEARS.
# RestGatedRecognizer scores the window every `stride` frames and needs
# `exit_confirm` consecutive rest verdicts to close a sign. At the measured
# 24 fps that is stride/24 * exit_confirm seconds of pure latency on EVERY sign:
#   stride 3, confirm 2 -> ~250 ms   (default; what the demo was tuned at)
#   stride 2, confirm 2 -> ~167 ms   (faster, but a brief mid-sign pause is
#                                     more likely to be read as the end)
# Exposed so this can be A/B'd against real signing without editing code.
REC_STRIDE = int(os.environ.get("SIGN_STRIDE", "3"))
REC_EXIT_CONFIRM = int(os.environ.get("SIGN_EXIT_CONFIRM", "2"))

REC_GATE = os.environ.get("SIGN_GATE", "rest")
# TEST ONLY. The __REST__ veto is categorical, so no confidence threshold can
# bypass it -- which makes the downstream chain (utterance -> LLM -> TTS)
# untestable with synthetic input, because synthetic motion correctly
# classifies as rest. Never set this in a real run.
REC_IGNORE_REST = os.environ.get("SIGN_IGNORE_REST", "") == "1"

# How long with no new sign before the utterance is considered finished and
# sent to the LLM. This is the ONLY thing that decides where a sentence ends.
# Longer  -> more words per sentence, better grammar, later speech.
# Shorter -> snappier speech, more fragments.
# Lowered from 2.0 once keep_alive() existed. Before that the timeout had to
# cover (pause + next sign duration + quiet), so it could not be short. Now it
# only counts GENUINE stillness, so it only has to exceed the longest still
# pause WITHIN a sentence -- about 0.8 s in practice.
#   shorter -> speech sooner, but a long thinking pause splits the sentence
#   longer  -> safer grouping, later speech
UTTERANCE_TIMEOUT_S = float(os.environ.get("SIGN_UTTERANCE_TIMEOUT_S", "1.2"))

# HOW MANY SIGNS MAKE AN UTTERANCE BEFORE IT IS SENT.
# 2 means the sentence goes to the LLM the instant a second sign lands, and a
# lone sign goes after UTTERANCE_TIMEOUT_S of quiet -- so speech follows one or
# two words instead of waiting for a whole phrase. Raise it to hold longer
# utterances together at the cost of waiting for them.
UTTERANCE_MAX_GLOSSES = int(os.environ.get("SIGN_MAX_GLOSSES", "2"))

# Feature spec -- see app/landmarks.py. Single source of truth lives there.
WINDOW_FRAMES = 32          # frames per classified window
WINDOW_SECONDS = 1.5        # wall-clock span the window covers
