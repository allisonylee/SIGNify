"""
Text to speech, behind a provider interface.

ElevenLabsTTS is the real thing. MacSayTTS is a credential-free local fallback
so the audio leg of the pipeline is testable before anyone has an API key --
swapping back is a one-line change, because callers only see `synthesize()`.

Audio on the wire is raw PCM s16le @16k (see config.AUDIO_FORMAT): the harness
can hand it straight to sounddevice and the browser can play it through
WebAudio, with no mp3 decoder on either side.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from dataclasses import dataclass, asdict

from . import config


def _clamp(v, lo, hi, default):
    """Never trust a client value. A page sending speed=50 gets clamped."""
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


@dataclass
class VoiceSettings:
    """
    Plan section 15. NOTE which knobs cost latency:
      stability, speed, voice choice -> FREE
      style > 0                      -> extra compute
      use_speaker_boost              -> 200-500 ms, comparable to our whole
                                        TTS budget. Default OFF.
    There is NO pitch parameter in the ElevenLabs API; pitch comes from which
    voice you pick. Do not add a pitch field here backed by nothing.
    """
    voice_id: str = ""
    stability: float = 0.5          # LOW = expressive, HIGH = flat
    similarity_boost: float = 0.8
    style: float = 0.0              # >0 costs latency
    speed: float = 1.0
    use_speaker_boost: bool = False  # costs 200-500 ms

    @classmethod
    def from_client(cls, d: dict | None) -> "VoiceSettings":
        d = d or {}
        # pitch and age are SLIDER POSITIONS (1..5), not audio parameters. There
        # is no pitch knob in the ElevenLabs API -- pitch comes from which voice
        # you pick -- so they are resolved against the pre-baked grid to a
        # voice_id. A dict lookup, no network, nothing added to TTS latency.
        vid = d.get("voice_id")
        if not vid and ("pitch" in d or "age" in d):
            from .voices import resolve
            vid, how = resolve(d.get("pitch", 3), d.get("age", 3))
            print(f"[voice] pitch={d.get('pitch')} age={d.get('age')} -> {how}")
        return cls(
            voice_id=str(vid or config.ELEVENLABS_VOICE_ID),
            stability=_clamp(d.get("stability", 0.5), 0.0, 1.0, 0.5),
            similarity_boost=_clamp(d.get("similarity_boost", 0.8), 0.0, 1.0, 0.8),
            style=_clamp(d.get("style", 0.0), 0.0, 1.0, 0.0),
            # 0.7-1.2 is the sensible band for speech; the API allows more.
            speed=_clamp(d.get("speed", 1.0), 0.7, 1.2, 1.0),
            use_speaker_boost=bool(d.get("use_speaker_boost", False)),
        )

    def as_dict(self):
        return asdict(self)

    def eleven_payload(self):
        return {"stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style, "speed": self.speed,
                "use_speaker_boost": self.use_speaker_boost}


# Presets matter more than sliders here: someone choosing how they sound to
# hearing people is making an identity choice, not tuning an audio plugin.
PRESETS = {
    "calm":       {"stability": 0.75, "style": 0.0},
    "natural":    {"stability": 0.50, "style": 0.0},   # default
    "expressive": {"stability": 0.30, "style": 0.3},   # slower
}


class TTSProvider:
    name = "base"

    async def synthesize(self, text: str, lang: str = "en",
                         voice: VoiceSettings | None = None,
                         stream=None) -> bytes:
        """-> raw PCM s16le mono @ config.AUDIO_SAMPLE_RATE."""
        raise NotImplementedError

    async def open_stream(self, lang: str = "en",
                          voice: VoiceSettings | None = None):
        """
        Optional. Pay the connection cost BEFORE the text exists, so it overlaps
        the LLM instead of queueing behind it. A provider with no connection to
        make returns None, and `synthesize` then behaves exactly as before.
        """
        return None

    async def list_voices(self) -> list[dict]:
        return []


class MacSayTTS(TTSProvider):
    """macOS `say`. No credentials, real speech, good enough to prove the pipe."""
    name = "macos-say"
    VOICES = {"en": "Samantha", "es": "Monica", "fr": "Thomas", "de": "Anna"}

    async def synthesize(self, text: str, lang: str = "en",
                         voice: VoiceSettings | None = None,
                         stream=None) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.wav"
            cmd = ["say", "-o", str(out),
                   "--data-format=LEI16@%d" % config.AUDIO_SAMPLE_RATE]
            if voice is not None and voice.speed != 1.0:
                cmd += ["-r", str(int(175 * voice.speed))]   # words per minute
            if (v := self.VOICES.get(lang)):
                cmd += ["-v", v]
            cmd.append(text)
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _, err = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"say failed: {err.decode()[:200]}")
            with wave.open(str(out), "rb") as w:
                return w.readframes(w.getnframes())


class ElevenLabsTTS(TTSProvider):
    """
    ElevenLabs Flash v2.5 over the streaming websocket.

    Flash v2.5 is ~75ms model inference across 32 languages. We ask for
    pcm_16000 so no client needs an mp3 decoder.
    """
    name = "elevenlabs-flash-v2.5"

    def __init__(self, api_key=None, voice_id=None, model=None):
        self.api_key = api_key or config.ELEVENLABS_API_KEY
        self.voice_id = voice_id or config.ELEVENLABS_VOICE_ID
        self.model = model or config.ELEVENLABS_MODEL
        self._voices_cache = None
        self.voices_limited = False
        self.voices_error = ""
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

    async def list_voices(self) -> list[dict]:
        """
        Proxy the voice library so the frontend never sees the API key.

        An ElevenLabs key can be SCOPED. A text-to-speech-only key synthesises
        fine but 401s on every REST read, including /v1/voices. When that
        happens we return just the configured voice and flag it, rather than
        failing the request -- the UI can then show a picker with one entry and
        an explanation instead of an error.
        """
        import httpx
        if self._voices_cache is not None:
            return self._voices_cache
        try:
            async with httpx.AsyncClient(timeout=20) as cl:
                r = await cl.get("https://api.elevenlabs.io/v1/voices",
                                 headers={"xi-api-key": self.api_key})
                r.raise_for_status()
                data = r.json()
            self._voices_cache = [
                {"voice_id": v.get("voice_id"), "name": v.get("name"),
                 "labels": v.get("labels", {}), "preview_url": v.get("preview_url")}
                for v in data.get("voices", [])
            ]
            self.voices_limited = False
        except Exception as e:                                   # noqa: BLE001
            self.voices_limited = True
            self.voices_error = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"[tts] /v1/voices unavailable ({self.voices_error}); "
                  f"the API key is probably scoped to TTS only")
            self._voices_cache = [{
                "voice_id": self.voice_id, "name": "configured voice",
                "labels": {}, "preview_url": None,
            }]
        return self._voices_cache

    def _url(self, lang: str, voice: VoiceSettings) -> str:
        vid = voice.voice_id or self.voice_id
        url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{vid}"
               f"/stream-input?model_id={self.model}"
               f"&output_format=pcm_{config.AUDIO_SAMPLE_RATE}")
        # FORCE the language instead of letting Flash v2.5 guess it from the
        # text. `lang` was accepted and then ignored here, so a Spanish sentence
        # was read with whatever phonetics the model inferred. Measured on
        # "Hola, me alegra verte": 38,638 bytes of PCM without the parameter,
        # 49,040 with it -- the same words, pronounced differently.
        # The endpoint documents language_code as a query parameter, and
        # eleven_flash_v2_5 accepts it for en/es/hi (tested 2026-09-19).
        if lang:
            url += f"&language_code={lang}"
        return url

    async def open_stream(self, lang: str = "en",
                          voice: VoiceSettings | None = None):
        """
        Connect now, speak later. MEASURED: the handshake to ElevenLabs costs
        88.8 ms (p50 of 5) -- 38% of a 235 ms synthesize() -- and it does not
        depend on the text. Kicked off the moment the utterance ends, it
        completes while Gemini is still writing, so by the time there IS text
        the socket is already open.
        """
        import websockets
        voice = voice or VoiceSettings(voice_id=self.voice_id)
        try:
            return await websockets.connect(
                self._url(lang, voice),
                additional_headers={"xi-api-key": self.api_key})
        except Exception as e:                       # noqa: BLE001
            print(f"[tts] prewarm failed ({e}); connecting inline instead")
            return None

    async def synthesize(self, text: str, lang: str = "en",
                         voice: VoiceSettings | None = None,
                         stream=None) -> bytes:
        import websockets

        voice = voice or VoiceSettings(voice_id=self.voice_id)
        chunks: list[bytes] = []
        ws = stream or await websockets.connect(
            self._url(lang, voice),
            additional_headers={"xi-api-key": self.api_key})
        try:
            # auto_mode lets the model decide when to commit, which is the
            # right tradeoff for whole sentences.
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": voice.eleven_payload(),
                "generation_config": {"chunk_length_schedule": [50]},
            }))
            await ws.send(json.dumps({"text": text + " ", "flush": True}))
            await ws.send(json.dumps({"text": ""}))       # end of stream
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("audio"):
                    import base64
                    chunks.append(base64.b64decode(msg["audio"]))
                if msg.get("isFinal"):
                    break
        finally:
            # Ours to close whether we opened it or inherited it prewarmed.
            await ws.close()
        return b"".join(chunks)


def get_provider() -> TTSProvider:
    """ElevenLabs when a key exists, otherwise the local fallback."""
    if config.ELEVENLABS_API_KEY:
        try:
            return ElevenLabsTTS()
        except Exception as e:                       # noqa: BLE001
            print(f"[tts] ElevenLabs unavailable ({e}); falling back")
    return MacSayTTS()
