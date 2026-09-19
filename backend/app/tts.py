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

from . import config


class TTSProvider:
    name = "base"

    async def synthesize(self, text: str, lang: str = "en") -> bytes:
        """-> raw PCM s16le mono @ config.AUDIO_SAMPLE_RATE."""
        raise NotImplementedError


class MacSayTTS(TTSProvider):
    """macOS `say`. No credentials, real speech, good enough to prove the pipe."""
    name = "macos-say"
    VOICES = {"en": "Samantha", "es": "Monica", "fr": "Thomas", "de": "Anna"}

    async def synthesize(self, text: str, lang: str = "en") -> bytes:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.wav"
            cmd = ["say", "-o", str(out),
                   "--data-format=LEI16@%d" % config.AUDIO_SAMPLE_RATE]
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
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

    async def synthesize(self, text: str, lang: str = "en") -> bytes:
        import websockets

        url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
               f"/stream-input?model_id={self.model}"
               f"&output_format=pcm_{config.AUDIO_SAMPLE_RATE}")
        chunks: list[bytes] = []
        async with websockets.connect(
                url, additional_headers={"xi-api-key": self.api_key}) as ws:
            # auto_mode lets the model decide when to commit, which is the
            # right tradeoff for whole sentences.
            await ws.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
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
        return b"".join(chunks)


def get_provider() -> TTSProvider:
    """ElevenLabs when a key exists, otherwise the local fallback."""
    if config.ELEVENLABS_API_KEY:
        try:
            return ElevenLabsTTS()
        except Exception as e:                       # noqa: BLE001
            print(f"[tts] ElevenLabs unavailable ({e}); falling back")
    return MacSayTTS()
