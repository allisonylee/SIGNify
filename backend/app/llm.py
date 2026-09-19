"""
Gloss sequence -> fluent sentence, via Gemini. SHIPS.

Why this layer exists (plan section 8):
  * ASL grammar is not English grammar. Raw glosses read as "ME STORE GO",
    which is correct ASL and wrong English.
  * It gives output-language breadth almost free -- the same glosses become a
    Spanish or French sentence by changing one word in the prompt.
  * It can absorb recognition errors, because it sees the whole utterance.

LATENCY IS THE WHOLE DESIGN CONSTRAINT. Measured components put TTFT at a few
hundred ms, which is the largest single term after pause detection. So:

  * THINKING IS DISABLED where the model allows it. Reasoning adds seconds.
    NOTE: gemini-3.5-flash-lite REJECTS thinking_config with a 400, so we
    probe once and remember. See _supports_thinking_config.
  * output is capped -- latency scales with tokens produced, and we want one
    short sentence.
  * one client is reused for the process, so no per-request handshake.
  * short utterances BYPASS the model entirely (see should_bypass): one or two
    glosses do not need a language model, and skipping it saves the whole
    round trip on the most common case.

Falls back to a naive join whenever the key is missing or the call fails, so
the speech path never breaks because of this layer.
"""
from __future__ import annotations

import asyncio
import time

from . import config

SYSTEM = (
    "You convert sign-language glosses into one natural sentence.\n"
    "Glosses are UPPERCASE words recognised from a signer, in signing order. "
    "Sign-language grammar differs from spoken grammar: reorder as needed, add "
    "function words, and fix inflection.\n"
    "Rules:\n"
    "- Reply with ONE sentence and nothing else. No preamble, no quotes, no "
    "explanation, no alternatives.\n"
    "- Preserve the meaning. Do not invent content that is not in the glosses.\n"
    "- If the glosses are a single word, just write that word naturally.\n"
    "- Write in the requested language."
)

LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "hi": "Hindi", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian",
}


def naive_text(glosses: list[str], lang: str = "en") -> str:
    """Deterministic fallback. Also what the app used before this layer."""
    if not glosses:
        return ""
    return " ".join(g.replace("_", " ").capitalize() for g in glosses) + "."


def should_bypass(glosses: list[str]) -> bool:
    """
    One or two glosses need no language model. Skipping the round trip here is
    the single biggest latency win on the most common utterance (plan 8.3).
    """
    return len(glosses) <= 2


class GlossTranslator:
    name = "gemini"

    def __init__(self, api_key=None, model=None, max_output_tokens=None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        self.max_output_tokens = max_output_tokens or config.GEMINI_MAX_OUTPUT_TOKENS
        self._client = None
        # gemini-3.x-flash-lite rejects thinking_config outright (400); 3.1
        # accepts it. Probe on first failure rather than hardcoding a table
        # that will rot.
        self._thinking_ok = True
        self.last_ms = 0.0
        self.calls = 0
        self.bypassed = 0
        self.failures = 0
        if self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _config(self, with_thinking: bool):
        from google.genai import types
        kw = dict(system_instruction=SYSTEM,
                  max_output_tokens=self.max_output_tokens,
                  temperature=0.3)
        if with_thinking:
            # Disable reasoning where supported -- it costs seconds of latency.
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kw)

    async def translate(self, glosses: list[str], lang: str = "en") -> str:
        if not glosses:
            return ""
        if should_bypass(glosses):
            self.bypassed += 1
            self.last_ms = 0.0
            return naive_text(glosses, lang)
        if not self.available:
            return naive_text(glosses, lang)

        prompt = (f"Language: {LANG_NAMES.get(lang, lang)}\n"
                  f"Glosses: {' '.join(g.upper() for g in glosses)}")
        t0 = time.perf_counter()
        for attempt in (0, 1):
            try:
                resp = await self._client.aio.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self._config(self._thinking_ok))
                self.last_ms = (time.perf_counter() - t0) * 1000
                self.calls += 1
                text = (getattr(resp, "text", "") or "").strip().strip('"')
                return (text.splitlines()[0].strip() if text
                        else naive_text(glosses, lang))
            except Exception as e:                               # noqa: BLE001
                msg = str(e)
                if attempt == 0 and self._thinking_ok and "INVALID_ARGUMENT" in msg:
                    # This model does not accept thinking_config. Drop it and
                    # retry once; remember for the rest of the process.
                    self._thinking_ok = False
                    print(f"[llm] {self.model} rejects thinking_config; disabling it")
                    continue
                self.last_ms = (time.perf_counter() - t0) * 1000
                self.failures += 1
                print(f"[llm] {type(e).__name__}: {msg[:110]} -- falling back")
                return naive_text(glosses, lang)
        return naive_text(glosses, lang)


_translator: GlossTranslator | None = None


def get_translator() -> GlossTranslator:
    global _translator
    if _translator is None:
        _translator = GlossTranslator()
    return _translator
