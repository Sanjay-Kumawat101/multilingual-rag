"""
tts.py - text -> speech (independent of the rest of the app).

Uses gTTS (Google Translate's text-to-speech). Honest limits:
  * Needs internet.
  * No Sanskrit or Haryanvi voice exists, so a Hindi voice is used as the closest
    fallback (see config.LANGUAGE_META). The UI shows this note.
  * A Hindi voice reads Roman-script text badly. If the answer contains no
    Devanagari (Hinglish / Roman modes), we use an Indian-English voice instead,
    which pronounces Hinglish far more naturally.
"""
from __future__ import annotations

import io
import re

import config

MAX_SPOKEN_CHARS = 1500
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_MARKDOWN_NOISE = re.compile(r"[*_`#>~|]+")


class TTSError(Exception):
    """Audio could not be generated. The message is safe to show in the UI."""


def _prepare_text(text: str) -> tuple[str, bool]:
    """Remove markdown symbols (they would be read aloud) and limit the length."""
    text = _MARKDOWN_NOISE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SPOKEN_CHARS:
        return text[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0], True
    return text, False


def synthesize(text: str, language: str) -> tuple[bytes, str]:
    """Return (mp3_bytes, note). `note` explains any fallback used ('' if none)."""
    meta = config.LANGUAGE_META.get(language)
    if meta is None:
        raise ValueError(f"Unsupported language '{language}'.")

    speech, truncated = _prepare_text(text)
    if not speech:
        raise TTSError("⚠️ There is no text to read aloud.")

    notes = []
    tts_lang, tld = meta["tts_lang"], meta["tts_tld"]
    if meta["fallback"]:
        notes.append(meta["note"])
    if language != "English" and not _DEVANAGARI.search(speech):
        # Roman-script text: an Indian-English voice reads it better than a Hindi voice.
        tts_lang, tld = "en", "co.in"
        notes.append("Roman-script text is read with an Indian-English voice.")
    if truncated:
        notes.append(f"Only the first {MAX_SPOKEN_CHARS} characters are read aloud.")

    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text=speech, lang=tts_lang, tld=tld).write_to_fp(buffer)
    except Exception as exc:
        raise TTSError(
            "⚠️ Could not generate audio. Text-to-speech needs an internet connection. "
            f"({type(exc).__name__})"
        ) from exc
    return buffer.getvalue(), " ".join(notes)
