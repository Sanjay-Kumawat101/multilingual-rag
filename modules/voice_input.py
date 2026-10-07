"""
voice_input.py - speech -> text (independent of the rest of the app).

The browser records audio (st.audio_input gives us WAV bytes). This module sends it
to Google's free web speech-recognition service through the SpeechRecognition
library and returns the text. Notes for the viva:
  * It needs an internet connection.
  * It is an unofficial free endpoint with usage limits; a production system would
    use a paid, official speech API.
  * Marathi and Hindi are supported. Sanskrit and Haryanvi have no dedicated
    recognizer, so Hindi ('hi-IN') is used as the closest fallback (see config.py).
"""
from __future__ import annotations

import io

import config

CANT_UNDERSTAND = "⚠️ I couldn't understand the audio. Please try again."


class VoiceError(Exception):
    """Speech could not be converted to text. The message is safe to show in the UI."""


def transcribe(audio_bytes: bytes, language: str) -> str:
    """Convert WAV bytes spoken in `language` into text."""
    import speech_recognition as sr

    meta = config.LANGUAGE_META.get(language)
    if meta is None:
        raise ValueError(f"Unsupported language '{language}'.")

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
    except Exception as exc:                       # empty or unreadable recording
        raise VoiceError(CANT_UNDERSTAND) from exc

    try:
        text = recognizer.recognize_google(audio, language=meta["stt_locale"])
    except sr.UnknownValueError as exc:            # audio was heard but not understood
        raise VoiceError(CANT_UNDERSTAND) from exc
    except sr.RequestError as exc:                 # no internet / service unavailable
        raise VoiceError(
            "⚠️ The speech-recognition service could not be reached. "
            "Check your internet connection and try again."
        ) from exc

    text = text.strip()
    if not text:
        raise VoiceError(CANT_UNDERSTAND)
    return text
