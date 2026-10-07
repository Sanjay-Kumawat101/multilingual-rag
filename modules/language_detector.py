"""
language_detector.py - guess the language of the uploaded document.

Honest limits 
  * langdetect knows English, Hindi and Marathi.
  * It has NO Sanskrit or Haryanvi model. Those texts usually come out as
    Hindi / Marathi / Nepali, so they can never be detected automatically.
  * Therefore we return a `reliable` flag and the UI offers a manual override.

Strategy: look at the SCRIPT first (Devanagari vs Latin), then ask langdetect to
choose only among the languages that make sense for that script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is randomized by default; this makes it repeatable

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")

MIN_LETTERS = 20          # below this we do not even try
RELIABLE_PROB = 0.85      # minimum langdetect probability to call a result reliable

# Function words that are very common in Sanskrit but rare in modern Hindi/Marathi.
# This is only a HINT (a lexical heuristic), never a certain detection.
SANSKRIT_MARKERS = {"अस्ति", "सन्ति", "इति", "कथ्यते", "भवति", "अस्मि", "तस्य", "तस्मात्", "यदा", "तदा", "अपि"}

MANUAL_HINT = "Sanskrit and Haryanvi cannot be auto-detected - use the manual selector if this is wrong."


@dataclass
class DetectionResult:
    language: str | None   # "English" / "Hindi" / "Marathi" (or None if too little text)
    confidence: float      # 0.0 - 1.0
    reliable: bool         # False -> the UI should ask the user to confirm
    script: str            # "Devanagari", "Latin" or "Unknown"
    note: str


def _sample(text: str, size: int = 3000) -> str:
    """Use the start AND the middle of the document (title pages can be misleading)."""
    if len(text) <= size:
        return text
    half = size // 2
    mid = len(text) // 2
    return text[:half] + " " + text[mid: mid + half]


def detect_language(text: str) -> DetectionResult:
    sample = _sample(text)
    dev = len(_DEVANAGARI.findall(sample))
    lat = len(_LATIN.findall(sample))

    if dev + lat < MIN_LETTERS:
        return DetectionResult(None, 0.0, False, "Unknown",
                               "Too little text to detect the language. Please select it manually.")

    try:
        probs = {r.lang: r.prob for r in detect_langs(sample)}
    except LangDetectException:
        probs = {}

    # ---- Devanagari script: Hindi, Marathi (or Sanskrit/Haryanvi/Nepali) ----
    if dev >= lat:
        words = set(re.findall(r"[\u0900-\u097F]+", sample))
        if len(words & SANSKRIT_MARKERS) >= 2:
            return DetectionResult(
                "Sanskrit", 0.5, False, "Devanagari",
                "Contains several Sanskrit function words (keyword heuristic, not a "
                "statistical model). Please confirm.",
            )
        hi, mr = probs.get("hi", 0.0), probs.get("mr", 0.0)
        top_lang, top_prob = max(probs.items(), key=lambda kv: kv[1], default=("", 0.0))
        language = "Marathi" if mr > hi else "Hindi"
        if top_lang in ("hi", "mr") and top_prob >= RELIABLE_PROB:
            return DetectionResult(language, top_prob, True, "Devanagari", MANUAL_HINT)
        return DetectionResult(
            language, top_prob, False, "Devanagari",
            "Devanagari text that could not be confidently classified. "
            "It may be Sanskrit, Haryanvi or Nepali. " + MANUAL_HINT,
        )

    # ---- Latin script: English, or Roman-script Hindi/Marathi/Haryanvi ----
    top_lang, top_prob = max(probs.items(), key=lambda kv: kv[1], default=("", 0.0))
    if top_lang == "en" and top_prob >= RELIABLE_PROB:
        return DetectionResult("English", top_prob, True, "Latin", MANUAL_HINT)
    return DetectionResult(
        "English", top_prob, False, "Latin",
        "Latin-script text that is not clearly English (it may be Roman-script "
        "Hindi/Marathi/Haryanvi). Please confirm the language.",
    )
