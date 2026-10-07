import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")  # reads .env if present; never commit that file

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")


def ensure_dirs() -> None:
    """Create runtime folders if they do not exist."""
    for folder in (UPLOAD_DIR, VECTORSTORE_DIR):
        if folder.exists() and not folder.is_dir():
            raise RuntimeError(
                f"'{folder}' exists but is a file, not a folder. "
                "Delete it and run the program again."
            )
        folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Small helpers to read typed values from environment variables
# ---------------------------------------------------------------------------
def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------
APP_NAME = "BhashaRAG"
APP_SUBTITLE = "Multilingual Document Intelligence Assistant"
APP_TAGLINE = "One Document. Five Languages. One Intelligent Conversation."

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Sanskrit": "sa",
    "Haryanvi": "hr",  # informal code: Haryanvi has no official ISO 639-1 code
}

# Per-language details used by the UI, speech input and text-to-speech.
#   tts_lang    -> language code passed to gTTS
#   stt_locale  -> locale passed to the speech-recognition service
#   fallback    -> True when we use a *different* language's voice/recognizer,
#                  so the UI can tell the user honestly.
LANGUAGE_META = {
    "English": {
        "code": "en", "flag": "🇬🇧",
        "tts_lang": "en", "tts_tld": "co.in", "stt_locale": "en-IN",
        "fallback": False, "note": "",
    },
    "Hindi": {
        "code": "hi", "flag": "🇮🇳",
        "tts_lang": "hi", "tts_tld": "co.in", "stt_locale": "hi-IN",
        "fallback": False, "note": "",
    },
    "Marathi": {
        "code": "mr", "flag": "🇮🇳",
        "tts_lang": "mr", "tts_tld": "co.in", "stt_locale": "mr-IN",
        "fallback": False, "note": "",
    },
    "Sanskrit": {
        "code": "sa", "flag": "🕉️",
        "tts_lang": "hi", "tts_tld": "co.in", "stt_locale": "hi-IN",
        "fallback": True,
        "note": "No dedicated Sanskrit voice/recognizer is used; Hindi is the closest fallback.",
    },
    "Haryanvi": {
        "code": "hr", "flag": "🟠",
        "tts_lang": "hi", "tts_tld": "co.in", "stt_locale": "hi-IN",
        "fallback": True,
        "note": "No native Haryanvi voice/recognizer exists; Hindi is used as the closest fallback.",
    },
}

# ---------------------------------------------------------------------------
# Response modes  (UI label -> internal key used in prompts.py)
# ---------------------------------------------------------------------------
MODES = {
    "Simple": "simple",
    "Hinglish / Roman Regional": "roman",
    "GenZ": "genz",
}

# ---------------------------------------------------------------------------
# Embeddings + chunking + retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# The default MiniLM model only "reads" the first ~256 tokens of a chunk, and Devanagari
# text needs many tokens per character. 500 characters keeps chunks inside that limit.
EMBEDDING_MAX_SEQ_LENGTH = _get_int("EMBEDDING_MAX_SEQ_LENGTH", 256)
CHUNK_SIZE = _get_int("CHUNK_SIZE", 500)          # max characters per chunk
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 100)    # characters shared between neighbours
TOP_K = _get_int("TOP_K", 5)                      # chunks retrieved per question

# Chunks scoring below this cosine similarity are treated as "not relevant".
# Cross-lingual scores are usually lower than same-language scores, so tune this
# while testing (see the testing checklist).
MIN_SIMILARITY_SCORE = _get_float("MIN_SIMILARITY_SCORE", 0.25)

# ---------------------------------------------------------------------------
# LLM  (all four providers are reached through one OpenAI-compatible client)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
# Optional second model, tried once if the main one is overloaded (503) or rate-limited (429).
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()
# The OpenAI client retries automatically with growing pauses before giving up.
LLM_MAX_RETRIES = _get_int("LLM_MAX_RETRIES", 4)
DEFAULT_TEMPERATURE = _get_float("TEMPERATURE", 0.2)

PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",  # local, no API key needed
}
# You can override the URL for any OpenAI-compatible server via LLM_BASE_URL.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or PROVIDER_BASE_URLS.get(LLM_PROVIDER, "")


def llm_config_error() -> str | None:
    """Return a user-friendly message if the LLM is not configured, else None."""
    if LLM_PROVIDER not in PROVIDER_BASE_URLS and not LLM_BASE_URL:
        return (
            f"⚠️ Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Use one of: {', '.join(PROVIDER_BASE_URLS)}."
        )
    if LLM_PROVIDER != "ollama" and not LLM_API_KEY:
        return "⚠️ LLM API key is not configured.\nPlease check your .env file."
    if not LLM_MODEL:
        return "⚠️ LLM_MODEL is not set. Please check your .env file."
    return None