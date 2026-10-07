"""
llm.py - one function that turns (context, question, language, mode)
into an answer.

Gemini, Groq, OpenRouter and Ollama all expose an OpenAI-compatible API, so one
client class (openai.OpenAI) talks to all of them. Switching provider or model
means editing .env only - no code change.
"""
from __future__ import annotations

from functools import lru_cache

import config
from modules import prompts
from modules.retriever import RetrievalResult, format_context

REQUEST_TIMEOUT_SECONDS = 60


class LLMError(Exception):
    """A problem with the LLM call. The message is safe to show in the UI."""


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    # Ollama ignores the key but the client requires a non-empty string.
    api_key = config.LLM_API_KEY or "ollama"
    return OpenAI(
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,   # automatic retries with backoff on 429/5xx
    )


def generate_answer(
    context: str,
    question: str,
    language: str,
    mode: str,
    temperature: float | None = None,
) -> str:
    """Ask the LLM to answer `question` from `context` in `language` and `mode`."""
    problem = config.llm_config_error()
    if problem:
        raise LLMError(problem)

    messages = prompts.build_messages(context, question, language, mode)
    temp = config.DEFAULT_TEMPERATURE if temperature is None else temperature

    models = [config.LLM_MODEL]
    if config.LLM_FALLBACK_MODEL and config.LLM_FALLBACK_MODEL != config.LLM_MODEL:
        models.append(config.LLM_FALLBACK_MODEL)

    last_error: Exception | None = None
    for model in models:
        try:
            response = _get_client().chat.completions.create(
                model=model, messages=messages, temperature=temp,
            )
            break
        except Exception as exc:                  # translate SDK errors into friendly text
            last_error = exc
            if not _is_overloaded(exc):           # e.g. bad key: a second model will not help
                raise LLMError(_friendly_error(exc)) from exc
    else:                                         # every model failed
        raise LLMError(_friendly_error(last_error)) from last_error

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise LLMError("⚠️ The LLM returned an empty answer. Please try again.")
    return text


def answer_question(
    retrieval: RetrievalResult,
    question: str,
    language: str,
    mode: str,
    temperature: float | None = None,
) -> str:
    """Full answer step: if nothing relevant was retrieved, do NOT call the LLM."""
    if not retrieval.has_relevant_context:
        return prompts.get_not_found_message(language, mode)
    return generate_answer(format_context(retrieval.chunks), question, language, mode, temperature)


def _is_overloaded(exc: Exception) -> bool:
    """True for temporary provider-side problems (overload / rate limit)."""
    status = getattr(exc, "status_code", None)
    return type(exc).__name__ in ("InternalServerError", "RateLimitError") or status in (429, 500, 502, 503, 504)


def _friendly_error(exc: Exception) -> str:
    """Map common API failures to clear messages."""
    name = type(exc).__name__
    provider = config.LLM_PROVIDER
    if name == "AuthenticationError" or name == "PermissionDeniedError":
        return "⚠️ The LLM API key was rejected. Please check LLM_API_KEY in your .env file."
    if name == "NotFoundError":
        return f"⚠️ Model '{config.LLM_MODEL}' was not found. Check LLM_MODEL in your .env file."
    if name == "RateLimitError":
        return "⚠️ The LLM rate limit or free quota was reached. Wait a minute and try again."
    if _is_overloaded(exc):
        return ("⚠️ The LLM provider is overloaded right now (a temporary problem on their side). "
                "Please try again in a moment, or set LLM_FALLBACK_MODEL / change LLM_MODEL in .env.")
    if name in ("APIConnectionError", "APITimeoutError"):
        return f"⚠️ Could not reach the LLM provider '{provider}'. Check your internet connection."
    return f"⚠️ LLM request failed ({name}): {exc}"