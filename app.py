from __future__ import annotations

import streamlit as st

import config
from modules import llm, tts, utils, voice_input
from modules.document_loader import DocumentReadError, EmptyDocumentError, UnsupportedFormatError
from modules.embeddings import EmbeddingError

st.set_page_config(page_title=config.APP_NAME, page_icon="🌐", layout="wide")

# Styling (semi-transparent colours so it works in both light and dark themes)
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.bhasha-header {background: linear-gradient(120deg,#1e3a8a,#0f766e); padding: 1.3rem 1.6rem;
                border-radius: 14px; margin-bottom: 1.2rem;}
.bhasha-header h1 {margin: 0; font-size: 2rem; color: #fff !important; padding: 0;}
.bhasha-header .sub {font-size: 1.05rem; color: #fff; opacity: .95; margin-top: .2rem;}
.bhasha-header .tag {font-size: .9rem; color: #fff; opacity: .8; font-style: italic; margin-top: .4rem;}
div[data-testid="stChatMessage"] {border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem;
                border: 1px solid rgba(128,128,128,.25);}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {background: rgba(59,130,246,.10);}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {background: rgba(16,185,129,.10);}
</style>
""",
    unsafe_allow_html=True,
)

LANGUAGES = list(config.SUPPORTED_LANGUAGES)
QUALITY_NOTES = {
    "Sanskrit": "ℹ️ Sanskrit output quality depends on the chosen LLM and may contain mistakes.",
    "Haryanvi": "ℹ️ Haryanvi has no standard written form or dedicated model here; the LLM imitates it "
                "using Hindi and prompt-based dialect control, so expect imperfections.",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "store": None,          # VectorStore of the current document
    "doc_info": None,       # utils.DocumentInfo
    "file_key": None,       # identifies the uploaded file so we only process it once
    "doc_error": None,
    "chat": [],             # list of dicts made by utils.run_rag
    "mic_counter": 0,       # changing the mic widget's key resets (empties) it
    "voice_notice": None,
}.items():
    st.session_state.setdefault(key, default)
state = st.session_state

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K chunks", 1, 10, min(max(config.TOP_K, 1), 10))
    temperature = st.slider("Temperature", 0.0, 1.0, float(config.DEFAULT_TEMPERATURE), 0.1)
    show_context = st.checkbox("Show Retrieved Context", value=False)
    enable_voice = st.checkbox("Enable Voice", value=True)
    enable_tts = st.checkbox("Enable Read Aloud", value=True)

    st.divider()
    st.subheader("Supported Languages")
    for name in LANGUAGES:
        st.markdown(f"{config.LANGUAGE_META[name]['flag']} {name}")

    st.divider()
    problem = config.llm_config_error()
    if problem:
        st.warning(problem)
    else:
        st.caption(f"🧠 LLM: {config.LLM_PROVIDER} / {config.LLM_MODEL}")
    st.caption(f"🔎 Embeddings: {config.EMBEDDING_MODEL.split('/')[-1]}")
    if st.button("🗑️ Clear chat"):
        state.chat = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="bhasha-header"><h1>🌐 {config.APP_NAME}</h1>
<div class="sub">{config.APP_SUBTITLE}</div><div class="tag">{config.APP_TAGLINE}</div></div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 1. Upload + processing
# ---------------------------------------------------------------------------
st.subheader("1. Upload Document")
uploaded = st.file_uploader("Drag & drop a PDF, DOCX or TXT file", type=["pdf", "docx", "txt"])

if uploaded is None:
    if state.file_key is not None:                    # the user removed the file
        state.store = state.doc_info = state.file_key = state.doc_error = None
        state.chat = []
else:
    file_key = f"{uploaded.name}:{uploaded.size}"
    if file_key != state.file_key:                    # a NEW document
        state.file_key = file_key
        state.store = state.doc_info = state.doc_error = None
        state.chat = []                               # old answers belong to the old document
        try:
            with st.spinner("Reading, chunking and indexing the document "
                            "(the first run also loads the embedding model)..."):
                state.store, state.doc_info = utils.process_document(uploaded.name, uploaded.getvalue())
            state["doc_lang_override"] = state.doc_info.language
        except (UnsupportedFormatError, EmptyDocumentError, DocumentReadError, EmbeddingError) as exc:
            state.doc_error = str(exc)

if state.doc_error:
    st.error(state.doc_error)

info = state.doc_info
if info is not None:
    st.success("✅ Document processed successfully!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks created", info.chunks)
    c2.metric("Pages", info.pages if info.pages is not None else "–")
    chosen = c3.selectbox("Document language", LANGUAGES, key="doc_lang_override",
                          format_func=utils.flag_label)
    if chosen != info.language:                       # manual correction
        utils.relabel_language(state.store, info, chosen)
    st.markdown(f"**Document:** {info.name} &nbsp;|&nbsp; **Detected Document Language:** "
                f"{utils.flag_label(info.language)} &nbsp;|&nbsp; **Vector index:** ready")
    if not info.language_reliable:
        st.warning(f"Language detection is not certain. {info.language_note}")

st.divider()

# ---------------------------------------------------------------------------
# 2. Response language + mode  (these are independent of the document language)
# ---------------------------------------------------------------------------
st.subheader("2. Response Language & Mode")
col_lang, col_mode = st.columns([1, 2])
language = col_lang.selectbox("Choose Response Language", LANGUAGES, format_func=utils.flag_label)
mode = col_mode.radio("Response Mode", list(config.MODES), horizontal=True)
if language in QUALITY_NOTES:
    st.caption(QUALITY_NOTES[language])


# ---------------------------------------------------------------------------
# 3. Ask
# ---------------------------------------------------------------------------
def handle_question(question: str, from_voice: bool = False) -> None:
    """Validate, run the RAG pipeline, and append the result to the chat history."""
    question = question.strip()
    if state.store is None:
        st.warning("⚠️ Please upload a document first.")
        return
    if not question:
        st.warning("⚠️ Please type or speak a question first.")
        return
    try:
        with st.spinner("Searching the document and writing the answer..."):
            entry = utils.run_rag(state.store, question, language, mode, top_k, temperature)
    except (llm.LLMError, EmbeddingError) as exc:
        st.error(str(exc))
        return
    entry["voice"] = from_voice
    state.chat.append(entry)


st.subheader("3. Ask your question")
with st.form("ask_form", clear_on_submit=True):
    typed = st.text_input("Ask your question", placeholder="Type your question here...")
    submitted = st.form_submit_button("Ask Question", type="primary")
if submitted:
    handle_question(typed)

if enable_voice:
    if state.voice_notice:
        st.warning(state.voice_notice)
        state.voice_notice = None
    state.setdefault("speech_lang", language)
    col_mic, col_speech = st.columns([3, 2])
    speech_lang = col_speech.selectbox("Speak in", LANGUAGES, key="speech_lang", format_func=utils.flag_label)
    if config.LANGUAGE_META[speech_lang]["fallback"]:
        col_speech.caption(f"ℹ️ {config.LANGUAGE_META[speech_lang]['note']}")
    audio = col_mic.audio_input("🎤 Speak Question", key=f"mic_{state.mic_counter}")
    if audio is not None:
        state.mic_counter += 1                       # empties the widget on the next run
        try:
            with st.spinner("Recognizing speech..."):
                recognized = voice_input.transcribe(audio.getvalue(), speech_lang)
        except voice_input.VoiceError as exc:
            state.voice_notice = str(exc)
            st.rerun()
        else:
            st.info(f"🎤 **Recognized Question:** {recognized}")
            handle_question(recognized, from_voice=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Conversation (newest first, because the input box is above)
# ---------------------------------------------------------------------------
st.subheader("4. Answers")
if not state.chat:
    st.caption("Your conversation will appear here.")

for idx in range(len(state.chat) - 1, -1, -1):
    entry = state.chat[idx]

    with st.chat_message("user", avatar="👤"):
        st.markdown("**User**" + ("  🎤" if entry.get("voice") else ""))
        st.write(entry["question"])

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("**AI Assistant**")
        st.markdown(entry["answer"])
        caption = (f"{utils.flag_label(entry['language'])} · {entry['mode']} · "
                   f"best similarity {entry['best_score']:.2f}")
        if not entry["used_llm"]:
            caption += " · no relevant text found, LLM not called"
        st.caption(caption)

        if enable_tts and st.button("🔊 Read Aloud", key=f"tts_{idx}"):
            try:
                with st.spinner("Generating audio..."):
                    mp3, note = tts.synthesize(entry["answer"], entry["language"])
            except tts.TTSError as exc:
                st.error(str(exc))
            else:
                st.audio(mp3, format="audio/mp3", autoplay=True)
                if note:
                    st.caption(f"ℹ️ {note}")

        if show_context:
            with st.expander("📚 Retrieved Sources", expanded=True):
                if not entry["sources"]:
                    st.write(f"No chunk reached the similarity threshold ({config.MIN_SIMILARITY_SCORE}); "
                             f"best score was {entry['best_score']:.2f}.")
                for src in entry["sources"]:
                    st.markdown(f"**Source {src['rank']}** · {utils.describe_source(src['metadata'])}")
                    st.markdown(f"Similarity Score: `{src['score']:.2f}`")
                    st.text(src["text"])   # st.text: document text is shown as plain text, never as markdown
