# BhashaRAG

**A multilingual document intelligence assistant:** upload a document, ask questions about its contents, and receive answers in one of five supported languages.

BhashaRAG is a Streamlit retrieval-augmented generation (RAG) app. It extracts and chunks a document, creates multilingual embeddings, retrieves relevant passages, and sends those passages to a configurable OpenAI-compatible language model to answer questions with document context.

## Features

- Reads PDF, DOCX, and TXT documents.
- Answers questions in English, Hindi, Marathi, Sanskrit, and Haryanvi.
- Uses a multilingual sentence-transformer model for cross-language retrieval.
- Offers **Simple**, **Hinglish / Roman Regional**, and **GenZ** response modes.
- Shows retrieved source passages and similarity scores on request.
- Supports typed or spoken questions and optional text-to-speech.
- Works with Gemini, Groq, OpenRouter, Ollama, or another OpenAI-compatible endpoint.
- Avoids calling the LLM when no retrieved passage meets the configured relevance threshold.

Language detection is best-effort. Sanskrit and Haryanvi are not reliably detected automatically, so confirm or correct the document language in the app. Voice recognition and text-to-speech use Hindi as a fallback for those languages.

## Requirements

- Python 3.10 or newer
- An LLM provider and its model; Ollama can be used locally
- Internet access on first run to download the embedding model; internet is also needed for hosted LLM, speech recognition, and text-to-speech services

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create your local environment file from the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set `LLM_PROVIDER`, `LLM_MODEL`, and (for hosted providers) `LLM_API_KEY`. Do not commit `.env` or share its contents.

Start the application:

```powershell
streamlit run app.py
```

Streamlit will print a local URL to open in your browser. Upload a document, choose a response language and mode, and ask a question. The first document may take longer to process while the embedding model is downloaded.

### LLM provider configuration

The provider defaults are configured in `config.py`; supported `LLM_PROVIDER` values are `gemini`, `groq`, `openrouter`, and `ollama`.

- **Gemini, Groq, or OpenRouter:** set `LLM_PROVIDER`, choose a model supported by that provider in `LLM_MODEL`, and set the provider key in `LLM_API_KEY`.
- **Ollama:** start Ollama, download a model with `ollama pull <model>`, then set `LLM_PROVIDER=ollama` and `LLM_MODEL` to that model name. An API key is not required.
- **Other OpenAI-compatible endpoint:** set `LLM_BASE_URL` to its API base URL, then configure `LLM_MODEL` and any required `LLM_API_KEY`.

## Configuration

Configuration values can be set in `.env`. Defaults are shown below.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | LLM provider: `gemini`, `groq`, `openrouter`, or `ollama` |
| `LLM_MODEL` | unset | Model name required by the chosen provider |
| `LLM_API_KEY` | unset | Provider API key; not needed for Ollama |
| `LLM_BASE_URL` | provider default | Optional custom OpenAI-compatible API URL |
| `LLM_FALLBACK_MODEL` | unset | Optional model to try after a temporary provider overload or rate limit |
| `LLM_MAX_RETRIES` | `4` | Automatic LLM request retries |
| `TEMPERATURE` | `0.2` | Default answer-generation temperature |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers embedding model |
| `EMBEDDING_MAX_SEQ_LENGTH` | `256` | Maximum embedding model sequence length |
| `CHUNK_SIZE` | `500` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks in characters |
| `TOP_K` | `5` | Number of passages retrieved per question |
| `MIN_SIMILARITY_SCORE` | `0.25` | Minimum cosine similarity for a passage to be used as context |

The app also provides sidebar controls for retrieval count and answer temperature.

## How it works

1. The app extracts text from the uploaded PDF, DOCX, or TXT file.
2. It detects the likely document language and splits the text into overlapping chunks.
3. A multilingual sentence-transformer embeds the chunks, which are indexed in a persistent ChromaDB vector store.
4. Each question is embedded and matched against the document chunks.
5. Relevant passages are sent to the configured LLM, which generates an answer in the selected response language and style.
6. The conversation can show retrieved sources and optionally read an answer aloud.

## Project layout

```text
app.py                  Streamlit user interface
config.py               App, model, language, and runtime configuration
modules/                Document processing, embeddings, retrieval, LLM, and voice helpers
data/uploads/           Local uploaded documents (ignored by Git)
vectorstore/             Local persistent ChromaDB data (ignored by Git)
requirements.txt        Python dependencies
test_*.py                Standalone test and diagnostic scripts
```

## Tests

The repository includes standalone scripts for app logic, ingestion, retrieval, vector-store, and LLM checks. For example:

```powershell
python test_ingest.py
python test_retrieval.py
python test_vector_store.py
python test_app_logic.py
```

Some checks may require network access or provider credentials, depending on the script and configuration.
