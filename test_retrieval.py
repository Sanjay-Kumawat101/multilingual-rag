"""
test_retrieval.py - the real cross-lingual retrieval test (needs internet on first run
to download the embedding model).

One ENGLISH document is indexed, then questions in English, Hindi, Marathi and
Roman-Hindi are asked. For each question we check whether the top-1 chunk is the
expected one and print the similarity scores.

Run:  python test_retrieval.py
"""
import config
from modules.retriever import Retriever, index_chunks
from modules.text_processor import build_chunks
from modules.document_loader import PageText

DOCUMENT = """Artificial Intelligence is a branch of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence.

Machine learning is a subset of AI in which computers learn patterns from data instead of being explicitly programmed.

Deep learning uses neural networks with many layers to learn complex patterns such as images and speech.

A vector database stores embeddings, which are lists of numbers that represent the meaning of text, and finds similar items quickly.

Natural language processing allows computers to understand, translate and generate human language.

The Taj Mahal is a marble mausoleum in Agra, India, built by Emperor Shah Jahan in memory of his wife Mumtaz Mahal."""

# (question, text that the top-1 chunk should contain)
TESTS = [
    ("What is machine learning?",              "Machine learning"),
    ("आर्टिफिशियल इंटेलिजेंस क्या है?",          "Artificial Intelligence is"),
    ("कृत्रिम बुद्धिमत्ता म्हणजे काय?",           "Artificial Intelligence is"),
    ("मशीन लर्निंग म्हणजे काय?",                 "Machine learning"),
    ("वेक्टर डेटाबेस क्या होता है?",             "vector database"),
    ("ताजमहल किसने बनवाया था?",                 "Taj Mahal"),
    ("AI kya hai?",                             "Artificial Intelligence is"),
]
NEGATIVE = "What is the capital of France?"   # answer is NOT in the document

if __name__ == "__main__":
    print(f"Embedding model : {config.EMBEDDING_MODEL}")
    print(f"Threshold       : {config.MIN_SIMILARITY_SCORE}\n")

    # Small chunks so each paragraph becomes its own chunk.
    chunks = build_chunks([PageText(None, DOCUMENT)], "demo.txt", "English", 300, 50)
    print(f"Indexing {len(chunks)} chunks (first run downloads the model)...")
    retriever = Retriever(index_chunks(chunks))

    passed = 0
    for question, expected in TESTS:
        result = retriever.retrieve(question, top_k=3, min_score=0.0)
        top = result.chunks[0]
        ok = expected.lower() in top.text.lower()
        passed += ok
        print(f"\n{'PASS' if ok else 'FAIL'}  {question}")
        for c in result.chunks:
            print(f"      #{c.rank} score={c.score:.2f}  {c.text[:70]}...")

    neg = retriever.retrieve(NEGATIVE, top_k=3, min_score=0.0)
    kept = retriever.retrieve(NEGATIVE)   # with the real threshold
    print(f"\nNegative test: '{NEGATIVE}'")
    print(f"      best score={neg.best_score:.2f}  ->  relevant context found: {kept.has_relevant_context}")
    print(f"\n{passed}/{len(TESTS)} cross-lingual questions retrieved the right chunk.")
