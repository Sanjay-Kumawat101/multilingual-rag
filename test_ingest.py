"""
test_ingest.py - quick check (load -> detect language -> chunk).

Run:   python test_ingest.py                 (uses built-in sample texts)
       python test_ingest.py my_file.pdf     (tests your own file)
"""
import sys
from pathlib import Path

from modules.document_loader import (
    EmptyDocumentError, UnsupportedFormatError, load_document, save_uploaded_file,
)
from modules.language_detector import detect_language
from modules.text_processor import build_chunks

SAMPLES = {
    "english.txt": (
        "Artificial Intelligence is a branch of computer science that focuses on creating "
        "systems capable of performing tasks that normally require human intelligence. "
        "Machine learning is a subset of AI in which computers learn patterns from data "
        "instead of being explicitly programmed. Deep learning uses neural networks with "
        "many layers to learn complex patterns such as images and speech."
    ),
    "hindi.txt": (
        "आर्टिफिशियल इंटेलिजेंस कंप्यूटर विज्ञान की एक शाखा है जो ऐसी प्रणालियाँ बनाने पर ध्यान "
        "केंद्रित करती है जो सामान्यतः मानव बुद्धि की आवश्यकता वाले कार्य कर सकें। मशीन लर्निंग "
        "एआई का एक उपसमूह है, जिसमें कंप्यूटर डेटा से पैटर्न सीखते हैं। डीप लर्निंग में कई परतों "
        "वाले न्यूरल नेटवर्क का उपयोग किया जाता है।"
    ),
    "marathi.txt": (
        "कृत्रिम बुद्धिमत्ता ही संगणक विज्ञानाची एक शाखा आहे. ज्या कामांसाठी सामान्यतः मानवी "
        "बुद्धिमत्ता लागते, ती कामे करू शकणाऱ्या प्रणाली तयार करण्यावर यात भर दिला जातो. यंत्र "
        "शिक्षण हा कृत्रिम बुद्धिमत्तेचा एक भाग आहे, ज्यामध्ये संगणक माहितीमधून शिकतात."
    ),
    "sanskrit.txt": (
        "संस्कृतं भारतस्य प्राचीना भाषा अस्ति। सा देववाणी इति अपि कथ्यते। अस्याः साहित्यं "
        "विशालम् अस्ति। वेदाः उपनिषदः पुराणानि च संस्कृतभाषायां लिखितानि सन्ति॥"
    ),
}


def run(path: Path) -> None:
    pages = load_document(path)
    text = "\n\n".join(p.text for p in pages)
    result = detect_language(text)
    # Small chunk size here so even the short samples are split into several chunks.
    chunks = build_chunks(pages, path.name, result.language, chunk_size=200, chunk_overlap=40)
    print(f"\n=== {path.name} ===")
    print(f"pages={len(pages)}  chars={len(text)}  chunks={len(chunks)}")
    print(f"detected={result.language}  confidence={result.confidence:.2f}  "
          f"reliable={result.reliable}  script={result.script}")
    print("first chunk metadata:", chunks[0].metadata)
    print("first chunk text    :", chunks[0].text[:120])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(Path(sys.argv[1]))
    else:
        for name, content in SAMPLES.items():
            saved = save_uploaded_file(name, content.encode("utf-8"))
            run(saved)
            saved.unlink()  # clean up

        print("\n--- error handling ---")
        try:
            save_uploaded_file("virus.exe", b"x")
        except UnsupportedFormatError as e:
            print("unsupported:", e)
        empty = save_uploaded_file("empty.txt", b"   \n  ")
        try:
            load_document(empty)
        except EmptyDocumentError as e:
            print("empty      :", e)
        empty.unlink()
        print("path trick :", save_uploaded_file("../../escape.txt", b"hello").parent.name)
        (Path("data/uploads/escape.txt")).unlink()
