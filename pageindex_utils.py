"""
Shared utilities for the PageIndex-style vector-less RAG pipeline.

These helpers do the "boring" deterministic work (reading PDF pages,
finding where a heading physically starts, parsing JSON out of an LLM
response) so that indexer.py and retriever.py can stay focused on the
PageIndex logic itself.
"""
import re
import json
import difflib


def load_pdf_pages(pdf_source):
    """Return a list of page texts. pages[0] is page 1, pages[1] is page 2, etc.

    pdf_source can be a file path (str/Path) OR a file-like object such as
    io.BytesIO -- the latter lets the Streamlit app index an uploaded PDF
    straight from memory, without ever writing a temp file to disk.
    """
    from pypdf import PdfReader
    reader = PdfReader(pdf_source)
    return [(page.extract_text() or "") for page in reader.pages]


def _normalize(text):
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def find_chapter_page(pages, chapter_number, search_from):
    """
    Locate the physical page (0-based index) where 'Chapter <chapter_number>'
    actually starts as a heading.

    We require the match to be near the TOP of the page. Without that,
    a stray mid-paragraph sentence like '...will be covered in Chapter 3.'
    would be mistaken for the start of Chapter 3.
    """
    pattern = re.compile(r"chapter\s+" + str(chapter_number) + r"\b", re.IGNORECASE)
    for i in range(search_from, len(pages)):
        match = pattern.search(pages[i])
        if match and match.start() < 60:
            return i
    return None


def find_heading_page(pages, number, title, page_lo, page_hi):
    """
    Locate the physical page (0-based index) where a numbered heading like
    '1.2 What is Human Learning?' starts within pages[page_lo:page_hi].

    Two-stage strategy:
      1. Fast path: regex match for "<number> <first few title words>".
      2. Fallback: fuzzy line-by-line comparison. Real documents are messy
         (OCR errors, typos in the original book, inconsistent spacing),
         so an exact match isn't always possible. This mirrors why the
         real PageIndex project lets an LLM do this matching instead of a
         strict string search -- we approximate that with fuzzy matching
         to keep this example dependency-light and fast.
    """
    page_hi = min(page_hi, len(pages))
    title_words = re.sub(r"[^A-Za-z0-9\s]", " ", title).split()
    core_words = title_words[:6] if len(title_words) > 1 else title_words

    if core_words:
        pattern = re.compile(
            re.escape(number) + r"[.\s]+" + r"\s+".join(re.escape(w) for w in core_words),
            re.IGNORECASE,
        )
        for i in range(page_lo, page_hi):
            collapsed = re.sub(r"\s+", " ", pages[i])
            if pattern.search(collapsed):
                return i

    target = _normalize(f"{number} {title}")
    best_idx, best_ratio = None, 0.0
    for i in range(page_lo, page_hi):
        for line in pages[i].split("\n"):
            norm_line = _normalize(line)
            if not norm_line:
                continue
            ratio = difflib.SequenceMatcher(None, target, norm_line).ratio()
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, i

    return best_idx if best_ratio >= 0.55 else None


def extract_json(text):
    """Strip markdown code fences and parse the JSON object/array in an LLM reply."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from model output:\n{cleaned[:500]}")