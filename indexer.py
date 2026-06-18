"""
STEP 1 of vector-less, PageIndex-style RAG: build a hierarchical
"table of contents" tree index of a PDF.

This is a refactor of the original build_tree.py: same three-step idea --

  1. Ask the LLM to read the raw table of contents and turn it into a
     clean JSON outline of chapters and sections.
  2. Figure out which physical page each chapter/section actually starts
     on (by locating the heading text in the document body).
  3. Ask the LLM for a short summary of each chapter and section.

-- but turned into a function, index_pdf(), that takes raw PDF bytes and
an original filename instead of a hardcoded PDF_PATH, and saves its
output through storage.py instead of two loose JSON files. That's what
lets the Streamlit app index any number of uploaded PDFs, each kept
separately.
"""
import io
import itertools

from langchain_mistralai import ChatMistralAI

from pageindex_utils import load_pdf_pages, find_chapter_page, find_heading_page, extract_json
import storage

TOC_CHECK_PAGES = 20             # how many initial pages to scan for a table of contents
MAX_CHAPTERS = 30                # safety cap
CHAPTER_TEXT_CHARS_FOR_SUMMARY = 9000  # how much of a chapter to show the LLM when summarizing

llm = ChatMistralAI(model="mistral-small-2603")


def _llm_json(prompt):
    response = llm.invoke(prompt)
    return extract_json(response.content)


def _extract_toc_structure(pages):
    """Turn the raw table-of-contents pages into a clean JSON outline.

    We only keep numbered chapters and their first-level numbered sections
    (e.g. '1', '1.1', '1.2' -- not the deeper '1.3.1') to keep the tree a
    manageable size for this example. You can extend this to more levels.
    """
    raw_toc = "\n".join(pages[:TOC_CHECK_PAGES])
    prompt = f"""You will be given the first pages of a book, including its table of
contents mixed in with other front-matter text (preface, syllabus, etc.).

Extract ONLY the numbered chapters and their first-level numbered sections.
- Include entries like "1 Introduction to Machine Learning" and "1.2 What is Human Learning?"
- Do NOT include second-level entries like "1.3.1 Learning under expert guidance"
- Do NOT include unnumbered front-matter or back-matter (Preface, Acknowledgements,
  Appendix, Model Question Paper, Index, etc.)

Return ONLY valid JSON in exactly this shape, nothing else, no markdown fences:
{{
  "chapters": [
    {{
      "number": "1",
      "title": "Introduction to Machine Learning",
      "sections": [
        {{"number": "1.1", "title": "Introduction"}},
        {{"number": "1.2", "title": "What is Human Learning?"}}
      ]
    }}
  ]
}}

Document text:
{raw_toc}
"""
    return _llm_json(prompt)


def _summarize_chapter(chapter_title, sections, chapter_text):
    """One LLM call per chapter: produce a chapter summary plus a one-line
    summary for each of its sections. These summaries -- NOT the original
    text -- are what the retrieval step reads later, so keep them short."""
    section_list = "\n".join(f"- {s['number']} {s['title']}" for s in sections) or "(none)"
    prompt = f"""You are building a navigation index for a long document, like a table
of contents enriched with short descriptions. Someone will later read ONLY
these descriptions (not the original text) to judge whether a section is
worth opening in full, so be specific about what each section actually covers.

Chapter: {chapter_title}
Sections in this chapter:
{section_list}

Chapter text (may be truncated):
{chapter_text[:CHAPTER_TEXT_CHARS_FOR_SUMMARY]}

Return ONLY valid JSON in exactly this shape, no markdown fences:
{{
  "chapter_summary": "1-2 sentence summary of the whole chapter",
  "sections": [
    {{"number": "1.1", "summary": "1 sentence summary of this specific section"}}
  ]
}}
"""
    return _llm_json(prompt)


def index_pdf(file_bytes, original_name, on_progress=None):
    """
    Build (or reuse) the PageIndex tree for one PDF.

    file_bytes:    raw bytes of the uploaded PDF
    original_name: the filename the user uploaded, kept for display
    on_progress:   optional callback(str) used to report status while
                   indexing -- the Streamlit app uses this to show a
                   live "what's happening right now" message

    Returns the document's meta dict (see storage.make_meta). If this
    exact file was already indexed before, the existing result is
    reused instead of calling the LLM again.
    """
    def report(msg):
        if on_progress:
            on_progress(msg)

    doc_id = storage.make_doc_id(file_bytes)
    if storage.doc_exists(doc_id):
        report(f"'{original_name}' is already in your library -- skipping re-indexing")
        return storage.load_meta(doc_id)

    report("Reading PDF pages ...")
    pages = load_pdf_pages(io.BytesIO(file_bytes))

    report("Extracting the table-of-contents structure ...")
    toc = _extract_toc_structure(pages)
    chapters = toc.get("chapters", [])[:MAX_CHAPTERS]

    report("Locating chapter and section page numbers ...")
    located_chapters = []  # list of (chapter_dict, start_page_idx)
    search_from = 0
    for ch in chapters:
        start_idx = find_chapter_page(pages, ch["number"], search_from)
        if start_idx is None:
            continue
        located_chapters.append((ch, start_idx))
        search_from = start_idx + 1

    # Node ids only need to be unique within THIS document's tree, so a
    # counter local to this call (instead of module-level global state)
    # keeps indexing of one PDF from ever bleeding into another's ids.
    node_id_gen = (f"{n:04d}" for n in itertools.count(1))

    tree = []
    for i, (ch, start_idx) in enumerate(located_chapters):
        end_idx = (located_chapters[i + 1][1] - 1) if i + 1 < len(located_chapters) else len(pages) - 1

        located_sections = []
        sec_search_from = start_idx
        for sec in ch.get("sections", []):
            sec_idx = find_heading_page(pages, sec["number"], sec["title"], sec_search_from, end_idx + 1)
            if sec_idx is None:
                continue
            located_sections.append((sec, sec_idx))
            sec_search_from = sec_idx

        chapter_text = "\n".join(pages[start_idx:end_idx + 1])
        report(f"Summarizing chapter {ch['number']} '{ch['title']}' ...")
        try:
            summaries = _summarize_chapter(ch["title"], [s for s, _ in located_sections], chapter_text)
        except Exception:
            summaries = {"chapter_summary": "", "sections": []}

        section_summary_by_number = {s["number"]: s["summary"] for s in summaries.get("sections", [])}
        chapter_node_id = next(node_id_gen)  # assign the chapter's id before its children's

        section_nodes = []
        for j, (sec, sec_idx) in enumerate(located_sections):
            sec_end_idx = (located_sections[j + 1][1] - 1) if j + 1 < len(located_sections) else end_idx
            section_nodes.append({
                "node_id": next(node_id_gen),
                "title": f"{sec['number']} {sec['title']}",
                "level": 2,
                "start_index": sec_idx + 1,    # stored as 1-based page numbers
                "end_index": sec_end_idx + 1,
                "summary": section_summary_by_number.get(sec["number"], ""),
            })

        tree.append({
            "node_id": chapter_node_id,
            "title": f"{ch['number']} {ch['title']}",
            "level": 1,
            "start_index": start_idx + 1,
            "end_index": end_idx + 1,
            "summary": summaries.get("chapter_summary", ""),
            "nodes": section_nodes,
        })

    meta = storage.make_meta(doc_id, original_name, num_pages=len(pages), num_chapters=len(tree))
    pages_by_number = {str(i + 1): pages[i] for i in range(len(pages))}
    storage.save_document(doc_id, meta, {"doc": original_name, "nodes": tree}, pages_by_number)

    report(f"Done -- indexed {len(tree)} chapter(s) from {len(pages)} pages")
    return meta