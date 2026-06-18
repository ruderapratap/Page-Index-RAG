"""
Storage layer for the vector-less PageIndex RAG app.

Each processed PDF gets its own folder under STORAGE_DIR, named after a
short hash of its content (so re-uploading the exact same file is a
no-op instead of creating a duplicate). A folder holds:

  meta.json   - original filename, upload time, page/chapter counts
  tree.json   - the PageIndex tree (titles, summaries, page ranges)
  pages.json  - raw text of every page, used later to pull full section text

This is the piece that makes "multiple PDFs", "history", "delete one",
and "clear all" possible -- the original project only ever knew about a
single hardcoded PDF_PATH.
"""
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

STORAGE_DIR = Path(__file__).parent / "storage"


def _doc_dir(doc_id):
    return STORAGE_DIR / doc_id


def make_doc_id(file_bytes):
    """Short, stable id derived from file content so the same PDF always
    maps to the same id, and uploading it twice doesn't duplicate work."""
    return hashlib.sha1(file_bytes).hexdigest()[:12]


def doc_exists(doc_id):
    return (_doc_dir(doc_id) / "meta.json").exists()


def make_meta(doc_id, original_name, num_pages, num_chapters):
    return {
        "doc_id": doc_id,
        "original_name": original_name,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        "num_pages": num_pages,
        "num_chapters": num_chapters,
    }


def save_document(doc_id, meta, tree, pages):
    folder = _doc_dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(folder / "tree.json", "w") as f:
        json.dump(tree, f, indent=2)
    with open(folder / "pages.json", "w") as f:
        json.dump(pages, f)


def load_meta(doc_id):
    with open(_doc_dir(doc_id) / "meta.json") as f:
        return json.load(f)


def load_tree(doc_id):
    with open(_doc_dir(doc_id) / "tree.json") as f:
        return json.load(f)["nodes"]


def load_pages(doc_id):
    with open(_doc_dir(doc_id) / "pages.json") as f:
        return json.load(f)


def list_documents():
    """All indexed documents, most recently uploaded first."""
    if not STORAGE_DIR.exists():
        return []
    docs = []
    for folder in STORAGE_DIR.iterdir():
        meta_path = folder / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                docs.append(json.load(f))
    docs.sort(key=lambda m: m["uploaded_at"], reverse=True)
    return docs


def delete_document(doc_id):
    folder = _doc_dir(doc_id)
    if folder.exists():
        shutil.rmtree(folder)


def clear_all():
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)