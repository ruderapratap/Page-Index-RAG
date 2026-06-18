"""
Storage layer for the vector-less PageIndex RAG app.

Each processed PDF gets its own folder under STORAGE_DIR/<user_id>/,
named after a short hash of its content (so re-uploading the exact same
file is a no-op instead of creating a duplicate). A folder holds:

  meta.json   - original filename, upload time, page/chapter counts
  tree.json   - the PageIndex tree (titles, summaries, page ranges)
  pages.json  - raw text of every page, used later to pull full section text

USER ISOLATION FIX:
  Previously all users shared a single STORAGE_DIR, meaning any uploaded
  PDF was visible to every user. Now each user gets their own sub-directory
  keyed by their Streamlit session_id, so documents are completely private.

This makes "multiple PDFs", "history", "delete one", and "clear all"
possible — and fully isolated per user.
"""

import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

STORAGE_DIR = Path(__file__).parent / "storage"


# ---------------------------------------------------------------------------
# User isolation — derive a stable per-session user directory
# ---------------------------------------------------------------------------

def _get_user_id() -> str:
    """
    Return a stable ID for the current Streamlit session.
    Falls back to a fixed string only during unit-test / non-Streamlit runs.
    """
    try:
        ctx = get_script_run_ctx()
        if ctx is not None:
            return hashlib.sha1(ctx.session_id.encode()).hexdigest()[:16]
    except Exception:
        pass
    return "default_user"


def _user_storage_dir() -> Path:
    """Each user gets their own private sub-directory inside STORAGE_DIR."""
    return STORAGE_DIR / _get_user_id()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _doc_dir(doc_id: str) -> Path:
    return _user_storage_dir() / doc_id


# ---------------------------------------------------------------------------
# Public API (unchanged signatures — drop-in replacement)
# ---------------------------------------------------------------------------

def make_doc_id(file_bytes: bytes) -> str:
    """Short, stable id derived from file content so the same PDF always
    maps to the same id, and uploading it twice doesn't duplicate work."""
    return hashlib.sha1(file_bytes).hexdigest()[:12]


def doc_exists(doc_id: str) -> bool:
    return (_doc_dir(doc_id) / "meta.json").exists()


def make_meta(doc_id: str, original_name: str, num_pages: int, num_chapters: int) -> dict:
    return {
        "doc_id": doc_id,
        "original_name": original_name,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
        "num_pages": num_pages,
        "num_chapters": num_chapters,
    }


def save_document(doc_id: str, meta: dict, tree: dict, pages: dict) -> None:
    folder = _doc_dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(folder / "tree.json", "w") as f:
        json.dump(tree, f, indent=2)
    with open(folder / "pages.json", "w") as f:
        json.dump(pages, f)


def load_meta(doc_id: str) -> dict:
    with open(_doc_dir(doc_id) / "meta.json") as f:
        return json.load(f)


def load_tree(doc_id: str) -> list:
    with open(_doc_dir(doc_id) / "tree.json") as f:
        return json.load(f)["nodes"]


def load_pages(doc_id: str) -> dict:
    with open(_doc_dir(doc_id) / "pages.json") as f:
        return json.load(f)


def list_documents() -> list:
    """All indexed documents for the CURRENT USER, most recently uploaded first."""
    user_dir = _user_storage_dir()
    if not user_dir.exists():
        return []

    docs = []
    for folder in user_dir.iterdir():
        meta_path = folder / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                docs.append(json.load(f))

    docs.sort(key=lambda m: m["uploaded_at"], reverse=True)
    return docs


def delete_document(doc_id: str) -> None:
    folder = _doc_dir(doc_id)
    if folder.exists():
        shutil.rmtree(folder)


def clear_all() -> None:
    """Clears ALL documents for the CURRENT USER only."""
    user_dir = _user_storage_dir()
    if user_dir.exists():
        shutil.rmtree(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
