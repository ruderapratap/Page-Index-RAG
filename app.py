"""
Streamlit front-end for the vector-less, PageIndex-style RAG pipeline.

Run with:
    streamlit run app.py

Requires a .env file (see .env.example) with MISTRAL_API_KEY set.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from indexer import index_pdf
from retriever import answer_question
import storage

st.set_page_config(page_title="PageIndex RAG", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Styling — Teal/Emerald palette:
# Primary: #0E6251 #148F76 #1AB798 #20DFB9 #48E5C5 #70EBD2 #98F0DF #C1F6EB #E9FCF8
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Base ---- */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f0fdf9;
    }
    h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0E6251;
    }

    /* ---- Main app background ---- */
    .stApp {
        background: linear-gradient(135deg, #E9FCF8 0%, #f0fdf9 50%, #C1F6EB 100%);
        min-height: 100vh;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E6251 0%, #148F76 60%, #1AB798 100%) !important;
        border-right: none;
        box-shadow: 4px 0 24px rgba(14,98,81,0.18);
    }
    section[data-testid="stSidebar"] * {
        color: #E9FCF8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.18) !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #99C68E !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
        text-shadow: 0 1px 4px rgba(0,0,0,0.18);
    }
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #C1F6EB !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }

    /* ---- File uploader dropzone (fixed contrast) ---- */
    /* Note: no tag name on the attribute selector — Streamlit renders this
       as a <section>, not a <div>, so a "div[data-testid=...]" selector
       silently never matches and the white default background shows through. */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #0C5246 !important;
        border: 2px dashed rgba(255,255,255,0.5) !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
        color: #C1F6EB !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff !important;
        color: #0E6251 !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
        color: #0E6251 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
        background: #E9FCF8 !important;
    }
    /* Uploaded file chip (name + size shown after upload) */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.14) !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
        color: #E9FCF8 !important;
        background-color: transparent !important;
    }

    /* ---- Sidebar document buttons ---- */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.12) !important;
        color: #E9FCF8 !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: background 0.2s, border 0.2s;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.22) !important;
        border-color: rgba(255,255,255,0.42) !important;
    }

    /* ---- Main area buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #1AB798 0%, #148F76 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 0.5rem 1.25rem;
        box-shadow: 0 2px 12px rgba(26,183,152,0.22);
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px rgba(26,183,152,0.38);
        transform: translateY(-1px);
    }

    /* ---- Chat bubbles ---- */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #C1F6EB 0%, #98F0DF 100%);
        border-radius: 16px;
        border: 1px solid #70EBD2;
        box-shadow: 0 2px 10px rgba(32,223,185,0.10);
        margin-bottom: 8px;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(135deg, #ffffff 0%, #E9FCF8 100%);
        border-radius: 16px;
        border: 1px solid #C1F6EB;
        box-shadow: 0 2px 10px rgba(14,98,81,0.07);
        margin-bottom: 8px;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li {
        color: #0E6251 !important;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* ---- Chat input ----
       Streamlit wraps the textarea AND a bottom toolbar (send button, any
       file-attach controls) inside the same stChatInput container. Styling
       only the textarea leaves that toolbar strip showing Streamlit's own
       default (reddish) background, so we style the whole container and
       make the inner pieces transparent on top of it. */
    div[data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 2px solid #70EBD2 !important;
        border-radius: 14px !important;
        box-shadow: none !important;
        overflow: hidden;
    }
    div[data-testid="stChatInput"]:has(textarea:focus) {
        border-color: #1AB798 !important;
        box-shadow: 0 0 0 3px rgba(26,183,152,0.15) !important;
    }
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] section,
    div[data-testid="stChatInput"] [class*="Toolbar"] {
        background: #ffffff !important;
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: #0E6251 !important;
        padding: 12px 16px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #7fb8ad !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #1AB798 0%, #148F76 100%) !important;
        border: none !important;
        border-radius: 8px !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    div[data-testid="stChatInput"] button:disabled {
        background: #C1F6EB !important;
        opacity: 1 !important;
    }
    div[data-testid="stChatInput"] button:disabled svg {
        fill: #70EBD2 !important;
        color: #70EBD2 !important;
    }

    /* ---- Trace step expanders ---- */
    details {
        border: 1.5px solid #70EBD2 !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        margin-bottom: 8px;
        box-shadow: 0 1px 6px rgba(26,183,152,0.08);
    }
    details summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
        color: #148F76 !important;
        font-weight: 500 !important;
        padding: 4px 0;
    }
    details[open] summary {
        color: #0E6251 !important;
        font-weight: 600 !important;
    }
    .pi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1AB798;
        margin-top: 8px;
        margin-bottom: 2px;
        font-weight: 600;
    }

    /* ---- Section badges ---- */
    .pi-badges { margin-top: 10px; }
    .pi-badge {
        display: inline-block;
        background: linear-gradient(135deg, #20DFB9 0%, #1AB798 100%);
        color: #ffffff;
        border-radius: 999px;
        padding: 4px 14px;
        margin: 3px 4px 3px 0;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 1px 6px rgba(26,183,152,0.22);
    }

    /* ---- Library cards ---- */
    .pi-doc-meta {
        font-size: 0.76rem;
        color: rgba(233,252,248,0.75);
        margin: -4px 0 10px 2px;
        font-weight: 500;
    }
    .pi-active-tag {
        background: linear-gradient(135deg, #20DFB9, #48E5C5);
        color: #0E6251;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.68rem;
        font-weight: 800;
        margin-left: 7px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ---- Info / success / warning alerts ---- */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border-left-width: 4px !important;
    }

    /* ---- Page header card ---- */
    .pi-header-card {
        background: linear-gradient(135deg, #0E6251 0%, #1AB798 100%);
        border-radius: 18px;
        padding: 28px 32px 22px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(14,98,81,0.18);
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .pi-header-card h1 {
        color: #ffffff !important;
        font-size: 1.9rem !important;
        margin: 0 0 4px 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .pi-header-card .pi-sub {
        color: #C1F6EB;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 0;
    }
    .pi-header-icon {
        font-size: 2.8rem;
        line-height: 1;
        filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));
    }

    /* ---- Active doc header ---- */
    .pi-doc-header {
        background: linear-gradient(135deg, #E9FCF8 0%, #C1F6EB 100%);
        border: 1.5px solid #70EBD2;
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 2px 10px rgba(26,183,152,0.10);
    }
    .pi-doc-header .pi-doc-title {
        color: #0E6251;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 0;
    }
    .pi-doc-header .pi-doc-icon {
        font-size: 1.5rem;
    }

    /* ---- Spinner ---- */
    .stSpinner > div {
        border-top-color: #1AB798 !important;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: #E9FCF8; }
    ::-webkit-scrollbar-thumb { background: #70EBD2; border-radius: 99px; }
    ::-webkit-scrollbar-thumb:hover { background: #1AB798; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# ---------------------------------------------------------------------------
# Sidebar — Document Library
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🗂️ Document Library")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded_files and st.button("➕ Add to Library", use_container_width=True):
        status = st.empty()
        last_doc_id = None
        for f in uploaded_files:
            def _report(msg, _name=f.name):
                status.info(f"**{_name}** — {msg}")

            meta = index_pdf(f.read(), f.name, on_progress=_report)
            last_doc_id = meta["doc_id"]
        status.empty()
        if last_doc_id:
            st.session_state.active_doc_id = last_doc_id
        st.success(f"✅ Processed {len(uploaded_files)} file(s) successfully.")
        st.rerun()

    st.markdown("---")

    docs = storage.list_documents()
    if not docs:
        st.caption("📭 No documents yet — upload a PDF above to get started.")
    else:
        for doc in docs:
            is_active = doc["doc_id"] == st.session_state.active_doc_id
            cols = st.columns([5, 1])
            label = ("🟢 " if is_active else "📄 ") + doc["original_name"]
            if cols[0].button(label, key=f"select_{doc['doc_id']}", use_container_width=True):
                st.session_state.active_doc_id = doc["doc_id"]
                st.rerun()
            if cols[1].button("🗑️", key=f"del_{doc['doc_id']}", help="Delete this document"):
                storage.delete_document(doc["doc_id"])
                st.session_state.chats.pop(doc["doc_id"], None)
                if st.session_state.active_doc_id == doc["doc_id"]:
                    st.session_state.active_doc_id = None
                st.rerun()
            tag = "<span class='pi-active-tag'>● active</span>" if is_active else ""
            st.markdown(
                f"<div class='pi-doc-meta'>🗒️ {doc['num_pages']} pages &nbsp;·&nbsp; "
                f"📑 {doc['num_chapters']} chapters{tag}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if docs:
        if st.session_state.confirm_clear:
            st.warning("⚠️ Delete **all** documents and chat history? This cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("🗑️ Yes, clear all", use_container_width=True):
                storage.clear_all()
                st.session_state.chats = {}
                st.session_state.active_doc_id = None
                st.session_state.confirm_clear = False
                st.rerun()
            if c2.button("↩️ Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
        else:
            if st.button("🧹 Clear All Documents", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()

# ---------------------------------------------------------------------------
# Main Panel — Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="pi-header-card">
        <div class="pi-header-icon">🔍</div>
        <div>
            <h1>PageIndex RAG</h1>
            <p class="pi-sub">Intelligent answers powered by document reasoning — no embeddings, no vector database.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active_id = st.session_state.active_doc_id

if not active_id:
    st.info("👈 Upload a PDF from the sidebar, or select an existing document from your library to start asking questions.")
else:
    meta = storage.load_meta(active_id)
    st.markdown(
        f"""
        <div class="pi-doc-header">
            <span class="pi-doc-icon">📖</span>
            <span class="pi-doc-title">{meta['original_name']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat = st.session_state.chats.setdefault(active_id, [])

    for msg in chat:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("steps"):
                for step in msg["steps"]:
                    with st.expander(f"{step['icon']} {step['title']}"):
                        st.markdown("<div class='pi-label'>⚙ Parameters</div>", unsafe_allow_html=True)
                        for k, v in step["params"]:
                            st.markdown(f"- **{k}:** {v}")
                        st.markdown("<div class='pi-label'>✦ Result</div>", unsafe_allow_html=True)
                        for line in step["result"]:
                            st.markdown(f"- {line}")
            st.markdown(msg["content"])
            if msg.get("sections_used"):
                badges = "".join(f"<span class='pi-badge'>📌 {s}</span>" for s in msg["sections_used"])
                st.markdown(f"<div class='pi-badges'>{badges}</div>", unsafe_allow_html=True)

    question = st.chat_input(f"💬 Ask anything about '{meta['original_name']}' ...")
    if question:
        chat.append({"role": "user", "content": question})
        with st.spinner("🔎 Searching the document and composing an answer..."):
            result = answer_question(active_id, question)
        chat.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "steps": result["steps"],
                "sections_used": result["sections_used"],
            }
        )
        st.rerun()