"""
STEP 2 of vector-less, PageIndex-style RAG: answer a question.

Same idea as the original main.py:
  1. "Tree search": show the LLM the document's tree of titles + short
     summaries (NOT the full text) and ask which section(s) would
     contain the answer -- the way a person scans a table of contents
     before flipping to a page.
  2. Pull the FULL, unmodified text of just those section(s) using their
     page ranges from the tree.
  3. Answer the question using only that text.

Refactored into answer_question(), which returns a dict with the final
answer AND a step-by-step "trace" (plain title/params/result strings,
never raw JSON) so the Streamlit app can show the reasoning the same
way it shows the final answer: clean and human-readable.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pageindex_utils import extract_json
import storage

llm = ChatMistralAI(model="mistral-small-2603")

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def flatten_tree(nodes, depth=0, out=None):
    """Turn the nested tree into a flat list for showing to the LLM."""
    if out is None:
        out = []
    for node in nodes:
        out.append({
            "node_id": node["node_id"],
            "title": ("    " * depth) + node["title"],
            "summary": node["summary"],
        })
        flatten_tree(node.get("nodes", []), depth + 1, out)
    return out


def find_node(nodes, node_id):
    for node in nodes:
        if node["node_id"] == node_id:
            return node
        found = find_node(node.get("nodes", []), node_id)
        if found:
            return found
    return None


def tree_search(question, tree):
    """Reasoning-based retrieval over the tree index -- the vector-less
    replacement for retriever.invoke(query)."""
    flat = flatten_tree(tree)
    listing = "\n".join(f"[{n['node_id']}] {n['title']} - {n['summary']}" for n in flat)

    prompt = f"""You are navigating the table of contents of a document to answer a
question, the way a human expert would: scan section titles and summaries,
then decide which section(s) are worth opening in full.

Table of contents:
{listing}

Question: {question}

Pick at most 3 of the most relevant node_ids.
Return ONLY valid JSON in exactly this shape, no markdown fences:
{{"node_ids": ["0007", "0008"]}}
"""
    response = llm.invoke(prompt)
    result = extract_json(response.content)
    return result.get("node_ids", [])


def get_node_context(node_ids, tree, pages):
    """Pull the full text for the page range of each selected node."""
    pieces = []
    used_nodes = []
    for node_id in node_ids:
        node = find_node(tree, node_id)
        if not node:
            continue
        used_nodes.append(node)
        text = "\n".join(
            pages[str(p)] for p in range(node["start_index"], node["end_index"] + 1)
            if str(p) in pages
        )
        pieces.append(f"### {node['title']} (pages {node['start_index']}-{node['end_index']})\n{text}")
    return "\n\n".join(pieces), used_nodes


def answer_question(doc_id, question):
    """
    Run the full vector-less RAG pipeline for one question.

    Returns:
        {
          "steps": [ {"icon", "title", "params": [(k, v), ...], "result": [str, ...]}, ... ],
          "sections_used": [str, ...],
          "answer": str,
        }
    "steps" is the human-readable trace -- analogous to the tool-call
    Parameters/Result panes -- meant to be rendered as plain text/markdown,
    never dumped as a JSON blob.
    """
    meta = storage.load_meta(doc_id)
    tree = storage.load_tree(doc_id)
    pages = storage.load_pages(doc_id)

    steps = []

    node_ids = tree_search(question, tree)
    chosen_nodes = [n for n in (find_node(tree, nid) for nid in node_ids) if n]

    steps.append({
        "icon": "🔍",
        "title": "Searching the table of contents",
        "params": [("Question", question), ("Document", meta["original_name"])],
        "result": (
            [f"{n['title']} — {n['summary']}" for n in chosen_nodes]
            if chosen_nodes else ["No matching section was found in the table of contents."]
        ),
    })

    if not chosen_nodes:
        return {"steps": steps, "sections_used": [], "answer": "I could not find the answer in the document."}

    context, used_nodes = get_node_context(node_ids, tree, pages)
    page_ranges = ", ".join(f"p.{n['start_index']}-{n['end_index']}" for n in used_nodes)

    steps.append({
        "icon": "📖",
        "title": "Reading the full section text",
        "params": [
            ("Section(s)", ", ".join(n["title"] for n in used_nodes)),
            ("Pages", page_ranges),
        ],
        "result": [f"Retrieved {len(context):,} characters of source text from {len(used_nodes)} section(s)."],
    })

    if not context:
        return {"steps": steps, "sections_used": [], "answer": "I could not find the answer in the document."}

    final_prompt = ANSWER_PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)

    steps.append({
        "icon": "🤖",
        "title": "Generating the answer",
        "params": [("Model", "mistral-small-2603"), ("Context source", "retrieved sections only")],
        "result": ["Answer generated below."],
    })

    return {
        "steps": steps,
        "sections_used": [n["title"] for n in used_nodes],
        "answer": response.content,
    }