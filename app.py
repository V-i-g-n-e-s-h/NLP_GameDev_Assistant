import time
from typing import List, Dict
from pathlib import Path

import streamlit as st
from humanfriendly import format_timespan
import chromadb
from sentence_transformers import SentenceTransformer
import ollama


CHROMA_DIR = Path("unity_index")
COLLECTION_NAME = "unity_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLAMA_MODEL = "llama3.2"
TOP_K = 3
MAX_TOKENS_CONTEXT = 3500


@st.cache_resource(show_spinner="Loading vector index & embedding model...")
def load_resources():
    # Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    # Embedding model (CPU‑friendly)
    embedder = SentenceTransformer(EMBED_MODEL)
    return collection, embedder


collection, embedder = load_resources()


def retrieve_docs(query: str, k: int = TOP_K) -> List[Dict]:
    """Return top-k relevant chunks with metadata and distances."""
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    res = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for doc, meta, dist in zip(res["documents"], res["metadatas"], res["distances"]):
        docs.append({"content": doc, "meta": meta, "distance": dist})
    return docs


def build_prompt(query: str, docs: List[Dict], history: List[Dict]) -> List[Dict]:
    """Create a chat prompt list for Ollama with system + history + user."""
    # System message
    system_msg = {
        "role": "system",
        "content": (
            "You are GameDev Assistant, an expert on Unity game engine.\n"
            "Answer strictly based on the provided documentation excerpts.\n"
            "If unsure, say you don't know. Include citations as [#] linking to the URLs."
        ),
    }

    # Compose context snippets with numeric IDs for citation
    context_lines = []
    url_map = []  # keep parallel list of urls for citations later
    for idx, ret in enumerate(zip(docs[0]["content"], docs[0]["meta"], docs[0]["distance"]), 1):
        content, meta, distance = ret
        title = meta["title"]
        url = meta["url"]
        snippet = content
        context_lines.append(f"[#${idx}] {title}\n{snippet}")
        url_map.append(url)

    context_block = "\n\n".join(context_lines)

    # Final user query incorporating the context
    user_content = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUERY: {query}\n\n"
        f"Respond with Markdown. Cite sources as [#]."
    )

    # Assemble chat list: system, history, new user
    messages = [system_msg] + history + [{"role": "user", "content": user_content}]

    return messages, url_map


def stream_response(messages: List[Dict]):
    """Stream response from Ollama and yield chunks."""
    stream = ollama.chat(model=LLAMA_MODEL, messages=messages, stream=True)
    for part in stream:
        yield part["message"]["content"]


st.set_page_config(page_title="GameDev Assistant", layout="wide")
st.title("GameDev Assistant - Unity Docs RAG")

# initialise session state
if "history" not in st.session_state:
    st.session_state.history = []  # chat history for llama

query = st.chat_input("Ask anything about Unity...")

# Display previous conversation
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

if query:
    # Echo user message in UI immediately
    st.chat_message("user").markdown(query)

    # Retrieval & Llama call
    start_time = time.perf_counter()
    docs = retrieve_docs(query, TOP_K)
    messages, url_map = build_prompt(query, docs, st.session_state.history)

    # Stream answer
    placeholder = st.chat_message("assistant")
    answer_container = placeholder.empty()
    partial = ""
    for chunk in stream_response(messages):
        partial += chunk
        answer_container.markdown(partial + "|")
    exec_ms = (time.perf_counter() - start_time) * 1000

    # Append reference links
    unique_urls = list(dict.fromkeys(url_map))  # dedupe, preserve order
    refs_md = "\n\n" + "\n".join(f"[#${i+1}]: {url}" for i, url in enumerate(unique_urls))
    final_answer = partial + refs_md + f"\n\n_Answered in {format_timespan(exec_ms / 1000)}_"
    answer_container.markdown(final_answer)

    # Update chat history (truncate if exceeding limits)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": final_answer})

    # prevent session from blowing up
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]
