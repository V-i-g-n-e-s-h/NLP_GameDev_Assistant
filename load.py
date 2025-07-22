import json
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


JSONL_FILE = Path("unity_docs.jsonl")
CHROMA_DIR = Path("unity_index")
COLLECTION_NAME = "unity_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1024


def load_docs(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def main():
    docs = load_docs(JSONL_FILE)
    print(f"Loaded {len(docs)} pages from {JSONL_FILE}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance
    )

    indexed = 0
    for doc in tqdm(docs, desc="Indexing", unit="page"):
        chunks = chunk_text(doc["content"], CHUNK_SIZE)
        if not chunks:
            continue
        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        ids = [f"{doc['id']}_{i}" for i in range(len(chunks))]
        metas = [
            {"url": doc["url"], "title": doc["title"], "chunk": i}
            for i in range(len(chunks))
        ]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metas)
        indexed += len(chunks)

    print(f"Indexed {indexed} chunks to {CHROMA_DIR} (collection '{COLLECTION_NAME}')")


if __name__ == "__main__":
    main()
