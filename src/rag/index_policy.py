"""Index the demonstration policy corpus in ChromaDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .policy_docs import POLICIES

COLLECTION_NAME = "claims_iq_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def index_policies(persist_dir: Path) -> int:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    ids = [p["policy_id"] for p in POLICIES]
    texts = [p["text"] for p in POLICIES]
    metadatas = [{"title": p["title"], "policy_id": p["policy_id"]} for p in POLICIES]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(POLICIES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    args = parser.parse_args()
    print(f"Indexed {index_policies(args.persist_dir)} demonstration policy documents")
