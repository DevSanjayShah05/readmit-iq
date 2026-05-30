"""
Qdrant indexer for the ReadmitIQ RAG corpus.

Loads abstracts from data/rag/abstracts.json, embeds each, and upserts
into a Qdrant collection. Idempotent: re-running replaces existing points
with the same PMIDs.

Run as a script:
    python -m readmit_iq.rag.indexer
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from readmit_iq.rag.embedder import EMBEDDING_DIM, embed


COLLECTION_NAME = "readmit_iq_corpus"
QDRANT_URL = "http://localhost:6333"
DEFAULT_CORPUS_PATH = Path("data/rag/abstracts.json")
UPSERT_BATCH_SIZE = 64


def get_client() -> QdrantClient:
    """Get a Qdrant client connected to the local instance."""
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME) -> None:
    """Create the collection if it doesn't exist. Idempotent."""
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        logger.info(f"Collection {name!r} already exists")
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.success(f"Created collection {name!r} (dim={EMBEDDING_DIM}, metric=cosine)")


def text_for_embedding(abstract: dict) -> str:
    """Build the text we feed to the embedder. Title + abstract body."""
    return f"{abstract['title']}\n\n{abstract['abstract']}"


def index_corpus(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Load the corpus JSON, embed each abstract, upsert into Qdrant.
    Returns the number of points indexed.
    """
    logger.info(f"Loading corpus from {corpus_path}")
    abstracts = json.loads(corpus_path.read_text())
    logger.info(f"Loaded {len(abstracts)} abstracts")

    client = get_client()
    ensure_collection(client, collection_name)

    # Embed all texts in one batch (the embedder handles internal batching).
    texts = [text_for_embedding(a) for a in abstracts]
    logger.info("Embedding corpus (this takes ~20-30 seconds)...")
    vectors = embed(texts, batch_size=32)
    logger.success(f"Embedded {len(vectors)} texts (shape={vectors.shape})")

    # Build point records and upsert in batches.
    points = [
        PointStruct(
            id=int(abstract["pmid"]),  # PMID is numeric; use as Qdrant id
            vector=vectors[i].tolist(),
            payload={
                "pmid": abstract["pmid"],
                "title": abstract["title"],
                "abstract": abstract["abstract"],
                "journal": abstract["journal"],
                "year": abstract["year"],
                "authors": abstract["authors"],
                "query": abstract["query"],
            },
        )
        for i, abstract in enumerate(abstracts)
    ]

    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info(f"  Upserted {i + len(batch)} / {len(points)}")

    logger.success(f"Indexed {len(points)} points into {collection_name!r}")
    return len(points)


if __name__ == "__main__":
    index_corpus()
