"""
Retriever service for ReadmitIQ RAG.

Wraps the Qdrant client and embedder in a typed interface. Lazy-loads
both resources on first use; subsequent calls reuse the cached instances.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger
from qdrant_client import QdrantClient

from readmit_iq.rag.embedder import embed_one
from readmit_iq.rag.indexer import COLLECTION_NAME, QDRANT_URL


@dataclass(frozen=True)
class RetrievedDocument:
    """One result from a similarity search."""

    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    authors: list[str]
    score: float

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    """One Qdrant client, reused across all callers."""
    logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
    return QdrantClient(url=QDRANT_URL)


class Retriever:
    """
    Retrieves relevant biomedical abstracts for a free-text clinical query.

    Usage:
        retriever = Retriever()
        docs = retriever.retrieve("heart failure patient needs follow-up", top_k=3)
        for doc in docs:
            print(doc.title, doc.score)
    """

    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.collection_name = collection_name
        self.client = _get_client()

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        """Embed the query and return the top_k most similar documents."""
        if not query.strip():
            return []

        query_vector = embed_one(query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        documents: list[RetrievedDocument] = []
        for point in result.points:
            payload = point.payload or {}
            documents.append(
                RetrievedDocument(
                    pmid=str(payload.get("pmid", "")),
                    title=payload.get("title", ""),
                    abstract=payload.get("abstract", ""),
                    journal=payload.get("journal", ""),
                    year=payload.get("year", ""),
                    authors=list(payload.get("authors", [])),
                    score=float(point.score or 0.0),
                )
            )
        return documents
