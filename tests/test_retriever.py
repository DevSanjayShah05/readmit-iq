"""Tests for the RAG retriever service."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from readmit_iq.rag.retriever import RetrievedDocument, Retriever


# ---------- RetrievedDocument ----------


def test_retrieved_document_is_frozen():
    """Results should be immutable."""
    doc = RetrievedDocument(
        pmid="123", title="t", abstract="a", journal="j",
        year="2024", authors=["A"], score=0.5,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        doc.score = 0.9  # type: ignore[misc]


def test_retrieved_document_pubmed_url():
    doc = RetrievedDocument(
        pmid="12345", title="t", abstract="a", journal="j",
        year="2024", authors=[], score=0.5,
    )
    assert doc.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/12345/"


# ---------- Retriever.retrieve ----------


def _fake_qdrant_point(pmid: str, title: str, score: float) -> MagicMock:
    """Build a fake Qdrant point with the payload our retriever expects."""
    point = MagicMock()
    point.payload = {
        "pmid": pmid,
        "title": title,
        "abstract": f"abstract for {title}",
        "journal": "Test Journal",
        "year": "2024",
        "authors": ["Author One", "Author Two"],
    }
    point.score = score
    return point


@patch("readmit_iq.rag.retriever.embed_one")
@patch("readmit_iq.rag.retriever._get_client")
def test_retrieve_returns_typed_documents(mock_get_client, mock_embed):
    """The retriever should return RetrievedDocument objects from Qdrant points."""
    # Set up the embedder to return a deterministic vector
    mock_embed.return_value = np.zeros(384, dtype=np.float32)

    # Set up Qdrant to return two points
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.points = [
        _fake_qdrant_point("111", "Paper One", 0.85),
        _fake_qdrant_point("222", "Paper Two", 0.72),
    ]
    mock_client.query_points.return_value = mock_result
    mock_get_client.return_value = mock_client

    retriever = Retriever()
    docs = retriever.retrieve("heart failure follow-up", top_k=2)

    # Verify shape and content
    assert len(docs) == 2
    assert all(isinstance(d, RetrievedDocument) for d in docs)
    assert docs[0].pmid == "111"
    assert docs[0].title == "Paper One"
    assert docs[0].score == 0.85
    assert docs[1].pmid == "222"
    assert docs[1].score == 0.72


@patch("readmit_iq.rag.retriever.embed_one")
@patch("readmit_iq.rag.retriever._get_client")
def test_retrieve_empty_query_short_circuits(mock_get_client, mock_embed):
    """Empty query should return [] without hitting Qdrant or the embedder."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    retriever = Retriever()
    assert retriever.retrieve("", top_k=3) == []
    assert retriever.retrieve("   ", top_k=3) == []  # whitespace counts as empty

    # Neither the embedder nor Qdrant should have been called
    mock_embed.assert_not_called()
    mock_client.query_points.assert_not_called()


@patch("readmit_iq.rag.retriever.embed_one")
@patch("readmit_iq.rag.retriever._get_client")
def test_retrieve_passes_top_k_to_qdrant(mock_get_client, mock_embed):
    """top_k should propagate to Qdrant's limit parameter."""
    mock_embed.return_value = np.zeros(384, dtype=np.float32)
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.points = []
    mock_client.query_points.return_value = mock_result
    mock_get_client.return_value = mock_client

    retriever = Retriever()
    retriever.retrieve("some query", top_k=7)

    call = mock_client.query_points.call_args
    assert call.kwargs["limit"] == 7


@patch("readmit_iq.rag.retriever.embed_one")
@patch("readmit_iq.rag.retriever._get_client")
def test_retrieve_handles_missing_payload_fields(mock_get_client, mock_embed):
    """If Qdrant returns a point with a partial payload, retriever degrades."""
    mock_embed.return_value = np.zeros(384, dtype=np.float32)
    mock_client = MagicMock()
    mock_result = MagicMock()
    bad_point = MagicMock()
    bad_point.payload = {"pmid": "999"}  # missing title, abstract, etc.
    bad_point.score = 0.4
    mock_result.points = [bad_point]
    mock_client.query_points.return_value = mock_result
    mock_get_client.return_value = mock_client

    retriever = Retriever()
    docs = retriever.retrieve("query", top_k=1)

    assert len(docs) == 1
    assert docs[0].pmid == "999"
    assert docs[0].title == ""  # defaulted
    assert docs[0].abstract == ""
    assert docs[0].authors == []
