"""
Embedder for the ReadmitIQ RAG corpus.

Wraps sentence-transformers with lazy model loading.
The model is downloaded from Hugging Face on first use and cached at
~/.cache/huggingface/. Subsequent loads are instant.

We use sentence-transformers/all-MiniLM-L6-v2:
- 384-dimensional embeddings
- ~80MB model size
- Fast on CPU (no GPU required)
- Trained on 1B+ sentence pairs across many domains

For our biomedical corpus, a domain-specific model like
PubMedBERT or BioSentVec would likely retrieve more precisely.
We use the general MiniLM as a strong default; swapping to a
biomedical model is a one-line change.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for all callers."""
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded (embedding dim={model.get_embedding_dimension()})")
    return model


def embed(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts into a (N, 384) numpy array of unit vectors.

    Args:
        texts: list of strings to embed
        batch_size: how many to process at once. Larger = faster but more memory.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32, L2-normalized.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns a (384,) array."""
    return embed([text])[0]
