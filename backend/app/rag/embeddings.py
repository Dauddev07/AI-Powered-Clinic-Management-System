"""Hosted-embeddings client — calls the Hugging Face Inference API's feature-extraction
pipeline instead of running sentence-transformers/torch in-process. Free-tier Render
hosting doesn't have the RAM to load a local embedding model alongside FastAPI/langchain,
so embedding calls are routed to HF's hosted API instead.

The model must still produce 768-dim vectors — same invariant the local version
enforced — checked on the first real call rather than at import time, since checking
requires a network call that can't run at module import.
"""
import httpx
from langchain_core.embeddings import Embeddings

from app.core.config import settings

EXPECTED_DIMENSION = 768
_INFERENCE_URL = "https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{model}"

_dimension_checked = False


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"} if settings.HF_TOKEN else {}
    response = httpx.post(
        _INFERENCE_URL.format(model=settings.EMBEDDING_MODEL),
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60.0,
    )
    response.raise_for_status()
    vectors = response.json()

    global _dimension_checked
    if not _dimension_checked:
        actual_dimension = len(vectors[0])
        if actual_dimension != EXPECTED_DIMENSION:
            raise RuntimeError(
                f"Embedding model '{settings.EMBEDDING_MODEL}' produced {actual_dimension}-dim "
                f"vectors, expected {EXPECTED_DIMENSION}. Refusing to continue: a silent "
                "dimension mismatch would corrupt retrieval for every clinic."
            )
        _dimension_checked = True

    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


class HostedEmbeddings(Embeddings):
    """Adapts the hosted embedding client to LangChain's Embeddings interface, so the
    same backend serves both direct ingestion calls (embed_texts) and the LangChain
    Chroma vectorstore wrapper used by the hybrid retriever."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)
