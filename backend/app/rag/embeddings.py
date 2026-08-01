"""Local sentence-transformers embedding model, loaded once at process start.

No embedding API calls — inference runs locally via sentence-transformers. The model
must produce 768-dim vectors (all-mpnet-base-v2's native dimension); a silent mismatch
here would corrupt every future retrieval in a way that looks like bad answers rather
than an error, so the dimension is asserted eagerly at import time and fails loudly
(raises, refusing to start the app) rather than deferring the check to first use.
"""
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.core.config import settings

EXPECTED_DIMENSION = 768

_model = SentenceTransformer(settings.EMBEDDING_MODEL)
_actual_dimension = (
    _model.get_embedding_dimension()
    if hasattr(_model, "get_embedding_dimension")
    else _model.get_sentence_embedding_dimension()
)

if _actual_dimension != EXPECTED_DIMENSION:
    raise RuntimeError(
        f"Embedding model '{settings.EMBEDDING_MODEL}' produced {_actual_dimension}-dim "
        f"vectors, expected {EXPECTED_DIMENSION}. Refusing to start: a silent dimension "
        "mismatch would corrupt retrieval for every clinic."
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


class LocalEmbeddings(Embeddings):
    """Adapts the local sentence-transformers model to LangChain's Embeddings
    interface, so the same model backs both direct ingestion calls (embed_texts) and
    the LangChain Chroma vectorstore wrapper used by the hybrid retriever."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)
