"""Hybrid (dense + sparse) retrieval for the chatbot's hospital-info answers, scoped
per clinic.

Combines ChromaDB semantic search with BM25 keyword search via LangChain's
EnsembleRetriever, so an exact token (a doctor's name, a fee amount) that a dense
embedding might drift past is still caught by BM25, and vice versa.

Anti-hallucination guard: the similarity threshold check runs BEFORE the ensemble
step, against the raw cosine similarity of the top dense match. If nothing clears
RETRIEVAL_SIMILARITY_THRESHOLD, the fixed fallback is returned and the LLM is never
even handed retrieved context — it must not improvise around a gap in the KB.

Clinic isolation is structural, not a filter: each clinic's hospital-info data lives
in its own ChromaDB collection (named from the clinic_id-derived slug), so there is
no shared collection a cross-clinic query could leak from.

medical_kb no longer exists as a concept: symptom/triage messages are routed by
app.services.chat (via app.services.message_classifier.is_symptom_message) to real
department-list context from app.services.department_availability instead of KB text
search — there was never a reliable way to guarantee a medical-reference document
covered every symptom/department combination, and the structured doctors/departments
tables are the actual source of truth for what departments exist. This module now
only ever queries the hospital-info collection (timings, location, fees, policies,
booking-process questions) — no dual-collection routing, no classify_intent()
tiebreak.

Query expansion: a short, informally-phrased patient question ("what's the parking
situation") often sits meaningfully farther from a KB chunk's embedding than a more
formally-worded one, even when the chunk is genuinely the right answer. expand_query()
folds in a small clinic-domain synonym dictionary (rule-based, local, no network call)
before embedding, to nudge genuinely relevant queries closer to their matching chunks
without an LLM round-trip on every message. It only ever touches the query side —
ingested document text is untouched.
"""
import math
import uuid
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from app.core.config import settings
from app.rag.chroma_client import get_chroma_client, get_hospital_info_collection
from app.rag.embeddings import HostedEmbeddings, embed_query
from app.rag.preprocessing import preprocess_query

FALLBACK_MESSAGE = (
    "I don't have that information. Please contact the clinic directly for details."
)

# Fixed Urdu refusal — NOT a translation performed at request time (an LLM asked to
# translate tends to soften a refusal into something that sounds like an answer). Kept
# as its own hardcoded string so both language variants stay an equally hard stop.
FALLBACK_MESSAGE_UR = (
    "میرے پاس یہ معلومات موجود نہیں ہیں۔ براہ کرم تفصیلات کے لیے کلینک سے براہ راست رابطہ کریں۔"
)

RETRIEVER_K = 5


@dataclass
class RetrievalResult:
    matched: bool
    best_score: float
    chunks: list[str]
    fallback_message: str | None


# Small, clinic-domain-specific synonym map — deliberately not a general-purpose
# thesaurus. Keys are the LEMMATIZED form a term takes after preprocess_query() (e.g.
# "opening hours" -> "open hour"), since expansion runs on already-preprocessed text.
# Values are related terms a patient might use instead, or that phrase a KB chunk in
# more formal language than the patient's own wording.
_QUERY_EXPANSION_TERMS: dict[str, tuple[str, ...]] = {
    "hour": ("timing", "schedule", "open"),
    "open": ("timing", "schedule", "hour"),
    "timing": ("hour", "schedule", "open"),
    "close": ("closing", "hour"),
    "location": ("address", "direction"),
    "address": ("location", "direction"),
    "fee": ("price", "cost", "charge"),
    "cost": ("fee", "price", "charge"),
    "price": ("fee", "cost", "charge"),
    "appointment": ("booking", "schedule", "visit"),
    "book": ("appointment", "schedule", "reserve"),
    "contact": ("phone", "call", "reach"),
    "holiday": ("closed", "closure"),
    "weekend": ("saturday", "sunday", "closed"),
}


def expand_query(preprocessed_query: str) -> str:
    """Appends related domain terms found in _QUERY_EXPANSION_TERMS for any word
    already present in the (lemmatized) query — a local dict lookup, so this adds no
    meaningful latency. Terms already present are never duplicated. Returns the
    original string unchanged if nothing in it matches the dictionary."""
    words = preprocessed_query.split()
    seen = set(words)
    additions: list[str] = []
    for word in words:
        for extra in _QUERY_EXPANSION_TERMS.get(word, ()):
            if extra not in seen:
                additions.append(extra)
                seen.add(extra)

    if not additions:
        return preprocessed_query
    return f"{preprocessed_query} {' '.join(additions)}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_dense_score(collection, query_embedding: list[float]) -> float:
    """Cosine similarity of the closest stored chunk, computed directly from raw
    embeddings rather than trusting Chroma's configured distance space — this way the
    threshold comparison is correct regardless of how a given collection's hnsw:space
    metadata was set at creation time."""
    result = collection.query(query_embeddings=[query_embedding], n_results=1, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0 or len(embeddings[0]) == 0:
        return -1.0
    return _cosine_similarity(query_embedding, list(embeddings[0][0]))


def _build_hybrid_retriever(collection) -> EnsembleRetriever | None:
    """Builds the dense+sparse EnsembleRetriever over one clinic-scoped collection.
    Returns None if the collection is empty (nothing to retrieve)."""
    raw = collection.get(include=["documents", "metadatas"])
    documents = raw.get("documents") or []
    if not documents:
        return None
    metadatas = raw.get("metadatas") or [{} for _ in documents]

    corpus = [Document(page_content=text, metadata=meta) for text, meta in zip(documents, metadatas)]

    vectorstore = Chroma(
        client=get_chroma_client(),
        collection_name=collection.name,
        embedding_function=HostedEmbeddings(),
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    bm25_retriever = BM25Retriever.from_documents(corpus, k=RETRIEVER_K)

    # Equal weighting: dense catches paraphrase/semantic drift, BM25 catches exact
    # tokens (names, fee amounts) dense embeddings can miss — neither is presumed
    # more reliable than the other for this KB's mixed content.
    return EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5])


def retrieve(db: Session, clinic_id: uuid.UUID, query: str) -> RetrievalResult:
    """Hospital-info-only retrieval — see module docstring for why there's no
    medical_kb collection or dual-collection routing here anymore."""
    preprocessed = preprocess_query(query)
    expanded = expand_query(preprocessed)
    query_embedding = embed_query(expanded)

    collection = get_hospital_info_collection(db, clinic_id)
    best_score = _best_dense_score(collection, query_embedding)

    if best_score < settings.RETRIEVAL_SIMILARITY_THRESHOLD:
        return RetrievalResult(
            matched=False,
            best_score=best_score,
            chunks=[],
            fallback_message=FALLBACK_MESSAGE,
        )

    ensemble = _build_hybrid_retriever(collection)
    docs = ensemble.invoke(expanded) if ensemble is not None else []

    return RetrievalResult(
        matched=True,
        best_score=best_score,
        chunks=[d.page_content for d in docs],
        fallback_message=None,
    )
