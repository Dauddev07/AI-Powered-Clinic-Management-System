"""Knowledge-base document ingestion.

Re-upload handling: a same-filename upload within a clinic is always treated as a
REPLACEMENT of the existing document, never a second copy. Extraction/chunking/
embedding of the NEW file happens first — only once that has fully succeeded do we
delete the old document's chunks and DB row, so a bad re-upload can never destroy a
working document while failing to install its replacement.
"""
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.kb_document import KBDocument
from app.rag.chroma_client import get_collection_by_name, get_hospital_info_collection
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts
from app.rag.extraction import UnsupportedDocumentType, extract_text
from app.rag.preprocessing import clean_text


@dataclass
class IngestResult:
    document: KBDocument
    replaced: bool


def delete_document_chunks(db: Session, clinic_id: uuid.UUID, document: KBDocument) -> None:
    """Removes ALL of a document's chunks from the vector store — used by both the
    DELETE endpoint and the re-upload replacement path, so both clear chunks the
    same way. Reaches the collection by the document's own stored chroma_collection
    name rather than re-deriving one from clinic_id, since a document may live in
    either the medical-kb or hospital-info collection."""
    collection = get_collection_by_name(document.chroma_collection)
    collection.delete(where={"kb_document_id": str(document.id)})


def ingest_document(db: Session, clinic_id: uuid.UUID, filename: str, file_bytes: bytes) -> IngestResult:
    try:
        raw_text = extract_text(filename, file_bytes)
    except UnsupportedDocumentType:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}': the file is corrupt or not a valid document ({exc}).")
    if not raw_text.strip():
        raise ValueError(f"No extractable text found in '{filename}'.")

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError(f"'{filename}' produced no chunks after splitting.")

    cleaned_chunks = [clean_text(c) for c in chunks]
    embeddings = embed_texts(cleaned_chunks)

    collection = get_hospital_info_collection(db, clinic_id)
    source_type = "pdf" if filename.lower().endswith(".pdf") else "docx"

    existing = db.execute(
        select(KBDocument).where(
            KBDocument.clinic_id == clinic_id,
            KBDocument.title == filename,
            KBDocument.chroma_collection == collection.name,
        )
    ).scalar_one_or_none()

    replaced = existing is not None
    if existing is not None:
        delete_document_chunks(db, clinic_id, existing)
        db.delete(existing)
        db.flush()

    document = KBDocument(
        clinic_id=clinic_id,
        source_type=source_type,
        source_ref_id=None,
        title=filename,
        content=raw_text,
        chroma_collection=collection.name,
        chunk_count=len(cleaned_chunks),
    )
    db.add(document)
    db.flush()

    ids = [f"{document.id}_{i}" for i in range(len(cleaned_chunks))]
    metadatas = [
        {
            "kb_document_id": str(document.id),
            "clinic_id": str(clinic_id),
            "filename": filename,
            "chunk_index": i,
        }
        for i in range(len(cleaned_chunks))
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=cleaned_chunks, metadatas=metadatas)

    return IngestResult(document=document, replaced=replaced)
