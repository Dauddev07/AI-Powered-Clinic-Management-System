"""ChromaDB access, scoped per clinic.

Every caller reaches a collection through get_medical_kb_collection/
get_hospital_info_collection, which derive the collection name from the clinic's slug —
never from client input — so a clinic can only ever read or write its own collections.
Naming matches app/scripts/seed_clinic.py exactly (which already creates these collections
empty when a clinic is onboarded): '<slug>__medical-kb' and '<slug>__hospital-info'.
"""
import uuid

import chromadb
from chromadb.api.models.Collection import Collection
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.clinic import Clinic

_client: chromadb.HttpClient | None = None


def get_chroma_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
    return _client


def medical_kb_collection_name(clinic_slug: str) -> str:
    return f"{clinic_slug}__medical-kb"


def hospital_info_collection_name(clinic_slug: str) -> str:
    return f"{clinic_slug}__hospital-info"


def kb_type_for_collection_name(collection_name: str) -> str:
    """Recovers "medical_kb"/"hospital_info" from a stored collection name — lets
    callers (e.g. serializing a KBDocument row) report which KB a document belongs to
    without a separate DB column, since the collection name already encodes it."""
    if collection_name.endswith("__hospital-info"):
        return "hospital_info"
    return "medical_kb"


def _clinic_slug(db: Session, clinic_id: uuid.UUID) -> str:
    clinic = db.get(Clinic, clinic_id)
    if clinic is None:
        raise ValueError(f"Unknown clinic_id {clinic_id}")
    return clinic.slug


def get_medical_kb_collection(db: Session, clinic_id: uuid.UUID) -> Collection:
    slug = _clinic_slug(db, clinic_id)
    return get_chroma_client().get_or_create_collection(medical_kb_collection_name(slug))


def get_hospital_info_collection(db: Session, clinic_id: uuid.UUID) -> Collection:
    slug = _clinic_slug(db, clinic_id)
    return get_chroma_client().get_or_create_collection(hospital_info_collection_name(slug))


def get_collection_by_name(name: str) -> Collection:
    """Reaches a collection by its already-resolved name — used when a caller (e.g. a
    KBDocument row) already recorded which collection it belongs to, so re-deriving the
    name from clinic_id + a hardcoded kb_type would risk picking the wrong one."""
    return get_chroma_client().get_or_create_collection(name)
