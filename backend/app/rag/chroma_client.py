"""ChromaDB access, scoped per clinic.

Every caller reaches a collection through get_hospital_info_collection, which derives
the collection name from the clinic's slug — never from client input — so a clinic can
only ever read or write its own collection. Naming matches app/scripts/seed_clinic.py
exactly (which already creates this collection empty when a clinic is onboarded):
'<slug>__hospital-info'.

There used to be a second, per-clinic 'medical-kb' collection for symptom/triage
reference material — removed along with its helpers (get_medical_kb_collection,
medical_kb_collection_name, kb_type_for_collection_name) once that concept was
replaced by real, structured department/doctor data (see app.rag.retrieval's own
module docstring and app.services.department_availability). Nothing ever wrote to or
read from it since.
"""
from __future__ import annotations

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
        _client = chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, ssl=settings.CHROMA_SSL
        )
    return _client


def hospital_info_collection_name(clinic_slug: str) -> str:
    return f"{clinic_slug}__hospital-info"


def _clinic_slug(db: Session, clinic_id: uuid.UUID) -> str:
    clinic = db.get(Clinic, clinic_id)
    if clinic is None:
        raise ValueError(f"Unknown clinic_id {clinic_id}")
    return clinic.slug


def get_hospital_info_collection(db: Session, clinic_id: uuid.UUID) -> Collection:
    slug = _clinic_slug(db, clinic_id)
    return get_chroma_client().get_or_create_collection(hospital_info_collection_name(slug))


def get_collection_by_name(name: str) -> Collection:
    """Reaches a collection by its already-resolved name — used when a caller (e.g. a
    KBDocument row) already recorded which collection it belongs to, so re-deriving the
    name from clinic_id + a hardcoded kb_type would risk picking the wrong one."""
    return get_chroma_client().get_or_create_collection(name)
