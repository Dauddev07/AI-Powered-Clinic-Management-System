import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.core.rate_limit import limiter
from app.models.audit_log import AuditLog
from app.models.kb_document import KBDocument
from app.models.user import User
from app.rag.extraction import UnsupportedDocumentType
from app.schemas.kb_document import (
    KBDocumentDeleteOut,
    KBDocumentListOut,
    KBDocumentOut,
    KBDocumentUploadOut,
)
from app.services.kb_documents import delete_document_chunks, ingest_document

router = APIRouter(prefix="/admin/kb/documents", tags=["admin-kb"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = (".pdf", ".docx")


def _serialize(document: KBDocument) -> KBDocumentOut:
    return KBDocumentOut(
        id=document.id,
        filename=document.title,
        source_type=document.source_type,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )


@router.get("", response_model=KBDocumentListOut)
@limiter.limit("60/minute")
def list_documents(
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> KBDocumentListOut:
    stmt = (
        select(KBDocument)
        .where(KBDocument.clinic_id == current_user.clinic_id, KBDocument.source_type.in_(("pdf", "docx")))
        .order_by(KBDocument.created_at.desc())
    )
    items = list(db.execute(stmt).scalars().all())
    return KBDocumentListOut(items=[_serialize(d) for d in items])


@router.post("", response_model=KBDocumentUploadOut)
@limiter.limit("10/minute")
async def upload_document(
    file: UploadFile = File(...),
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> KBDocumentUploadOut:
    if not file.filename or not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a .pdf or .docx")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File exceeds the 20MB limit")

    clinic_id = current_user.clinic_id
    try:
        result = ingest_document(db, clinic_id, file.filename, content)
    except UnsupportedDocumentType as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:
        db.rollback()
        raise

    db.add(
        AuditLog(
            clinic_id=clinic_id,
            actor_user_id=current_user.id,
            action="kb_document_replaced" if result.replaced else "kb_document_uploaded",
            entity_type="kb_document",
            entity_id=result.document.id,
            metadata_json={"filename": file.filename, "chunk_count": result.document.chunk_count},
        )
    )
    db.commit()
    db.refresh(result.document)

    return KBDocumentUploadOut(document=_serialize(result.document), replaced=result.replaced)


@router.delete("/{document_id}", response_model=KBDocumentDeleteOut)
@limiter.limit("20/minute")
def delete_document(
    document_id: uuid.UUID,
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> KBDocumentDeleteOut:
    clinic_id = current_user.clinic_id
    document = db.execute(
        select(KBDocument).where(KBDocument.clinic_id == clinic_id, KBDocument.id == document_id)
    ).scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    delete_document_chunks(db, clinic_id, document)
    filename = document.title
    db.delete(document)

    db.add(
        AuditLog(
            clinic_id=clinic_id,
            actor_user_id=current_user.id,
            action="kb_document_deleted",
            entity_type="kb_document",
            entity_id=document_id,
            metadata_json={"filename": filename},
        )
    )
    db.commit()

    return KBDocumentDeleteOut(id=document_id, chunks_removed=True)
