import uuid
from datetime import datetime

from pydantic import BaseModel


class KBDocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    source_type: str
    chunk_count: int
    created_at: datetime


class KBDocumentListOut(BaseModel):
    items: list[KBDocumentOut]


class KBDocumentUploadOut(BaseModel):
    document: KBDocumentOut
    replaced: bool


class KBDocumentDeleteOut(BaseModel):
    id: uuid.UUID
    chunks_removed: bool
