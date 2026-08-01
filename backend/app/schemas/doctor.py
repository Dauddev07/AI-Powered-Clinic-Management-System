import uuid

from pydantic import BaseModel


class DoctorOut(BaseModel):
    id: uuid.UUID
    external_doctor_id: str
    full_name: str
    department_name: str
    specialization: str | None
    is_active: bool


class DoctorListOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[DoctorOut]


class DoctorStatusUpdate(BaseModel):
    is_active: bool
