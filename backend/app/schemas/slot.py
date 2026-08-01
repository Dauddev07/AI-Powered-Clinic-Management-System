import uuid
from datetime import datetime

from pydantic import BaseModel


class SlotOut(BaseModel):
    id: uuid.UUID
    doctor_id: uuid.UUID
    doctor_name: str
    department_id: uuid.UUID
    department_name: str
    specialization: str | None
    start_utc: datetime
    end_utc: datetime
    status: str
    is_bookable: bool


class SlotListOut(BaseModel):
    clinic_timezone: str
    total: int
    limit: int
    offset: int
    slots: list[SlotOut]


class DepartmentOptionOut(BaseModel):
    id: uuid.UUID
    name: str


class DoctorOptionOut(BaseModel):
    id: uuid.UUID
    name: str
