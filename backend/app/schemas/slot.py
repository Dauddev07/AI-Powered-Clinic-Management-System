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
    # None when the doctor has no AppointmentFeedback ratings yet, rather than 0 —
    # a doctor with zero ratings isn't "rated 0 stars", they're simply unrated.
    average_rating: float | None = None
    rating_count: int = 0


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
