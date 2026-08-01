import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class BookAppointmentRequest(BaseModel):
    slot_id: uuid.UUID
    reason: str | None = Field(default=None, max_length=1000)


class RescheduleRequest(BaseModel):
    new_slot_id: uuid.UUID


class AppointmentOut(BaseModel):
    id: uuid.UUID
    slot_id: uuid.UUID
    doctor_id: uuid.UUID
    doctor_name: str
    department_name: str
    start_utc: datetime
    end_utc: datetime
    status: str
    reason: str | None
    booked_via: str
    created_at: datetime
    updated_at: datetime
    cancelled_at: datetime | None


class AppointmentListOut(BaseModel):
    # start_utc/end_utc above are absolute instants, but the frontend needs the
    # clinic's timezone to render clock times the patient recognizes — the same
    # timezone the slots listing already carries, so this list is never rendered
    # against the viewer's browser-local timezone by mistake.
    clinic_timezone: str
    appointments: list[AppointmentOut]


class AppointmentHistoryOut(BaseModel):
    # Same clinic_timezone convention as AppointmentListOut — past/cancelled
    # appointments, newest first.
    clinic_timezone: str
    appointments: list[AppointmentOut]


class AppointmentSummaryOut(BaseModel):
    # Powers the patient dashboard's next-appointment card and upcoming/completed
    # counts. next_appointment is None rather than the list being empty, since
    # unlike AppointmentListOut this is a single soonest-first pick, not a list.
    clinic_timezone: str
    next_appointment: AppointmentOut | None
    upcoming_count: int
    completed_count: int
