import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PendingFeedbackAppointment(BaseModel):
    appointment_id: uuid.UUID
    doctor_name: str
    when: str


class PendingFeedbackOut(BaseModel):
    appointments: list[PendingFeedbackAppointment]
    prompt: str | None


class FeedbackSubmitIn(BaseModel):
    appointment_ids: list[uuid.UUID] = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    reason: str | None = None


class FeedbackSubmitOut(BaseModel):
    message: str


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_name: str
    doctor_name: str
    rating: int
    reason: str | None
    created_at: datetime


class FeedbackToneCounts(BaseModel):
    good: int
    neutral: int
    bad: int


class FeedbackListOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[FeedbackOut]
    # Clinic-wide (never affected by the `tone` filter or pagination) so the summary
    # tiles above the table always reflect every rating on file, not just what's
    # currently shown.
    average_rating: float | None
    tone_counts: FeedbackToneCounts
