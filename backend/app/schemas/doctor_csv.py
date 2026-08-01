import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RejectedRowOut(BaseModel):
    row_number: int
    external_doctor_id: str | None
    reason: str


class AcceptedRowOut(BaseModel):
    row_number: int
    external_doctor_id: str
    full_name: str
    department_name: str
    department_will_be_created: bool
    specialization: str | None
    is_active: bool
    shift_days: list[int]
    leave_dates_count: int


class HeaderSuggestionOut(BaseModel):
    original_header: str
    suggested_field: str
    confidence: float


class CSVPreviewOut(BaseModel):
    filename: str
    # True when one or more headers still need an explicit admin decision before
    # validation can run at all — accepted_rows/rejected_rows/etc. are meaningless
    # (left at their defaults) while this is true.
    needs_header_mapping: bool = False
    header_suggestions: list[HeaderSuggestionOut] = []
    unrecognized_headers: list[str] = []
    total_rows: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    accepted_rows: list[AcceptedRowOut] = []
    rejected_rows: list[RejectedRowOut] = []
    # Advisory only (e.g. same doctor name under different external_doctor_id values) —
    # never blocks Confirm.
    warnings: list[str] = []


class DeactivationSkippedOut(BaseModel):
    external_doctor_id: str
    full_name: str
    reason: str


class CSVConfirmOut(BaseModel):
    filename: str
    inserted: int
    updated: int
    marked_inactive: int
    rejected_count: int
    rejected_rows: list[RejectedRowOut]
    slots_inserted: int = 0
    slots_flagged_for_review: int = 0
    # Doctors absent from this upload that would normally be auto-deactivated, but
    # were left active because they still hold a confirmed appointment that
    # deactivation would orphan — resolve manually, then re-upload.
    deactivation_skipped: list[DeactivationSkippedOut] = []


class IngestionLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str | None
    uploaded_by_user_id: uuid.UUID | None
    rows_accepted: int | None
    rows_rejected: int | None
    status: str
    detail: str | None
    created_at: datetime


class IngestionLogListOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[IngestionLogOut]
