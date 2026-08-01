"""Single place notification rows get written from — booking_engine.py's book/
reschedule/cancel paths and the scheduler's appointment-auto-complete tick all call
create_notification() rather than constructing Notification rows themselves, so the
message format and read/unread defaults never drift between call sites.
"""
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.models.notification import Notification


def format_appointment_datetime(start_utc: datetime, tz_name: str) -> str:
    """e.g. "5 Aug at 10:00 AM" — clinic-local, matching how every other patient-
    facing screen in this stack renders appointment times (never the viewer's own
    browser timezone).
    """
    local = start_utc.astimezone(ZoneInfo(tz_name))
    return f"{local.day} {local.strftime('%b')} at {local.strftime('%I:%M %p').lstrip('0')}"


def create_notification(
    db: Session,
    clinic_id: uuid.UUID,
    user_id: uuid.UUID,
    notif_type: str,
    message: str,
    related_appointment_id: uuid.UUID | None = None,
) -> Notification:
    """Adds and flushes the row but never commits — callers write this in the same
    transaction as the booking action that triggered it, so the notification and the
    appointment state change land together or not at all.
    """
    notification = Notification(
        clinic_id=clinic_id,
        user_id=user_id,
        type=notif_type,
        message=message,
        related_appointment_id=related_appointment_id,
    )
    db.add(notification)
    db.flush()
    return notification
