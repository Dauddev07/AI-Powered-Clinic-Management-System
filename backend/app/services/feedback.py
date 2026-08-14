"""Post-appointment star-rating feedback: prompting the patient the first time they
open the chatbot after a completed appointment, and recording their answer.

Deliberately decoupled from the chat-turn/LLM pipeline (app.services.chat) rather
than woven into it — the prompt is asked once, plainly, as soon as the chat screen is
opened ("before anything"), and doesn't need any LLM reasoning, tool routing, or
red_flag/diagnosis_guard interaction. The frontend fetches it via a small dedicated
endpoint on mount and renders it as a synthetic first bubble, before any real message
has been sent.
"""
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.slot import Slot

_MAX_PENDING_APPOINTMENTS = 1

BAD_RATING_ACK = "Thanks for letting us know — we've shared this with the clinic so they can look into it."
GOOD_RATING_ACK = "Thank you for the feedback! We're glad your visit went well."
NEUTRAL_RATING_ACK = "Thanks for the feedback!"


def _format_when(start_utc: datetime, tz_name: str) -> str:
    local = start_utc.astimezone(ZoneInfo(tz_name))
    return f"{local.day} {local.strftime('%b')} at {local.strftime('%I:%M %p').lstrip('0')}"


def get_pending_feedback_appointments(db: Session, ctx: ClinicContext) -> list[dict]:
    """At most the single most recently completed appointment this patient hasn't
    rated yet — never a backlog of every unrated appointment at once. "Hasn't rated
    yet" is purely the absence of a matching AppointmentFeedback row; there's no
    separate "already asked" flag to track or reset, and any older unrated
    appointments simply stay unrated (never surfaced) once a newer one exists.
    """
    clinic = db.get(Clinic, ctx.clinic_id)
    tz_name = clinic.timezone if clinic else "UTC"

    rated_appointment_ids = select(AppointmentFeedback.appointment_id).where(
        AppointmentFeedback.clinic_id == ctx.clinic_id, AppointmentFeedback.patient_id == ctx.user_id
    )

    rows = db.execute(
        select(Appointment, Slot, Doctor)
        .join(Slot, Slot.id == Appointment.slot_id)
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .where(
            Appointment.clinic_id == ctx.clinic_id,
            Appointment.patient_id == ctx.user_id,
            Appointment.status == "completed",
            Appointment.id.not_in(rated_appointment_ids),
        )
        .order_by(Slot.start_utc.desc())
        .limit(_MAX_PENDING_APPOINTMENTS)
    ).all()

    return [
        {
            "appointment_id": appointment.id,
            "doctor_name": doctor.full_name,
            "when": _format_when(slot.start_utc, tz_name),
        }
        for appointment, slot, doctor in rows
    ]


def build_feedback_prompt(appointments: list[dict]) -> str | None:
    if not appointments:
        return None
    appt = appointments[0]
    return f"Before we continue — how was your recent appointment with {appt['doctor_name']} on {appt['when']}?"


def submit_feedback(
    db: Session,
    ctx: ClinicContext,
    appointment_ids: list[uuid.UUID],
    rating: int,
    reason: str | None,
) -> str:
    """Writes one AppointmentFeedback row per appointment_id — in practice always a
    single id, since get_pending_feedback_appointments only ever surfaces one
    appointment at a time, but this stays list-shaped rather than a single id to
    match how the frontend already sends it. Returns the acknowledgement message to
    show the patient. Silently skips any id that doesn't belong to this patient,
    isn't completed, or was already rated, rather than erroring — a stale/duplicate
    submit (e.g. a double-click) should never surface an error to the patient.
    """
    rows = db.execute(
        select(Appointment).where(
            Appointment.id.in_(appointment_ids),
            Appointment.clinic_id == ctx.clinic_id,
            Appointment.patient_id == ctx.user_id,
            Appointment.status == "completed",
        )
    ).scalars().all()

    already_rated = set(
        db.execute(
            select(AppointmentFeedback.appointment_id).where(
                AppointmentFeedback.appointment_id.in_([a.id for a in rows])
            )
        ).scalars().all()
    )

    for appointment in rows:
        if appointment.id in already_rated:
            continue
        db.add(
            AppointmentFeedback(
                clinic_id=ctx.clinic_id,
                patient_id=ctx.user_id,
                appointment_id=appointment.id,
                doctor_id=appointment.doctor_id,
                rating=rating,
                reason=reason if rating <= 2 else None,
            )
        )
    db.commit()

    if rating <= 2:
        return BAD_RATING_ACK
    if rating >= 4:
        return GOOD_RATING_ACK
    return NEUTRAL_RATING_ACK
