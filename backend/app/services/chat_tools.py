"""The five LLM function-calling tools for booking, reschedule, cancel,
appointment-lookup, and availability (SOW task 6.2.4).

Design principles enforced throughout this module:

- THE LLM NEVER DECIDES; IT ONLY CALLS. book_appointment/reschedule_appointment/
  cancel_appointment each call straight into app.services.booking_engine — the exact
  same functions the REST endpoints use — so every booking rule (slot lock, overlap,
  no past slots, no same-day cancel, ownership, transactional reschedule, daily
  department cap) re-runs server-side exactly once, never re-implemented here.
- patient_id/clinic_id are captured from the ClinicContext (itself built only from
  the verified JWT — see app.core.tenancy) at tool-build time via closures. The
  model-facing tool schemas below deliberately have NO patient_id/clinic_id
  parameter, so there is nothing for a model to spoof even if it tried.
- book_appointment/reschedule_appointment/get_department_availability compose their
  OWN final reply text deterministically from real DB rows — the LLM is not asked to
  freehand doctor names, times, or confirmation details into prose, which is exactly
  the kind of hallucination-risk surface this project avoids everywhere else.
- get_my_appointments is the one read-only exception: it returns structured data for
  the calling agent loop to phrase conversationally, per the SOW's explicit "let the
  LLM summarize, never a raw dump" requirement for that one tool only.
"""
import json
import uuid
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from langsmith import traceable
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.clinic import Clinic
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.services import booking_engine
from app.services.chat_markers import BOOKING_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER
from app.services.department_availability import find_doctors_by_name, get_department_availability


def _format_when(start_utc: datetime, clinic_tz: str) -> str:
    local = start_utc.astimezone(ZoneInfo(clinic_tz))
    day = f"{local.strftime('%a, %b')} {local.day}"
    hour_12 = local.strftime("%I:%M %p").lstrip("0")
    return f"{day} at {hour_12}"


def _clinic_timezone(db: Session, clinic_id: uuid.UUID) -> str:
    clinic = db.get(Clinic, clinic_id)
    return clinic.timezone if clinic else "UTC"


def _department_name_for_slot(db: Session, clinic_id: uuid.UUID, slot: Slot) -> str | None:
    doctor = db.get(Doctor, slot.doctor_id)
    if doctor is None:
        return None
    from app.models.department import Department

    department = db.get(Department, doctor.department_id)
    return department.name if department else None


def _doctor_options_payload(db: Session, clinic_id: uuid.UUID, availability, note: str | None = None) -> str:
    tz = _clinic_timezone(db, clinic_id)
    payload = {
        "note": note,
        "department_name": availability.department_name,
        "doctors": [
            {
                "doctor_id": str(d.doctor_id),
                "doctor_name": d.full_name,
                "specialization": d.specialization,
                "slots": [
                    {"slot_id": str(s.slot_id), "when": _format_when(s.start_utc, tz)} for s in d.slots
                ],
            }
            for d in availability.doctors
        ],
    }
    return DOCTOR_OPTIONS_MARKER + json.dumps(payload, default=str)


def _booking_card_payload(db: Session, appointment, note: str | None = None) -> str:
    out = booking_engine.serialize_appointment(db, appointment)
    tz = _clinic_timezone(db, appointment.clinic_id)
    payload = {
        "note": note,
        "doctor_name": out.doctor_name,
        "department_name": out.department_name,
        "when": _format_when(out.start_utc, tz),
    }
    return BOOKING_MARKER + json.dumps(payload, default=str)


def _no_slots_message(department_name: str) -> str:
    return (
        f"I couldn't find any doctors with free upcoming slots in {department_name} right now. "
        "Please check back later or contact the clinic directly."
    )


def _no_slots_in_window_message(db: Session, clinic_id: uuid.UUID, department_name: str, next_available_when) -> str:
    if next_available_when is None:
        return _no_slots_message(department_name)
    tz = _clinic_timezone(db, clinic_id)
    return (
        f"No one in {department_name} has a free slot in that window. "
        f"The earliest open slot I can find is {_format_when(next_available_when, tz)}. "
        "Would you like me to book that instead, or check a different day?"
    )


def _department_not_found_message(availability) -> str:
    if availability.available_department_names:
        names = ", ".join(availability.available_department_names)
        return f"I couldn't find a department called that. The departments we have are: {names}."
    return "I couldn't find a matching department, and no departments are currently configured."


def _parse_date_arg(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        # An unparseable date is treated as "no filter" rather than an error — the
        # tool still returns real, current availability instead of failing the turn
        # over a malformed argument the model produced.
        return None


def combine_department_availability_results(raw_results: list[str]) -> str:
    """Assembles a DEPARTMENT_LIST:: payload from multiple get_department_availability
    calls made within one agent turn — built entirely in code from each call's own
    real result, never handed to the LLM to summarize itself (that's how a raw
    slot_id previously leaked into freehand prose). Results that aren't a
    DOCTOR_OPTIONS:: card (a not-found or no-free-slots message) are skipped rather
    than fabricated into a fake department entry."""
    departments = []
    for raw in raw_results:
        if not raw.startswith(DOCTOR_OPTIONS_MARKER):
            continue
        payload = json.loads(raw[len(DOCTOR_OPTIONS_MARKER):])
        departments.append({"department_name": payload["department_name"], "doctors": payload["doctors"]})

    if not departments:
        return (
            "I couldn't find any departments with doctors who have free upcoming slots right now. "
            "Please check back later or contact the clinic directly."
        )

    return DEPARTMENT_LIST_MARKER + json.dumps({"departments": departments}, default=str)


# ---------------------------------------------------------------------------
# Tool implementations (each @traceable so it appears as its own step in the
# LangSmith trace tree, alongside retrieval and LLM call steps).
# ---------------------------------------------------------------------------


@traceable(name="book_appointment")
def _book_appointment_impl(db: Session, ctx: ClinicContext, slot_id: str, reason: str | None) -> str:
    try:
        parsed_slot_id = uuid.UUID(slot_id)
    except (ValueError, TypeError, AttributeError):
        return "I couldn't recognize that slot. Could you pick one of the listed options again?"

    slot_before = db.get(Slot, parsed_slot_id)

    try:
        appointment = booking_engine.book_appointment(
            db,
            clinic_id=ctx.clinic_id,
            patient_id=ctx.user_id,
            slot_id=parsed_slot_id,
            reason=reason,
            booked_via="chatbot",
        )
    except HTTPException as exc:
        # Only "This slot is no longer available." is a genuine lost-race case (the
        # slot was taken between offer and confirmation) — matched on the exact
        # detail text, not just "any 409", because book_appointment also raises a
        # 409 when the slot the patient picked overlaps one of THEIR OWN existing
        # appointments (self-overlap, nothing to do with another patient). Treating
        # every 409 as a lost race used to swallow that overlap detail entirely and
        # show "that slot was just taken by another patient" instead — wrong and
        # misleading when the real reason was the patient's own double-booking.
        if exc.status_code == 409 and slot_before is not None and exc.detail == "This slot is no longer available.":
            # Lost race: the slot was taken between offer and confirmation. Never
            # report a booking that didn't happen — apologise and hand back fresh,
            # genuinely free alternatives in the same department, in the same turn.
            department_name = _department_name_for_slot(db, ctx.clinic_id, slot_before)
            if department_name:
                availability = get_department_availability(db, ctx.clinic_id, department_name)
                if availability.found and availability.doctors:
                    return _doctor_options_payload(
                        db,
                        ctx.clinic_id,
                        availability,
                        note="Sorry — that slot was just taken by another patient. Here are fresh options:",
                    )
                return "Sorry — that slot was just taken by another patient, and " + _no_slots_message(department_name)
        return exc.detail

    return _booking_card_payload(db, appointment)


@traceable(name="reschedule_appointment")
def _reschedule_appointment_impl(db: Session, ctx: ClinicContext, appointment_id: str, new_slot_id: str) -> str:
    try:
        parsed_appointment_id = uuid.UUID(appointment_id)
        parsed_new_slot_id = uuid.UUID(new_slot_id)
    except (ValueError, TypeError, AttributeError):
        return "I couldn't recognize that appointment or slot. Could you try again?"

    new_slot_before = db.get(Slot, parsed_new_slot_id)

    try:
        appointment = booking_engine.reschedule_appointment(
            db,
            clinic_id=ctx.clinic_id,
            patient_id=ctx.user_id,
            appointment_id=parsed_appointment_id,
            new_slot_id=parsed_new_slot_id,
        )
    except HTTPException as exc:
        if exc.status_code == 409 and new_slot_before is not None and exc.detail == "This slot is no longer available.":
            department_name = _department_name_for_slot(db, ctx.clinic_id, new_slot_before)
            if department_name:
                availability = get_department_availability(db, ctx.clinic_id, department_name)
                if availability.found and availability.doctors:
                    return _doctor_options_payload(
                        db,
                        ctx.clinic_id,
                        availability,
                        note="Sorry — that slot was just taken by another patient. Here are fresh options:",
                    )
                return "Sorry — that slot was just taken by another patient, and " + _no_slots_message(department_name)
        return exc.detail

    return _booking_card_payload(db, appointment, note="Your appointment has been rescheduled.")


@traceable(name="cancel_appointment")
def _cancel_appointment_impl(db: Session, ctx: ClinicContext, appointment_id: str) -> str:
    try:
        parsed_appointment_id = uuid.UUID(appointment_id)
    except (ValueError, TypeError, AttributeError):
        return "I couldn't recognize that appointment. Could you try again?"

    try:
        appointment = booking_engine.cancel_appointment(
            db, clinic_id=ctx.clinic_id, patient_id=ctx.user_id, appointment_id=parsed_appointment_id
        )
    except HTTPException as exc:
        # Verbatim pass-through, including the same-day-cancel refusal — must read
        # identically to what the REST /appointments/{id}/cancel endpoint returns.
        return exc.detail

    out = booking_engine.serialize_appointment(db, appointment)
    tz = _clinic_timezone(db, ctx.clinic_id)
    return f"Your appointment with {out.doctor_name} in {out.department_name} on {_format_when(out.start_utc, tz)} has been cancelled."


@traceable(name="get_my_appointments")
def _get_my_appointments_impl(db: Session, ctx: ClinicContext, status_filter: str, limit: int) -> str:
    from sqlalchemy import select

    from app.models.appointment import Appointment
    from app.services.appointments import auto_complete_past_appointments

    auto_complete_past_appointments(db, ctx.clinic_id)
    db.commit()

    now = datetime.now(timezone.utc)
    stmt = (
        select(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .where(Appointment.clinic_id == ctx.clinic_id, Appointment.patient_id == ctx.user_id)
    )

    normalized = (status_filter or "all").strip().lower()
    if normalized == "upcoming":
        stmt = stmt.where(Appointment.status == "confirmed", Slot.start_utc >= now).order_by(Slot.start_utc.asc())
    elif normalized == "past":
        stmt = stmt.where(Appointment.status == "completed").order_by(Slot.start_utc.desc())
    elif normalized == "cancelled":
        stmt = stmt.where(Appointment.status == "cancelled").order_by(Slot.start_utc.desc())
    else:
        stmt = stmt.order_by(Slot.start_utc.desc())

    stmt = stmt.limit(max(1, min(limit, 50)))
    appointments = db.execute(stmt).scalars().all()
    tz = _clinic_timezone(db, ctx.clinic_id)

    results = []
    for appointment in appointments:
        out = booking_engine.serialize_appointment(db, appointment)
        results.append(
            {
                "appointment_id": str(out.id),
                "doctor_name": out.doctor_name,
                "department_name": out.department_name,
                "when": _format_when(out.start_utc, tz),
                "status": out.status,
            }
        )

    return json.dumps({"appointments": results}, default=str)


@traceable(name="get_department_availability")
def _get_department_availability_impl(
    db: Session,
    ctx: ClinicContext,
    department_name: str,
    note: str | None = None,
    earliest_date: str | None = None,
    latest_date: str | None = None,
) -> str:
    availability = get_department_availability(
        db,
        ctx.clinic_id,
        department_name,
        earliest_date=_parse_date_arg(earliest_date),
        latest_date=_parse_date_arg(latest_date),
    )
    if not availability.found:
        return _department_not_found_message(availability)
    if not availability.doctors:
        if availability.next_available_when is not None:
            return _no_slots_in_window_message(
                db, ctx.clinic_id, availability.department_name, availability.next_available_when
            )
        return _no_slots_message(availability.department_name)
    # `note` carries the model's own one-sentence triage reasoning (e.g. "Based on
    # what you've described, this sounds like something Cardiology should look at")
    # so it renders in the DOCTOR_OPTIONS:: card BEFORE the doctor/slot list, never
    # after — the note field is placed first in the payload dict for exactly that
    # reason. A direct, non-triage availability question ("who's available in
    # Cardiology") simply omits it.
    return _doctor_options_payload(db, ctx.clinic_id, availability, note=note)


@traceable(name="find_doctors_by_name")
def _find_doctors_by_name_impl(db: Session, ctx: ClinicContext, name_query: str) -> str:
    matches = find_doctors_by_name(db, ctx.clinic_id, name_query)
    return json.dumps(
        {
            "matches": [
                {"doctor_name": m.full_name, "department_name": m.department_name} for m in matches
            ]
        }
    )


# ---------------------------------------------------------------------------
# LangChain tool schemas — arguments the model is allowed to supply. No
# patient_id/clinic_id field exists on any of these, by design.
# ---------------------------------------------------------------------------


class _BookArgs(BaseModel):
    slot_id: str = Field(description="The exact slot_id (UUID) of the slot the patient chose, from a previously shown list of options.")
    reason: str | None = Field(default=None, description="Short reason for the visit, if the patient mentioned one.")


class _RescheduleArgs(BaseModel):
    appointment_id: str = Field(description="The exact appointment_id (UUID) of the patient's existing appointment to move.")
    new_slot_id: str = Field(description="The exact slot_id (UUID) of the new slot the patient chose.")


class _CancelArgs(BaseModel):
    appointment_id: str = Field(description="The exact appointment_id (UUID) of the appointment to cancel.")


class _GetMyAppointmentsArgs(BaseModel):
    status: str = Field(
        default="all",
        description="One of: upcoming, past, cancelled, all. Infer from the patient's phrasing (e.g. 'next appointment' -> upcoming, 'last visit' -> past, 'my history' -> all).",
    )
    limit: int = Field(default=10, description="Max number of appointments to return.")


class _GetDepartmentAvailabilityArgs(BaseModel):
    department_name: str = Field(description="The real department name (e.g. 'Cardiology') — see system prompt TOOL USE RULES for how to resolve this.")
    note: str | None = Field(
        default=None,
        description=(
            "Optional. Either your one-sentence triage reasoning (omit if the patient named the "
            "department themselves), or a short answer to another question in the same message — "
            "full rules in the system prompt's `note` paragraphs. Never restate doctor names/slots here."
        ),
    )
    earliest_date: str | None = Field(
        default=None,
        description="Optional ISO date (YYYY-MM-DD) — only slots on/after this date. Use for 'a different day' follow-ups.",
    )
    latest_date: str | None = Field(
        default=None,
        description="Optional ISO date (YYYY-MM-DD), inclusive. Set with earliest_date for a bounded window (e.g. 'today or tomorrow').",
    )


class _FindDoctorsByNameArgs(BaseModel):
    name_query: str = Field(
        description="The doctor name as the patient typed it — don't correct spelling/word order first, let the search match on individual words."
    )


def build_tools(db: Session, ctx: ClinicContext) -> list[StructuredTool]:
    """Builds the six tools bound to this request's db session and verified
    ClinicContext via closures — clinic_id/patient_id are never parameters the model
    can set, they are captured here from the JWT-derived ctx."""

    def _book(slot_id: str, reason: str | None = None) -> str:
        return _book_appointment_impl(db, ctx, slot_id, reason)

    def _reschedule(appointment_id: str, new_slot_id: str) -> str:
        return _reschedule_appointment_impl(db, ctx, appointment_id, new_slot_id)

    def _cancel(appointment_id: str) -> str:
        return _cancel_appointment_impl(db, ctx, appointment_id)

    def _get_my_appointments(status: str = "all", limit: int = 10) -> str:
        return _get_my_appointments_impl(db, ctx, status, limit)

    def _get_department_availability(
        department_name: str,
        note: str | None = None,
        earliest_date: str | None = None,
        latest_date: str | None = None,
    ) -> str:
        return _get_department_availability_impl(db, ctx, department_name, note, earliest_date, latest_date)

    def _find_doctors_by_name(name_query: str) -> str:
        return _find_doctors_by_name_impl(db, ctx, name_query)

    return [
        StructuredTool.from_function(
            func=_book,
            name="book_appointment",
            description=(
                "Books an appointment for the patient on a specific slot_id that was already "
                "shown to them as an option. Only call this once the patient has clearly picked "
                "a specific slot."
            ),
            args_schema=_BookArgs,
        ),
        StructuredTool.from_function(
            func=_reschedule,
            name="reschedule_appointment",
            description="Reschedules one of the patient's existing confirmed appointments to a new slot_id.",
            args_schema=_RescheduleArgs,
        ),
        StructuredTool.from_function(
            func=_cancel,
            name="cancel_appointment",
            description="Cancels one of the patient's existing confirmed appointments.",
            args_schema=_CancelArgs,
        ),
        StructuredTool.from_function(
            func=_get_my_appointments,
            name="get_my_appointments",
            description=(
                "Looks up the patient's own appointment history/upcoming visits. Use for questions "
                "like 'when is my next appointment', 'when was my last visit', or 'show my history'. "
                "Returns structured data — summarize it conversationally; if it's empty, say so plainly "
                "and never invent an appointment."
            ),
            args_schema=_GetMyAppointmentsArgs,
        ),
        StructuredTool.from_function(
            func=_get_department_availability,
            name="get_department_availability",
            description=(
                "Looks up every ACTIVE doctor in a given real department with their genuinely free "
                "upcoming slots (real structured data, never a guess). For a cross-department question, "
                "call once per real department name and the results are combined into one reply "
                "automatically — never write that summary yourself. Full usage rules (note/earliest_date/"
                "latest_date, doctor-name resolution) are in the system prompt's TOOL USE RULES."
            ),
            args_schema=_GetDepartmentAvailabilityArgs,
        ),
        StructuredTool.from_function(
            func=_find_doctors_by_name,
            name="find_doctors_by_name",
            description=(
                "Searches real, active doctors by name (word-level match, order-independent — e.g. "
                "'raza iqra' matches 'Dr. Iqra Raza'). Call this BEFORE asking anything whenever a "
                "patient names a doctor who isn't already an exact match in context. Returns 0/1/many "
                "matches — see system prompt TOOL USE RULES for how to respond to each case."
            ),
            args_schema=_FindDoctorsByNameArgs,
        ),
    ]
