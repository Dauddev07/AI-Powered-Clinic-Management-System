import json
import uuid
from datetime import datetime, time, timedelta, timezone

import pytest

from app.core.tenancy import ClinicContext
from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services.chat_markers import BOOKING_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER, NO_SLOTS_MARKER
from app.services.chat_tools import (
    _book_appointment_impl,
    _cancel_appointment_impl,
    _find_doctors_by_name_impl,
    _get_department_availability_impl,
    _get_my_appointments_impl,
    _reschedule_appointment_impl,
    build_tools,
    combine_department_availability_results,
    ensure_slot_pick_example_has_a_date,
    resolve_bare_weekday_window,
    resolve_date_window,
    resolve_time_of_day_window,
)
from app.services.department_availability import MAX_SLOTS_PER_DOCTOR


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def department(db, clinic):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()
    return dept


@pytest.fixture
def doctor(db, clinic, department):
    doc = Doctor(
        clinic_id=clinic.id,
        department_id=department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Jane Example",
        is_active=True,
    )
    db.add(doc)
    db.flush()
    return doc


@pytest.fixture
def patient(db, clinic):
    p = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(p)
    db.flush()
    return p


@pytest.fixture
def ctx(clinic, patient):
    return ClinicContext(clinic_id=clinic.id, user_id=patient.id, role="patient")


def _slot(db, clinic, doctor, start_utc, status="open"):
    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start_utc,
        end_utc=start_utc + timedelta(minutes=30), status=status,
    )
    db.add(slot)
    db.flush()
    return slot


def _appointment(db, clinic, patient, doctor, slot, status="confirmed"):
    appt = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status=status,
    )
    db.add(appt)
    db.flush()
    return appt


def _parse_marker(content, marker):
    assert content.startswith(marker)
    return json.loads(content[len(marker):])


# --- book_appointment: happy path ---------------------------------------------------


def test_book_appointment_success_returns_booking_card(db, clinic, doctor, patient, ctx):
    slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    result = _book_appointment_impl(db, ctx, str(slot.id), "Checkup")

    payload = _parse_marker(result, BOOKING_MARKER)
    assert payload["doctor_name"] == "Dr. Jane Example"
    assert payload["department_name"] == "Cardiology"
    assert payload["when"]


def test_book_appointment_invalid_slot_id_gives_friendly_error_not_a_crash(db, ctx):
    result = _book_appointment_impl(db, ctx, "not-a-uuid", None)

    assert "couldn't recognize" in result.lower()
    assert not result.startswith(BOOKING_MARKER)


# --- book_appointment: lost-race handling --------------------------------------------


def test_book_appointment_lost_race_apologizes_and_offers_fresh_alternatives(db, clinic, department, doctor, patient, ctx):
    # Simulate the slot being taken by someone else between the bot offering it and
    # the patient confirming: the slot the model is told to book is already booked.
    taken_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1), status="booked")
    fresh_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2), status="open")

    result = _book_appointment_impl(db, ctx, str(taken_slot.id), None)

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    assert "taken by another patient" in payload["note"].lower()
    assert payload["department_name"] == "Cardiology"
    slot_ids = {s["slot_id"] for doc in payload["doctors"] for s in doc["slots"]}
    assert str(fresh_slot.id) in slot_ids
    assert str(taken_slot.id) not in slot_ids


def test_book_appointment_lost_race_with_no_fresh_alternatives_gives_plain_apology(db, clinic, doctor, patient, ctx):
    taken_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1), status="booked")
    # No other open slot exists for this doctor/department.

    result = _book_appointment_impl(db, ctx, str(taken_slot.id), None)

    assert not result.startswith(DOCTOR_OPTIONS_MARKER)
    assert not result.startswith(BOOKING_MARKER)
    assert "taken by another patient" in result.lower()


def test_book_appointment_self_overlap_returns_overlap_message_not_taken_by_another_patient(
    db, clinic, department, doctor, patient, ctx
):
    # Regression: booking_engine raises the same 409 status for BOTH "someone else
    # just took this slot" and "this overlaps your own existing appointment" — the
    # handler used to treat any 409 as a lost race and always say "taken by another
    # patient", silently discarding the real self-overlap detail.
    existing_start = datetime.now(timezone.utc) + timedelta(days=1)
    existing_slot = _slot(db, clinic, doctor, existing_start, status="booked")
    _appointment(db, clinic, patient, doctor, existing_slot)

    # Overlaps the existing 30-minute appointment (starts 10 minutes into it) and is
    # itself genuinely open — nothing to do with a race against another patient.
    overlapping_slot = _slot(db, clinic, doctor, existing_start + timedelta(minutes=10), status="open")

    result = _book_appointment_impl(db, ctx, str(overlapping_slot.id), None)

    assert result == "You already have an appointment that overlaps with this time."
    assert "taken by another patient" not in result.lower()
    assert not result.startswith(DOCTOR_OPTIONS_MARKER)


# --- reschedule_appointment: lost-race handling --------------------------------------


def test_reschedule_lost_race_apologizes_and_offers_fresh_alternatives(db, clinic, department, doctor, patient, ctx, monkeypatch):
    old_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=3))
    appointment = _appointment(db, clinic, patient, doctor, old_slot)

    taken_new_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1), status="booked")
    fresh_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2), status="open")

    # booking_engine.reschedule_appointment rolls back its own transaction on ANY
    # failure, by design (see its docstring: "never zero, never two") — under this
    # test harness's SAVEPOINT-per-test isolation (see conftest.py), that rollback
    # would also wipe out this test's own not-yet-committed fixture rows, since
    # nothing here has been committed at the top level either. Stubbing the engine
    # call to raise the exact same 409 it would raise for a taken slot — without
    # actually touching the DB — lets this test verify chat_tools' OWN lost-race
    # composition logic (re-query, apologize, offer fresh alternatives) in isolation
    # from that unrelated test-harness transaction quirk.
    from app.services import chat_tools

    def _raise_slot_taken(*args, **kwargs):
        from fastapi import HTTPException

        raise HTTPException(status_code=409, detail="This slot is no longer available.")

    monkeypatch.setattr(chat_tools.booking_engine, "reschedule_appointment", _raise_slot_taken)

    result = _reschedule_appointment_impl(db, ctx, str(appointment.id), str(taken_new_slot.id))

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    assert "taken by another patient" in payload["note"].lower()
    slot_ids = {s["slot_id"] for doc in payload["doctors"] for s in doc["slots"]}
    assert str(fresh_slot.id) in slot_ids


# --- cancel_appointment: verbatim REST-message pass-through --------------------------


def test_cancel_appointment_within_cutoff_refusal_matches_rest_endpoint_message(db, clinic, doctor, patient, ctx):
    soon_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(hours=1))
    appointment = _appointment(db, clinic, patient, doctor, soon_slot)

    result = _cancel_appointment_impl(db, ctx, str(appointment.id))

    assert result == (
        "Appointments cannot be cancelled within 2 hours of the appointment time. "
        "Please contact the clinic directly for last-minute changes."
    )


def test_cancel_appointment_later_today_but_outside_cutoff_now_succeeds(db, clinic, doctor, patient, ctx):
    # The restriction is a flat 2-hour window by actual time remaining, not "same
    # calendar day" — a slot 6 hours away, even though still "today," must now be
    # cancellable, which the old same-day rule would have wrongly refused.
    later_today_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(hours=6))
    appointment = _appointment(db, clinic, patient, doctor, later_today_slot)

    result = _cancel_appointment_impl(db, ctx, str(appointment.id))

    assert "cannot be cancelled" not in result
    assert "has been cancelled" in result


def test_cancel_appointment_within_cutoff_refused_regardless_of_calendar_day(db, clinic, doctor, patient, ctx):
    # The flip side: a slot within 2 hours from now must still be blocked even if it
    # falls on the NEXT calendar day (e.g. a request made at 11:15pm for a 12:30am
    # slot) — the old same-day-only rule would have wrongly allowed this.
    soon_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(minutes=90))
    appointment = _appointment(db, clinic, patient, doctor, soon_slot)

    result = _cancel_appointment_impl(db, ctx, str(appointment.id))

    assert result == (
        "Appointments cannot be cancelled within 2 hours of the appointment time. "
        "Please contact the clinic directly for last-minute changes."
    )


def test_reschedule_appointment_within_cutoff_refusal_matches_rest_endpoint_message(
    db, clinic, doctor, patient, ctx
):
    soon_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(hours=1))
    new_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2))
    appointment = _appointment(db, clinic, patient, doctor, soon_slot)

    result = _reschedule_appointment_impl(db, ctx, str(appointment.id), str(new_slot.id))

    assert result == (
        "Appointments cannot be rescheduled within 2 hours of the appointment time. "
        "Please contact the clinic directly for last-minute changes."
    )


def test_reschedule_appointment_later_today_but_outside_cutoff_now_succeeds(db, clinic, doctor, patient, ctx):
    later_today_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(hours=6))
    new_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2))
    appointment = _appointment(db, clinic, patient, doctor, later_today_slot)

    result = _reschedule_appointment_impl(db, ctx, str(appointment.id), str(new_slot.id))

    assert "cannot be rescheduled" not in result


def test_cancel_appointment_success_gives_plain_confirmation_sentence(db, clinic, doctor, patient, ctx):
    future_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2))
    appointment = _appointment(db, clinic, patient, doctor, future_slot)

    result = _cancel_appointment_impl(db, ctx, str(appointment.id))

    assert "Dr. Jane Example" in result
    assert "Cardiology" in result
    assert "cancelled" in result.lower()


# --- list_upcoming_appointments (appointment_agent's deterministic ambiguity check) --


def test_list_upcoming_appointments_returns_only_confirmed_future_ones(db, clinic, doctor, patient, ctx):
    future_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2))
    past_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) - timedelta(days=2))
    cancelled_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=3))
    _appointment(db, clinic, patient, doctor, future_slot)
    _appointment(db, clinic, patient, doctor, past_slot, status="completed")
    _appointment(db, clinic, patient, doctor, cancelled_slot, status="cancelled")

    from app.services.chat_tools import list_upcoming_appointments

    result = list_upcoming_appointments(db, ctx)

    assert len(result) == 1
    assert result[0]["doctor_name"] == "Dr. Jane Example"
    assert result[0]["department_name"] == "Cardiology"
    assert result[0]["appointment_id"]
    assert result[0]["when"]


def test_list_upcoming_appointments_empty_when_none(db, ctx):
    from app.services.chat_tools import list_upcoming_appointments

    assert list_upcoming_appointments(db, ctx) == []


# --- build_tools: reschedule/cancel redirect (deterministic tool-level enforcement) --


def test_build_tools_redirects_book_appointment_to_reschedule_when_redirect_id_set(
    db, clinic, doctor, patient, ctx
):
    # Reported live: the model still called book_appointment for a reschedule's
    # slot-pick despite an explicit prompt rule against it, creating a stray
    # duplicate appointment instead of moving the existing one. This is the
    # deterministic enforcement that makes the outcome correct regardless of which
    # tool the model actually calls.
    existing_start = datetime.now(timezone.utc) + timedelta(days=1)
    existing_slot = _slot(db, clinic, doctor, existing_start)
    appointment = _appointment(db, clinic, patient, doctor, existing_slot)
    # Reschedules are same-day-only now — a couple hours later, same calendar
    # day as existing_slot, not a different day.
    shift = timedelta(hours=-2) if existing_start.hour >= 20 else timedelta(hours=2)
    new_slot = _slot(db, clinic, doctor, existing_start + shift)

    tools = build_tools(db, ctx, reschedule_redirect_appointment_id=str(appointment.id))
    book_tool = next(t for t in tools if t.name == "book_appointment")

    result = book_tool.invoke({"slot_id": str(new_slot.id)})

    payload = _parse_marker(result, BOOKING_MARKER)
    assert payload["note"] == "Your appointment has been rescheduled."
    # Still exactly one appointment for this patient — not a new, second one.
    from sqlalchemy import select

    from app.models.appointment import Appointment

    all_appointments = db.execute(
        select(Appointment).where(Appointment.patient_id == patient.id, Appointment.status == "confirmed")
    ).scalars().all()
    assert len(all_appointments) == 1
    assert all_appointments[0].slot_id == new_slot.id


def test_build_tools_forces_cancel_appointment_to_use_redirect_id_regardless_of_model_arg(
    db, clinic, doctor, patient, ctx
):
    real_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    real_appointment = _appointment(db, clinic, patient, doctor, real_slot)

    tools = build_tools(db, ctx, cancel_redirect_appointment_id=str(real_appointment.id))
    cancel_tool = next(t for t in tools if t.name == "cancel_appointment")

    # The model supplies a bogus/wrong id — the redirect must still win.
    result = cancel_tool.invoke({"appointment_id": str(uuid.uuid4())})

    assert "has been cancelled" in result
    assert "Dr. Jane Example" in result


def test_build_tools_cancel_appointment_refuses_without_a_verified_redirect_id(db, clinic, doctor, patient, ctx):
    # Reported live: a cancel-confirmation question was asked ("...let me know if
    # you'd like me to go ahead and cancel it"), the patient then asked an unrelated
    # question, got answered, and only THEN said "yes" — three turns later.
    # needs_booking_action_tools' full-history marker scan (message_classifier.py)
    # kept cancel_appointment bound for that later turn despite the confirmation no
    # longer being live, and the LLM — seeing its own earlier question sitting in
    # conversation history — called cancel_appointment on its own initiative,
    # supplying whatever appointment_id it inferred, and the real appointment got
    # cancelled with no fresh, code-verified confirmation at all. Without
    # cancel_redirect_appointment_id set (i.e. no deterministic, live confirmation
    # from appointment_agent's own resolution), the tool must refuse to cancel
    # anything — regardless of what appointment_id the model supplies — rather
    # than trusting the model's own judgment for a real mutating action.
    real_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    real_appointment = _appointment(db, clinic, patient, doctor, real_slot)

    tools = build_tools(db, ctx)  # no cancel_redirect_appointment_id
    cancel_tool = next(t for t in tools if t.name == "cancel_appointment")

    result = cancel_tool.invoke({"appointment_id": str(real_appointment.id)})

    assert "has been cancelled" not in result
    assert "cancel" in result.lower()


def test_build_tools_reschedule_appointment_refuses_without_a_verified_redirect_id(
    db, clinic, doctor, patient, ctx
):
    # Same vulnerability class and same fix as cancel's own test just above,
    # applied to reschedule: reschedule_redirect_appointment_id is ONLY ever set by
    # appointment_agent's own deterministic resolution — the `or appointment_id`
    # fallback that used to let the model supply its own appointment_id had no
    # legitimate use case, and was exploitable the same way (a stale
    # "reschedule_confirm" marker no longer the assistant's literal last turn,
    # then an unrelated later "yes" the model could still act on).
    real_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    real_appointment = _appointment(db, clinic, patient, doctor, real_slot)
    new_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2))

    tools = build_tools(db, ctx)  # no reschedule_redirect_appointment_id
    reschedule_tool = next(t for t in tools if t.name == "reschedule_appointment")

    result = reschedule_tool.invoke({"appointment_id": str(real_appointment.id), "new_slot_id": str(new_slot.id)})

    assert "rescheduled" not in result.lower()
    db.refresh(real_appointment)
    assert real_appointment.slot_id == real_slot.id

    db.refresh(real_appointment)
    assert real_appointment.status == "confirmed"


def test_build_tools_book_appointment_refuses_when_suppressed_for_a_bare_confirmation(
    db, clinic, doctor, patient, ctx
):
    # Reported live: "book with Dr. X at 3pm" -> confirm question -> an unrelated
    # question (correctly answered) -> "Yeah" (correctly not booked) -> "Yes" —
    # booked for real, from a confirmation question that was two turns stale. The
    # model read its own earlier "would you like me to book...?" question still
    # sitting in conversation history and decided this later "yes" answered it,
    # extracting the slot_id straight out of that stale marker's own JSON text.
    # appointment_agent sets suppress_bare_confirmation_booking exactly for this
    # shape of turn (a bare yes/no that resolved nothing live) — the tool must
    # refuse outright rather than book whatever slot_id the model supplies.
    real_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    tools = build_tools(db, ctx, suppress_bare_confirmation_booking=True)
    book_tool = next(t for t in tools if t.name == "book_appointment")

    result = book_tool.invoke({"slot_id": str(real_slot.id)})

    assert "booked" not in result.lower()
    db.refresh(real_slot)
    assert real_slot.status == "open"


def test_build_tools_book_appointment_still_works_for_a_genuine_natural_language_pick(
    db, clinic, doctor, patient, ctx
):
    # Guard against the new gate being too broad: a real slot description (not a
    # bare yes/no) must still book normally — suppress_bare_confirmation_booking
    # defaults to False and is only ever set True by appointment_agent for the
    # exact bare-confirmation case above.
    real_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    tools = build_tools(db, ctx)  # suppress_bare_confirmation_booking defaults False
    book_tool = next(t for t in tools if t.name == "book_appointment")

    result = book_tool.invoke({"slot_id": str(real_slot.id)})

    payload = _parse_marker(result, BOOKING_MARKER)
    assert payload["doctor_name"] == doctor.full_name


def test_build_tools_without_redirect_ids_behaves_exactly_as_before(db, clinic, doctor, patient, ctx):
    slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    tools = build_tools(db, ctx)
    book_tool = next(t for t in tools if t.name == "book_appointment")

    result = book_tool.invoke({"slot_id": str(slot.id)})

    payload = _parse_marker(result, BOOKING_MARKER)
    assert payload.get("note") is None


# --- get_department_availability: note threading -------------------------------------


def test_get_department_availability_threads_reasoning_note_into_payload(db, clinic, department, doctor, patient, ctx):
    _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    reasoning = "Based on what you've described, this sounds like something Cardiology should look at."
    result = _get_department_availability_impl(db, ctx, "Cardiology", reasoning)

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    assert payload["note"] == reasoning


def test_get_department_availability_without_note_omits_it(db, clinic, department, doctor, patient, ctx):
    _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    result = _get_department_availability_impl(db, ctx, "Cardiology")

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    assert payload["note"] is None


def test_get_department_availability_unknown_department_is_plain_text_not_a_card(db, ctx):
    result = _get_department_availability_impl(db, ctx, "Neurology")

    assert not result.startswith(DOCTOR_OPTIONS_MARKER)
    assert not result.startswith(NO_SLOTS_MARKER)
    assert "couldn't find" in result.lower()


def test_get_department_availability_found_but_no_free_slots_is_tagged_no_slots_marker(db, clinic, department, ctx):
    # department exists (has a real doctor) but that doctor has no upcoming free
    # slots — distinct from "department doesn't exist" (test above), and must stay
    # distinguishable so a multi-department turn can still surface it instead of
    # silently dropping it (see combine_department_availability_results below).
    result = _get_department_availability_impl(db, ctx, "Cardiology")

    assert result.startswith(NO_SLOTS_MARKER)
    payload = json.loads(result[len(NO_SLOTS_MARKER):])
    assert payload["department_name"] == "Cardiology"
    assert "couldn't find any doctors" in payload["message"].lower()


# --- get_my_appointments: structured data, never a raw dump on empty ----------------


def test_get_my_appointments_empty_returns_empty_structured_list(db, ctx):
    result = _get_my_appointments_impl(db, ctx, "upcoming", 10)

    parsed = json.loads(result)
    assert parsed == {"appointments": []}


def test_get_my_appointments_returns_structured_upcoming_appointment(db, clinic, doctor, patient, ctx):
    future_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    _appointment(db, clinic, patient, doctor, future_slot)

    result = _get_my_appointments_impl(db, ctx, "upcoming", 10)

    parsed = json.loads(result)
    assert len(parsed["appointments"]) == 1
    assert parsed["appointments"][0]["doctor_name"] == "Dr. Jane Example"
    assert parsed["appointments"][0]["status"] == "confirmed"


def test_get_my_appointments_includes_appointment_id_for_reschedule_cancel(db, clinic, doctor, patient, ctx):
    # Reproduces a reported bug: the prompt tells the model to get the appointment_id
    # from a fresh get_my_appointments call before rescheduling/cancelling, but the
    # tool's returned JSON omitted the id entirely, so the model had nothing real to
    # pass to reschedule_appointment and the backend rejected it with "Appointment not
    # found." even though the patient's appointment did exist.
    future_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    appointment = _appointment(db, clinic, patient, doctor, future_slot)

    result = _get_my_appointments_impl(db, ctx, "upcoming", 10)

    parsed = json.loads(result)
    assert parsed["appointments"][0]["appointment_id"] == str(appointment.id)


def test_get_my_appointments_cancelled_filter_orders_by_when_it_was_cancelled_not_slot_time(
    db, clinic, doctor, patient, ctx
):
    # Reported live: "what's my most recent cancelled appointment" was ordered
    # by the appointment's original SLOT time, not by when it was actually
    # cancelled — a patient who cancels an appointment scheduled FURTHER OUT
    # after already having cancelled one scheduled SOONER would get the wrong
    # one shown first. Two appointments are cancelled here in the OPPOSITE
    # order of their slot times, to prove the result reflects cancellation
    # recency, not slot recency.
    sooner_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    later_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=10))
    sooner_appt = _appointment(db, clinic, patient, doctor, sooner_slot, status="cancelled")
    later_appt = _appointment(db, clinic, patient, doctor, later_slot, status="cancelled")

    # sooner_appt (day+1) cancelled FIRST, later_appt (day+10) cancelled SECOND
    # (most recently) — despite later_appt's slot being further in the future.
    sooner_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(hours=2)
    later_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.flush()

    result = _get_my_appointments_impl(db, ctx, "cancelled", 10)

    parsed = json.loads(result)
    assert [a["appointment_id"] for a in parsed["appointments"]] == [
        str(later_appt.id),
        str(sooner_appt.id),
    ]


def test_get_my_appointments_oldest_first_reverses_the_cancelled_ordering(db, clinic, doctor, patient, ctx):
    # Reported live: "what's my EARLIEST cancelled appointment" needs the exact
    # opposite of "most recent" — same two appointments/timestamps as the test
    # above, but oldest_first=True must put sooner_appt (cancelled first) ahead
    # of later_appt (cancelled more recently).
    sooner_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))
    later_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=10))
    sooner_appt = _appointment(db, clinic, patient, doctor, sooner_slot, status="cancelled")
    later_appt = _appointment(db, clinic, patient, doctor, later_slot, status="cancelled")

    sooner_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(hours=2)
    later_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.flush()

    result = _get_my_appointments_impl(db, ctx, "cancelled", 10, oldest_first=True)

    parsed = json.loads(result)
    assert [a["appointment_id"] for a in parsed["appointments"]] == [
        str(sooner_appt.id),
        str(later_appt.id),
    ]


def test_get_my_appointments_past_filter_orders_by_when_it_was_completed_not_slot_time(
    db, clinic, doctor, patient, ctx
):
    # Same class of bug as the cancelled-filter test above, for "past"/
    # completed appointments: order must reflect when the visit was actually
    # marked completed (booking_engine.confirm_visit bumps updated_at), not
    # the slot's original scheduled time.
    sooner_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) - timedelta(days=1))
    later_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) - timedelta(days=10))
    sooner_appt = _appointment(db, clinic, patient, doctor, sooner_slot, status="completed")
    later_appt = _appointment(db, clinic, patient, doctor, later_slot, status="completed")
    db.flush()

    # later_appt's slot was further in the past, but it was confirmed as
    # completed by the patient MOST RECENTLY (right now) — sooner_appt was
    # confirmed completed earlier.
    sooner_appt.updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
    later_appt.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.flush()

    result = _get_my_appointments_impl(db, ctx, "past", 10)

    parsed = json.loads(result)
    assert [a["appointment_id"] for a in parsed["appointments"]] == [
        str(later_appt.id),
        str(sooner_appt.id),
    ]


def test_get_my_appointments_missed_filter_returns_only_no_show_ordered_by_recency(
    db, clinic, doctor, patient, ctx
):
    # Instructed live: "what appointments have I missed" had no dedicated
    # filter value at all — 'no_show' (booking_engine.confirm_visit's
    # completed=False path) fell through to the `else` branch, which returns
    # every status mixed together, never isolating just the missed ones.
    # A "past"/completed appointment must never appear in "missed" results,
    # and vice versa (see the sibling "past" test above).
    completed_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) - timedelta(days=1))
    _appointment(db, clinic, patient, doctor, completed_slot, status="completed")

    sooner_missed_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) - timedelta(days=5))
    later_missed_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) - timedelta(days=20))
    sooner_missed = _appointment(db, clinic, patient, doctor, sooner_missed_slot, status="no_show")
    later_missed = _appointment(db, clinic, patient, doctor, later_missed_slot, status="no_show")
    db.flush()

    # later_missed's slot was further in the past, but was marked no_show MOST
    # RECENTLY (right now) — sooner_missed was marked no_show earlier.
    sooner_missed.updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
    later_missed.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.flush()

    result = _get_my_appointments_impl(db, ctx, "missed", 10)

    parsed = json.loads(result)
    assert [a["appointment_id"] for a in parsed["appointments"]] == [
        str(later_missed.id),
        str(sooner_missed.id),
    ]
    assert all(a["status"] == "no_show" for a in parsed["appointments"])


# --- get_department_availability: minimum slot guarantee + never a raw slot_id -----


def test_at_least_four_slots_surfaced_and_slot_id_never_appears_as_prose(db, clinic, department, doctor, patient, ctx):
    base = datetime.now(timezone.utc) + timedelta(days=1)
    for i in range(6):
        _slot(db, clinic, doctor, base + timedelta(hours=i))

    result = _get_department_availability_impl(db, ctx, "Cardiology")

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    slots = payload["doctors"][0]["slots"]
    assert len(slots) >= 4
    assert MAX_SLOTS_PER_DOCTOR >= 4
    # Every slot_id lives only in its own JSON field, never mixed into readable prose
    # text — the button label a patient actually sees is `when`, not `slot_id`.
    for slot in slots:
        assert slot["slot_id"] not in slot["when"]


# --- get_department_availability: earliest_date threading (item 4) -----------------


def test_get_department_availability_earliest_date_reaches_the_query(db, clinic, department, doctor, patient, ctx):
    now = datetime.now(timezone.utc)
    soon_slot = _slot(db, clinic, doctor, now + timedelta(days=1))
    friday = (now + timedelta(days=5)).date()
    friday_slot = _slot(db, clinic, doctor, datetime.combine(friday, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=10))

    result = _get_department_availability_impl(db, ctx, "Cardiology", None, friday.isoformat())

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    slot_ids = {s["slot_id"] for doc in payload["doctors"] for s in doc["slots"]}
    assert str(friday_slot.id) in slot_ids
    assert str(soon_slot.id) not in slot_ids


def test_get_department_availability_malformed_earliest_date_is_ignored_not_an_error(db, clinic, department, doctor, patient, ctx):
    _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    result = _get_department_availability_impl(db, ctx, "Cardiology", None, "not-a-date")

    # Falls back to no date filter rather than erroring the whole turn over a
    # malformed argument the model produced.
    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    assert len(payload["doctors"][0]["slots"]) == 1


# --- resolve_bare_weekday_window / build_tools forced_date_window -------------------
# Reported live: "book a slot on sun" silently showed Friday's slots instead of
# "nothing open Sunday, earliest is Friday" — the model never set earliest_date/
# latest_date for the bare weekday. This resolves the weekday deterministically in
# code and forces it onto the tool call regardless of what the model passes.


def test_resolve_bare_weekday_window_resolves_full_weekday_name_to_next_occurrence():
    today = datetime.now(timezone.utc).date()
    # Today's own weekday name should resolve to today, not seven days out.
    todays_name = today.strftime("%A")
    assert resolve_bare_weekday_window(f"is there anything available on {todays_name}?") == (
        today.isoformat(),
        today.isoformat(),
    )
    # Three days out should resolve to that future date, not today.
    future = today + timedelta(days=3)
    future_name = future.strftime("%A")
    assert resolve_bare_weekday_window(f"is there anything available on {future_name}?") == (
        future.isoformat(),
        future.isoformat(),
    )


def test_resolve_bare_weekday_window_resolves_abbreviation_after_a_preposition():
    window = resolve_bare_weekday_window("book a slot on sun")
    assert window is not None
    earliest, latest = window
    resolved = datetime.fromisoformat(earliest).date()
    assert resolved.strftime("%A") == "Sunday"
    assert earliest == latest


def test_resolve_bare_weekday_window_none_when_no_weekday_named():
    assert resolve_bare_weekday_window("book a slot for cardiology") is None


def test_resolve_bare_weekday_window_none_for_a_bare_abbreviation_not_after_a_preposition():
    # "sat"/"sun" collide with ordinary English words ("I sat down", "the sun is
    # out") - only trigger when preceded by a date-shaped preposition.
    assert resolve_bare_weekday_window("I sat in the waiting room") is None


def test_resolve_bare_weekday_window_none_when_more_than_one_weekday_named():
    assert resolve_bare_weekday_window("anything monday or tuesday?") is None


def test_resolve_bare_weekday_window_tolerates_a_single_character_typo():
    window = resolve_bare_weekday_window("show me her available slots on thus")
    assert window is not None
    earliest, latest = window
    resolved = datetime.fromisoformat(earliest).date()
    assert resolved.strftime("%A") == "Thursday"
    assert earliest == latest


def test_resolve_bare_weekday_window_none_when_typo_is_too_far_off():
    assert resolve_bare_weekday_window("book a slot for cardiology") is None


@pytest.mark.parametrize("message", ["sat 1045", "sat 10.45", "sat 10:45", "on sat 1045"])
def test_resolve_bare_weekday_window_resolves_an_abbreviation_immediately_before_a_time(message):
    # Reported live: a fragmented reply to a "which doctor" clarifying question
    # ("sat 1045", no "on"/"for"/etc. word at all before the abbreviation)
    # matched neither the full-weekday regex nor the trigger-word-gated
    # abbreviation regex, so the weekday was silently dropped. An abbreviation
    # immediately followed by a clock-shaped token is unambiguous enough to
    # trust on its own, unlike a bare "sat"/"sun" with nothing after it (see
    # the "I sat in the waiting room" test above, which must still stay None).
    window = resolve_bare_weekday_window(message)
    assert window is not None
    earliest, latest = window
    resolved = datetime.fromisoformat(earliest).date()
    assert resolved.strftime("%A") == "Saturday"
    assert earliest == latest


def test_resolve_bare_weekday_window_still_none_for_a_word_with_unrelated_nearby_digits():
    # A weekday-abbreviation-shaped word immediately followed by digits is
    # trusted (see the test above) — but this must not regress the existing
    # "sat"/"sun" collision guard for messages with NO digits nearby at all.
    assert resolve_bare_weekday_window("I sat in the waiting room for 2 hours") is None


# --- resolve_date_window (explicit calendar date takes priority over a bare weekday) --


def test_resolve_date_window_prefers_the_explicit_date_when_a_weekday_name_is_also_present():
    # Reported live: "mon aug 31st" got resolved to the nearest upcoming Monday
    # from TODAY, silently ignoring "aug 31" — resolve_bare_weekday_window has no
    # awareness of a following month/day at all, so it matched "mon" on its own.
    # resolve_date_window must always resolve the real named date instead.
    from datetime import date, timedelta

    today = date.today()
    future_year = today.year if date(today.year, 8, 31) >= today else today.year + 1
    expected = date(future_year, 8, 31).isoformat()

    assert resolve_date_window("mon aug 31st") == (expected, expected)
    assert resolve_date_window("mon aug 31") == (expected, expected)
    assert resolve_date_window("on mon aug 31") == (expected, expected)
    assert resolve_date_window("monday aug 31") == (expected, expected)
    assert resolve_date_window("i want to see the doctor on mon aug 31st") == (expected, expected)


def test_resolve_date_window_falls_back_to_bare_weekday_when_no_explicit_date_named():
    assert resolve_date_window("show me his slots on mon") == resolve_bare_weekday_window("show me his slots on mon")


def test_resolve_date_window_none_when_neither_a_weekday_nor_an_explicit_date_is_named():
    assert resolve_date_window("book a slot for cardiology") is None


# --- ensure_slot_pick_example_has_a_date (moved out of the system prompt to cut tokens) --


def test_ensure_slot_pick_example_has_a_date_fixes_a_bare_time_example():
    # Reported live: a reply correctly listed real slots each with a full date
    # but then closed with a bare-time-only example.
    reply = (
        "Here are the available slots for Dr. Ali Raza (General Medicine):\n\n"
        "- Mon, Aug 24 at 9:00 AM\n"
        "- Mon, Aug 24 at 9:30 AM\n\n"
        'Please let me know which time you\'d like to book by replying with the exact slot (e.g., "9:30 AM").'
    )

    result = ensure_slot_pick_example_has_a_date(reply)

    assert '(e.g., "Aug 24 at 9:30 AM")' in result


def test_ensure_slot_pick_example_has_a_date_leaves_an_already_dated_example_untouched():
    reply = 'Please reply with the exact slot (e.g., "Aug 24 at 9:30 AM").'

    assert ensure_slot_pick_example_has_a_date(reply) == reply


def test_ensure_slot_pick_example_has_a_date_is_a_noop_with_no_slot_list_to_borrow_a_date_from():
    reply = 'Please reply with the exact slot (e.g., "9:30 AM").'

    assert ensure_slot_pick_example_has_a_date(reply) == reply


def test_ensure_slot_pick_example_has_a_date_is_a_noop_with_no_example_at_all():
    reply = "Here are the available slots:\n\n- Mon, Aug 24 at 9:00 AM\n\nWhich one would you like?"

    assert ensure_slot_pick_example_has_a_date(reply) == reply


def test_build_tools_forces_the_resolved_weekday_window_regardless_of_model_args(
    db, clinic, department, doctor, patient, ctx
):
    now = datetime.now(timezone.utc)
    # Find the next Sunday and put a slot there, plus a nearer slot on a different day.
    days_until_sunday = (6 - now.weekday()) % 7 or 7
    sunday = (now + timedelta(days=days_until_sunday)).date()
    sunday_slot = _slot(db, clinic, doctor, datetime.combine(sunday, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=9))
    nearer_slot = _slot(db, clinic, doctor, now + timedelta(days=1))

    forced_window = resolve_bare_weekday_window("book a slot on sun")
    tools = build_tools(db, ctx, forced_date_window=forced_window)
    availability_tool = next(t for t in tools if t.name == "get_department_availability")

    # The model omits earliest_date/latest_date entirely, same as the live report.
    result = availability_tool.invoke({"department_name": "Cardiology"})

    payload = _parse_marker(result, DOCTOR_OPTIONS_MARKER)
    slot_ids = {s["slot_id"] for doc in payload["doctors"] for s in doc["slots"]}
    assert str(sunday_slot.id) in slot_ids
    assert str(nearer_slot.id) not in slot_ids


# --- combine_department_availability_results (item 4: multi-department combiner) ---


def test_combine_preserves_every_department_and_all_their_doctors():
    # Two synthetic DOCTOR_OPTIONS:: results, simulating two real
    # get_department_availability calls made within one agent turn.
    result_a = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Cardiology", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    result_b = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Dermatology", "doctors": [{"doctor_id": "d2", "doctor_name": "Dr. B", "specialization": None, "slots": []}]}
    )

    combined = combine_department_availability_results([result_a, result_b])

    assert combined.startswith(DEPARTMENT_LIST_MARKER)
    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    assert [d["department_name"] for d in payload["departments"]] == ["Cardiology", "Dermatology"]
    assert payload["departments"][0]["doctors"][0]["doctor_name"] == "Dr. A"
    assert payload["departments"][1]["doctors"][0]["doctor_name"] == "Dr. B"


def test_combine_preserves_each_departments_own_note():
    # Reported live: a symptom-based multi-department reply (nose pain + skin
    # issues -> ENT + Dermatology) showed both departments' doctors but explained
    # the reasoning for neither — the note each call carried was silently dropped
    # when merged into a DEPARTMENT_LIST_MARKER, even though the single-department
    # card has always shown its note.
    result_a = DOCTOR_OPTIONS_MARKER + json.dumps(
        {
            "note": "The nose pain points to ENT.",
            "department_name": "ENT",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}],
        }
    )
    result_b = DOCTOR_OPTIONS_MARKER + json.dumps(
        {
            "note": "The skin issue points to Dermatology.",
            "department_name": "Dermatology",
            "doctors": [{"doctor_id": "d2", "doctor_name": "Dr. B", "specialization": None, "slots": []}],
        }
    )

    combined = combine_department_availability_results([result_a, result_b])

    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    notes = {d["department_name"]: d["note"] for d in payload["departments"]}
    assert notes == {"ENT": "The nose pain points to ENT.", "Dermatology": "The skin issue points to Dermatology."}


def test_combine_omits_note_when_the_department_was_named_directly_not_inferred():
    # A direct request ("book me with a cardiologist") never gets a note in the
    # first place (see the TOOL USE RULES' "Omit note entirely" rule) — this must
    # not resurface reasoning for a case that genuinely never had any.
    result_a = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Cardiology", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    result_b = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Dermatology", "doctors": [{"doctor_id": "d2", "doctor_name": "Dr. B", "specialization": None, "slots": []}]}
    )

    combined = combine_department_availability_results([result_a, result_b])

    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    assert all(d["note"] is None for d in payload["departments"])


def test_combine_deduplicates_the_same_department_called_more_than_once():
    # Reported bug: the model called get_department_availability twice for the same
    # department in one turn (confused by a typo'd day name) and the same
    # department/doctors/slots rendered twice, back-to-back, in the resulting card.
    result_a = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Cardiology", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    result_a_again = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Cardiology", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )

    combined = combine_department_availability_results([result_a, result_a_again])

    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    assert len(payload["departments"]) == 1
    assert payload["departments"][0]["department_name"] == "Cardiology"


def test_combine_skips_a_not_found_department_but_keeps_a_real_card():
    real_card = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Cardiology", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    not_found_text = "I couldn't find a department called that. The departments we have are: Cardiology."

    combined = combine_department_availability_results([not_found_text, real_card])

    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    assert len(payload["departments"]) == 1
    assert payload["departments"][0]["department_name"] == "Cardiology"
    assert payload["unavailable"] == []


def test_combine_surfaces_a_real_department_with_no_free_slots_instead_of_dropping_it():
    # Reported live: "neck pain and itchy skin" resolved Orthopedics (real card) and
    # Dermatology (no doctor free) — the reply's own note named both departments, but
    # the card only ever showed Orthopedics, with no mention of Dermatology at all.
    real_card = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Orthopedics", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    no_slots = NO_SLOTS_MARKER + json.dumps(
        {"department_name": "Dermatology", "message": "I couldn't find any doctors with free upcoming slots in Dermatology right now. Please check back later or contact the clinic directly."}
    )

    combined = combine_department_availability_results([real_card, no_slots])

    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    assert [d["department_name"] for d in payload["departments"]] == ["Orthopedics"]
    assert len(payload["unavailable"]) == 1
    assert payload["unavailable"][0]["department_name"] == "Dermatology"
    assert "dermatology" in payload["unavailable"][0]["message"].lower()


def test_combine_with_only_no_slots_results_returns_their_messages_not_a_card():
    no_slots_a = NO_SLOTS_MARKER + json.dumps({"department_name": "Cardiology", "message": "Nothing free in Cardiology."})
    no_slots_b = NO_SLOTS_MARKER + json.dumps({"department_name": "Dermatology", "message": "Nothing free in Dermatology."})

    combined = combine_department_availability_results([no_slots_a, no_slots_b])

    assert not combined.startswith(DEPARTMENT_LIST_MARKER)
    assert "Nothing free in Cardiology." in combined
    assert "Nothing free in Dermatology." in combined


def test_combine_with_no_real_results_returns_plain_text_not_a_card():
    combined = combine_department_availability_results(["nothing found here", "nor here"])

    assert not combined.startswith(DEPARTMENT_LIST_MARKER)
    assert "couldn't find" in combined.lower()


# --- find_doctors_by_name -------------------------------------------------------------


def test_find_doctors_by_name_impl_returns_structured_matches(db, clinic, department, doctor, ctx):
    # Reproduces the reported bug: "raza iqra" (reversed) must still resolve to a real
    # "Dr. Iqra Raza" via word-level matching, returned as structured data (not a
    # verbatim-reply card) for the model to phrase a confirming question from.
    doctor.full_name = "Dr. Iqra Raza"
    db.flush()

    raw = _find_doctors_by_name_impl(db, ctx, "raza iqra")
    payload = json.loads(raw)

    assert payload["matches"] == [{"doctor_name": "Dr. Iqra Raza", "department_name": "Cardiology"}]


def test_find_doctors_by_name_impl_no_match_returns_empty_list(db, clinic, department, doctor, ctx):
    raw = _find_doctors_by_name_impl(db, ctx, "Someone Else")
    payload = json.loads(raw)

    assert payload["matches"] == []


def test_build_tools_registers_find_doctors_by_name(db, ctx):
    tools = build_tools(db, ctx)
    names = {t.name for t in tools}

    assert "find_doctors_by_name" in names
    assert len(tools) == 6


_BOOKING_ACTION_TOOL_NAMES = {"book_appointment", "reschedule_appointment", "cancel_appointment"}
_INFO_TOOL_NAMES = {"get_my_appointments", "get_department_availability", "find_doctors_by_name"}


def test_build_tools_omits_booking_action_tools_when_told_to(db, ctx):
    tools = build_tools(db, ctx, include_booking_action_tools=False)
    names = {t.name for t in tools}

    assert names == _INFO_TOOL_NAMES
    assert len(tools) == 3


def test_build_tools_includes_booking_action_tools_by_default(db, ctx):
    tools = build_tools(db, ctx)
    names = {t.name for t in tools}

    assert names == _INFO_TOOL_NAMES | _BOOKING_ACTION_TOOL_NAMES


def test_build_tools_includes_booking_action_tools_when_explicitly_true(db, ctx):
    tools = build_tools(db, ctx, include_booking_action_tools=True)
    names = {t.name for t in tools}

    assert names == _INFO_TOOL_NAMES | _BOOKING_ACTION_TOOL_NAMES


# --- resolve_time_of_day_window -----------------------------------------------------
# Reported live: "on fri after 7.20 pm" returned the day's earliest slots (6:00pm
# onward) instead of nothing-before-7:20pm — the "after Xpm" regex only accepted a
# colon between hour and minute, so a dot-separated time like "7.20 pm" silently
# failed to match at all and no time-of-day constraint was ever applied.


def test_resolve_time_of_day_window_accepts_colon_separated_after_time():
    assert resolve_time_of_day_window("after 7:20 pm") == (time(19, 20), None)


def test_resolve_time_of_day_window_accepts_dot_separated_after_time():
    assert resolve_time_of_day_window("on fri after 7.20 pm") == (time(19, 20), None)


def test_resolve_time_of_day_window_accepts_dot_separated_before_time():
    assert resolve_time_of_day_window("before 9.30am") == (None, time(9, 30))


def test_resolve_time_of_day_window_accepts_bare_hour_after_time():
    assert resolve_time_of_day_window("after 12 pm") == (time(12, 0), None)


def test_resolve_time_of_day_window_falls_back_to_time_of_day_phrase():
    assert resolve_time_of_day_window("anything in the evening?") == (time(17, 0), time(21, 0))


def test_resolve_time_of_day_window_none_when_no_time_named():
    assert resolve_time_of_day_window("book a slot for cardiology") is None


# Reported live: "book me with dr waqas on mon aug 24 at 3.30 instead" and its
# follow-up "at 3.30" were both silently ignored — only "after"/"before" were
# recognized, so a bare "at X" request fell through with no time bound at all.


def test_resolve_time_of_day_window_accepts_at_time_with_explicit_meridiem():
    assert resolve_time_of_day_window("at 3:30 pm") == (time(15, 30), None)


def test_resolve_time_of_day_window_accepts_at_time_with_dot_separator():
    assert resolve_time_of_day_window("book me with dr waqas on mon aug 24 at 3.30 instead") == (
        time(15, 30),
        None,
    )


def test_resolve_time_of_day_window_infers_pm_for_a_bare_afternoon_hour():
    assert resolve_time_of_day_window("at 3.30") == (time(15, 30), None)


def test_resolve_time_of_day_window_infers_am_for_a_bare_morning_hour():
    assert resolve_time_of_day_window("at 9.15") == (time(9, 15), None)


def test_resolve_time_of_day_window_prefers_after_before_over_at():
    assert resolve_time_of_day_window("after 12 pm") == (time(12, 0), None)


# Reported live: "sat 1045", "sat 10.45", "sat 10:45" (a fragmented reply, no
# "at"/"after"/"before" word at all) were all silently ignored — every clock
# pattern above requires one of those trigger words first.


def test_resolve_time_of_day_window_accepts_a_bare_colon_separated_time():
    assert resolve_time_of_day_window("sat 10:45") == (time(10, 45), None)


def test_resolve_time_of_day_window_accepts_a_bare_dot_separated_time():
    assert resolve_time_of_day_window("sat 10.45") == (time(10, 45), None)


def test_resolve_time_of_day_window_accepts_a_bare_no_separator_time():
    assert resolve_time_of_day_window("sat 1045") == (time(10, 45), None)


def test_resolve_time_of_day_window_accepts_a_bare_three_digit_no_separator_time():
    assert resolve_time_of_day_window("945") == (time(9, 45), None)


def test_resolve_time_of_day_window_rejects_an_implausible_bare_digit_time():
    # "2024" (e.g. a year) splits as hour=20/minute=24 — not a plausible
    # 12-hour clock time, so this must NOT be misread as a time at all.
    assert resolve_time_of_day_window("in 2024 I had surgery") is None


# Reported live: "mon 12 pm" (an answer to a "what day would you like?"
# question) and "book at mon 12 pm" (the "at" here modifies "mon", not "12
# pm" — _AT_CLOCK_RE requires "at" immediately before the number) both went
# completely unrecognized as a time at all — no colon/dot, and only 2 digits
# (too few for the bare-digit-run case), but with an explicit am/pm marker.


def test_resolve_time_of_day_window_accepts_a_bare_hour_with_explicit_meridiem():
    assert resolve_time_of_day_window("12 pm") == (time(12, 0), None)
    assert resolve_time_of_day_window("3 pm") == (time(15, 0), None)


def test_resolve_time_of_day_window_accepts_a_bare_hour_with_meridiem_alongside_a_day():
    assert resolve_time_of_day_window("mon 12 pm") == (time(12, 0), None)
    assert resolve_time_of_day_window("book at mon 12 pm") == (time(12, 0), None)


def test_resolve_time_of_day_window_prefers_the_longer_digit_run_over_bare_hour_meridiem():
    # "1045 pm" must still resolve as 10:45 PM via the digit-clock case, not
    # get misread as just "45 pm" by the bare-hour-with-meridiem fallback.
    assert resolve_time_of_day_window("1045 pm") == (time(22, 45), None)
