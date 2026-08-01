import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.core.tenancy import ClinicContext
from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services.chat_markers import BOOKING_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER
from app.services.chat_tools import (
    _book_appointment_impl,
    _cancel_appointment_impl,
    _find_doctors_by_name_impl,
    _get_department_availability_impl,
    _get_my_appointments_impl,
    _reschedule_appointment_impl,
    build_tools,
    combine_department_availability_results,
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


def test_cancel_appointment_same_day_refusal_matches_rest_endpoint_message(db, clinic, doctor, patient, ctx):
    today_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(hours=1))
    appointment = _appointment(db, clinic, patient, doctor, today_slot)

    result = _cancel_appointment_impl(db, ctx, str(appointment.id))

    assert result == (
        "Appointments cannot be cancelled on the same day as the appointment. "
        "Please contact the clinic directly for same-day changes."
    )


def test_cancel_appointment_success_gives_plain_confirmation_sentence(db, clinic, doctor, patient, ctx):
    future_slot = _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=2))
    appointment = _appointment(db, clinic, patient, doctor, future_slot)

    result = _cancel_appointment_impl(db, ctx, str(appointment.id))

    assert "Dr. Jane Example" in result
    assert "Cardiology" in result
    assert "cancelled" in result.lower()


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
    assert "couldn't find" in result.lower()


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


def test_combine_skips_non_card_results_like_not_found_or_no_slots():
    real_card = DOCTOR_OPTIONS_MARKER + json.dumps(
        {"note": None, "department_name": "Cardiology", "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    not_found_text = "I couldn't find a department called that. The departments we have are: Cardiology."

    combined = combine_department_availability_results([not_found_text, real_card])

    payload = json.loads(combined[len(DEPARTMENT_LIST_MARKER):])
    assert len(payload["departments"]) == 1
    assert payload["departments"][0]["department_name"] == "Cardiology"


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
