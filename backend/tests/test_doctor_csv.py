import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services.doctor_csv import validate_csv


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _csv(header_row: str, data_row: str = "DOC-1,Dr. Jane Example,Cardiology,Cardiologist,Mon,09:00,17:00,true") -> bytes:
    return f"{header_row}\r\n{data_row}\r\n".encode("utf-8")


# --- Exact-match headers: no mapping step, straight to validation -----------------


def test_exact_match_headers_skip_mapping(db, clinic):
    content = _csv("external_doctor_id,name,department,specialty,shift_days,shift_start,shift_end,active")
    result = validate_csv(content, db, clinic.id, clinic.timezone, 30)

    assert result.needs_header_mapping is False
    assert result.header_suggestions == []
    assert result.unrecognized_headers == []
    assert result.is_structurally_valid
    assert len(result.accepted) == 1
    assert result.accepted[0].department_name == "Cardiology"


# --- Near-synonym header: suggested, not auto-applied ------------------------------


def test_synonym_header_is_suggested_not_auto_applied(db, clinic):
    content = _csv("external_doctor_id,name,Dept,specialty,shift_days,shift_start,shift_end,active")
    result = validate_csv(content, db, clinic.id, clinic.timezone, 30)

    assert result.needs_header_mapping is True
    assert result.accepted == []  # nothing validated yet — mapping isn't applied until confirmed
    assert result.rejected == []
    headers = {s.original_header: s.suggested_field for s in result.header_suggestions}
    assert "Dept" in headers
    assert headers["Dept"] == "department"
    assert all(0.0 <= s.confidence <= 1.0 for s in result.header_suggestions)


def test_synonym_header_applied_only_after_explicit_confirmation(db, clinic):
    content = _csv("external_doctor_id,name,Dept,specialty,shift_days,shift_start,shift_end,active")

    # Confirming the mapping explicitly is what actually applies it.
    result = validate_csv(
        content, db, clinic.id, clinic.timezone, 30, confirmed_header_mapping={"Dept": "department"}
    )

    assert result.needs_header_mapping is False
    assert result.is_structurally_valid
    assert len(result.accepted) == 1
    assert result.accepted[0].department_name == "Cardiology"


# --- Nonsense header: rejected as unrecognized, never guessed ----------------------


def test_unrelated_header_is_rejected_as_unrecognized(db, clinic):
    content = _csv(
        "external_doctor_id,name,Zzyzx Blorf Nonsense,specialty,shift_days,shift_start,shift_end,active"
    )
    result = validate_csv(content, db, clinic.id, clinic.timezone, 30)

    assert result.needs_header_mapping is True
    assert result.accepted == []
    assert "Zzyzx Blorf Nonsense" in result.unrecognized_headers
    assert not any(s.original_header == "Zzyzx Blorf Nonsense" for s in result.header_suggestions)


def test_unresolved_header_blocks_validation_even_if_other_headers_confirmed(db, clinic):
    # "Dept" -> department is confirmed, but the nonsense header is still unresolved:
    # the whole file must stay blocked, not just partially validated.
    content = _csv(
        "external_doctor_id,name,Dept,Zzyzx Blorf Nonsense,shift_days,shift_start,shift_end,active",
        "DOC-1,Dr. Jane Example,Cardiology,xyz,Mon,09:00,17:00,true",
    )
    result = validate_csv(
        content, db, clinic.id, clinic.timezone, 30, confirmed_header_mapping={"Dept": "department"}
    )
    assert result.needs_header_mapping is True
    assert result.accepted == []


# --- Conflict-blocking (Day 4/5) must be unaffected --------------------------------


def test_conflict_blocking_still_rejects_row_that_orphans_confirmed_appointment(db, clinic):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()

    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id="DOC-1",
        full_name="Dr. Jane Example", is_active=True,
    )
    db.add(doctor)
    db.flush()

    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()

    # A confirmed appointment on a Tuesday slot, inside the regeneration horizon.
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    db.add(Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed"))
    db.flush()

    # New CSV drops the doctor's shift entirely (e.g. flips to a day that never covers `start`),
    # which would orphan the confirmed appointment above.
    content = _csv(
        "external_doctor_id,name,department,specialty,shift_days,shift_start,shift_end,active",
        "DOC-1,Dr. Jane Example,Cardiology,Cardiologist,Sun,09:00,10:00,true",
    )
    result = validate_csv(content, db, clinic.id, clinic.timezone, 30)

    assert result.needs_header_mapping is False
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert "confirmed appointment" in result.rejected[0].reason


def test_conflict_blocking_still_works_through_header_mapping(db, clinic):
    """Same conflict-blocking scenario, but the CSV uses a synonym header — confirms the
    mapping step runs strictly before validation and doesn't bypass conflict checks.
    """
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id="DOC-1",
        full_name="Dr. Jane Example", is_active=True,
    )
    db.add(doctor)
    db.flush()
    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start, end_utc=start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    db.add(Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed"))
    db.flush()

    content = _csv(
        "external_doctor_id,name,Dept,specialty,shift_days,shift_start,shift_end,active",
        "DOC-1,Dr. Jane Example,Cardiology,Cardiologist,Sun,09:00,10:00,true",
    )
    result = validate_csv(
        content, db, clinic.id, clinic.timezone, 30, confirmed_header_mapping={"Dept": "department"}
    )
    assert result.needs_header_mapping is False
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert "confirmed appointment" in result.rejected[0].reason
