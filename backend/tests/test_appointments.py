import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from app.models.appointment import Appointment
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.services import booking_engine
from app.services.appointments import get_pending_visit_confirmations


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _appointment(db, clinic, start_utc, end_utc, status="confirmed"):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()

    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
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

    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=start_utc, end_utc=end_utc)
    db.add(slot)
    db.flush()

    appointment = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status=status
    )
    db.add(appointment)
    db.flush()
    return appointment


def test_appointment_not_pending_mid_slot_after_start_but_before_end(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=15)  # 17:00 if now is 17:15
    end = now + timedelta(minutes=15)  # 17:30
    appointment = _appointment(db, clinic, start, end)

    pending = get_pending_visit_confirmations(db, clinic.id, appointment.patient_id)

    assert pending == []


def test_appointment_is_pending_once_slot_end_has_passed(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=45)
    end = now - timedelta(minutes=15)  # already ended
    appointment = _appointment(db, clinic, start, end)

    pending = get_pending_visit_confirmations(db, clinic.id, appointment.patient_id)

    assert [a.id for a in pending] == [appointment.id]
    assert appointment.status == "confirmed"  # still confirmed — nothing auto-flips it


def test_cancelled_appointment_never_shows_up_as_pending(db, clinic):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=2)
    end = now - timedelta(hours=1)
    appointment = _appointment(db, clinic, start, end, status="cancelled")

    pending = get_pending_visit_confirmations(db, clinic.id, appointment.patient_id)

    assert pending == []


def test_confirm_visit_completed_marks_appointment_completed(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(hours=2), now - timedelta(hours=1))

    result = booking_engine.confirm_visit(
        db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=True
    )

    assert result.status == "completed"


def test_confirm_visit_missed_marks_appointment_no_show(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(hours=2), now - timedelta(hours=1))

    result = booking_engine.confirm_visit(
        db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=False
    )

    assert result.status == "no_show"


def test_confirm_visit_rejects_appointment_whose_slot_has_not_ended(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(minutes=5), now + timedelta(minutes=25))

    with pytest.raises(HTTPException) as exc_info:
        booking_engine.confirm_visit(
            db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=True
        )

    assert exc_info.value.status_code == 409
    assert appointment.status == "confirmed"


def test_confirm_visit_rejects_appointment_not_in_confirmed_status(db, clinic):
    now = datetime.now(timezone.utc)
    appointment = _appointment(db, clinic, now - timedelta(hours=2), now - timedelta(hours=1), status="cancelled")

    with pytest.raises(HTTPException) as exc_info:
        booking_engine.confirm_visit(
            db, clinic_id=clinic.id, patient_id=appointment.patient_id, appointment_id=appointment.id, completed=True
        )


def test_reschedule_enforces_daily_department_reschedule_cap(db, clinic):
    """DAILY_DEPARTMENT_RESCHEDULE_CAP (2) is keyed by the OLD appointment's own
    department/day (the one being moved AWAY from) and is independent of the
    booking cap. Three separate appointments all originally in the same
    department on the same day — rescheduling the first two of them (each to a
    DIFFERENT department, so the booking cap on the target day/department never
    itself becomes the blocker) is allowed; rescheduling the third, from that
    same original department/day, is refused once the cap is already used up."""
    dept1 = Department(clinic_id=clinic.id, name="Cardiology")
    dept2 = Department(clinic_id=clinic.id, name="General Medicine")
    dept3 = Department(clinic_id=clinic.id, name="Dermatology")
    db.add_all([dept1, dept2, dept3])
    db.flush()

    doctor1 = Doctor(
        clinic_id=clinic.id, department_id=dept1.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Cardio One", is_active=True,
    )
    doctor2 = Doctor(
        clinic_id=clinic.id, department_id=dept2.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. General One", is_active=True,
    )
    doctor3 = Doctor(
        clinic_id=clinic.id, department_id=dept3.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Derma One", is_active=True,
    )
    db.add_all([doctor1, doctor2, doctor3])
    db.flush()

    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()

    def _slot(doctor, hour):
        s = Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime(2030, 6, 10, hour, 0, tzinfo=timezone.utc),
            end_utc=datetime(2030, 6, 10, hour, 30, tzinfo=timezone.utc),
        )
        db.add(s)
        db.flush()
        return s

    def _booked_appointment(slot):
        appt = Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=slot.doctor_id, status="confirmed")
        db.add(appt)
        db.flush()
        return appt

    # Three appointments, all originally in dept1 on 2030-06-10 (inserted directly,
    # bypassing book_appointment's own daily cap — simulating three appointments
    # that already exist, isolating the reschedule cap under test from the
    # separate booking cap).
    appt_a = _booked_appointment(_slot(doctor1, 8))
    appt_b = _booked_appointment(_slot(doctor1, 10))
    appt_c = _booked_appointment(_slot(doctor1, 12))

    # Reschedule #1 out of (Cardiology, 2030-06-10), into General Medicine: allowed.
    booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appt_a.id, new_slot_id=_slot(doctor2, 14).id
    )
    # Reschedule #2 out of (Cardiology, 2030-06-10), into Dermatology this time —
    # a DIFFERENT target department, so the target-side booking cap (also 2, but
    # per target department/day) never itself becomes the blocker here: allowed.
    booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appt_b.id, new_slot_id=_slot(doctor3, 16).id
    )
    # Reschedule #3 out of that same (Cardiology, 2030-06-10), into yet another
    # fresh target (General Medicine again, but only its SECOND use — still well
    # under ITS cap): the cap (2) on the ORIGIN (Cardiology, 2030-06-10) is what
    # blocks this one.
    target_slot = _slot(doctor2, 18)
    with pytest.raises(HTTPException) as exc_info:
        booking_engine.reschedule_appointment(
            db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appt_c.id, new_slot_id=target_slot.id
        )

    assert exc_info.value.status_code == 409
    assert "reschedules" in exc_info.value.detail


def test_reschedule_to_a_different_time_same_day_same_department_is_not_blocked_by_booking_cap(db, clinic):
    """Reported live: with the daily booking cap (2) already fully used by 2 real
    bookings in a department/day, rescheduling ONE of them to a later time the
    SAME day in the SAME department was wrongly refused with "you already have 2
    appointments" — moving an appointment's time within its own department/day
    isn't a NEW use of that day's booking cap, it's the same appointment that was
    already counted. Only the (separate) reschedule cap should govern this case."""
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=dept.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name="Dr. Cardio One", is_active=True,
    )
    db.add(doctor)
    db.flush()
    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()

    def _slot(hour):
        s = Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime(2030, 6, 10, hour, 0, tzinfo=timezone.utc),
            end_utc=datetime(2030, 6, 10, hour, 30, tzinfo=timezone.utc),
        )
        db.add(s)
        db.flush()
        return s

    # Two real bookings in (Cardiology, 2030-06-10) — the daily booking cap (2)
    # for that department/day is now fully used, same as booking through the
    # real book_appointment() would leave it.
    appt_a = Appointment(clinic_id=clinic.id, slot_id=_slot(9).id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed")
    db.add(appt_a)
    appt_b = Appointment(clinic_id=clinic.id, slot_id=_slot(11).id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed")
    db.add(appt_b)
    db.flush()
    from app.models.appointment_department_day_use import AppointmentDepartmentDayUse
    db.add(AppointmentDepartmentDayUse(clinic_id=clinic.id, patient_id=patient.id, department_id=dept.id, appointment_id=appt_a.id, local_date=date(2030, 6, 10)))
    db.add(AppointmentDepartmentDayUse(clinic_id=clinic.id, patient_id=patient.id, department_id=dept.id, appointment_id=appt_b.id, local_date=date(2030, 6, 10)))
    db.flush()

    # Move appt_a to a later time, SAME day, SAME department — must succeed
    # despite the booking cap already sitting at 2 for that department/day.
    new_time_slot = _slot(15)
    result = booking_engine.reschedule_appointment(
        db, clinic_id=clinic.id, patient_id=patient.id, appointment_id=appt_a.id, new_slot_id=new_time_slot.id
    )

    assert result.slot_id == new_time_slot.id
