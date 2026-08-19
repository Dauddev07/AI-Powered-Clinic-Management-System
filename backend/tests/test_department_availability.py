import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.services.department_availability import (
    MAX_SLOTS_PER_DOCTOR,
    find_doctors_by_name,
    get_department_availability,
    list_active_department_names,
)


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _department(db, clinic, name="Cardiology", is_active=True):
    dept = Department(clinic_id=clinic.id, name=name, is_active=is_active)
    db.add(dept)
    db.flush()
    return dept


def _doctor(db, clinic, department, full_name="Dr. Jane Example", is_active=True):
    doctor = Doctor(
        clinic_id=clinic.id,
        department_id=department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name=full_name,
        is_active=is_active,
    )
    db.add(doctor)
    db.flush()
    return doctor


def _slot(db, clinic, doctor, start_utc, status="open"):
    slot = Slot(
        clinic_id=clinic.id,
        doctor_id=doctor.id,
        start_utc=start_utc,
        end_utc=start_utc + timedelta(minutes=30),
        status=status,
    )
    db.add(slot)
    db.flush()
    return slot


# --- exact match / case-insensitivity ------------------------------------------------


def test_exact_case_insensitive_match_finds_department_and_free_slots(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    _slot(db, clinic, doctor, future)

    result = get_department_availability(db, clinic.id, "cardiology")

    assert result.found is True
    assert result.department_name == "Cardiology"
    assert len(result.doctors) == 1
    assert result.doctors[0].full_name == "Dr. Jane Example"
    assert len(result.doctors[0].slots) == 1


def test_match_is_exact_not_fuzzy_or_substring(db, clinic):
    _department(db, clinic, name="Cardiology")

    # "Cardio" must NOT fuzzy/substring-match "Cardiology" — the department-name
    # lookup is deliberately exact-only so a typo/nonexistent department is reported
    # as not found rather than silently routed to the wrong one.
    result = get_department_availability(db, clinic.id, "Cardio")

    assert result.found is False


# --- not-found case -------------------------------------------------------------------


def test_professional_title_falls_back_to_the_real_department_it_maps_to(db, clinic):
    # Reported live: "available slots for cardiologist" reached this tool with
    # department_name="cardiologist" and failed with not-found, even though the
    # exact same title is already recognized by DEPARTMENT_TITLE_HINTS elsewhere
    # in the app (appointment_agent's deterministic checks). This is a curated
    # synonym table, not fuzzy/substring guessing, so it doesn't weaken the exact-
    # match-only guarantee test_match_is_exact_not_fuzzy_or_substring covers above.
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    _slot(db, clinic, doctor, future)

    result = get_department_availability(db, clinic.id, "cardiologist")

    assert result.found is True
    assert result.department_name == "Cardiology"
    assert len(result.doctors) == 1


def test_professional_title_fallback_works_for_dermatologist_too(db, clinic):
    dept = _department(db, clinic, name="Dermatology")
    doctor = _doctor(db, clinic, dept)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    _slot(db, clinic, doctor, future)

    result = get_department_availability(db, clinic.id, "Dermatologist")

    assert result.found is True
    assert result.department_name == "Dermatology"


def test_unknown_department_name_returns_not_found_with_real_department_list(db, clinic):
    _department(db, clinic, name="Cardiology")
    _department(db, clinic, name="Dermatology")

    result = get_department_availability(db, clinic.id, "Neurology")

    assert result.found is False
    assert result.department_name is None
    assert result.doctors == []
    assert set(result.available_department_names) == {"Cardiology", "Dermatology"}


def test_inactive_department_is_treated_as_not_found(db, clinic):
    _department(db, clinic, name="OldDept", is_active=False)

    result = get_department_availability(db, clinic.id, "OldDept")

    assert result.found is False
    assert "OldDept" not in result.available_department_names


# --- no free slots ----------------------------------------------------------------


def test_department_found_but_no_free_slots_returns_empty_doctor_list(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept)  # no slots created at all

    result = get_department_availability(db, clinic.id, "Cardiology")

    assert result.found is True
    assert result.department_name == "Cardiology"
    assert result.doctors == []


def test_only_open_future_slots_count_as_free(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    now = datetime.now(timezone.utc)

    _slot(db, clinic, doctor, now - timedelta(hours=1))  # already past
    _slot(db, clinic, doctor, now + timedelta(hours=1), status="booked")  # not open

    result = get_department_availability(db, clinic.id, "Cardiology")

    assert result.found is True
    assert result.doctors == []


def test_inactive_doctor_is_excluded_even_with_free_slots(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept, is_active=False)
    _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    result = get_department_availability(db, clinic.id, "Cardiology")

    assert result.found is True
    assert result.doctors == []


def test_multiple_doctors_only_those_with_free_slots_are_returned(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor_with_slots = _doctor(db, clinic, dept, full_name="Dr. Has Slots")
    doctor_without_slots = _doctor(db, clinic, dept, full_name="Dr. No Slots")
    _slot(db, clinic, doctor_with_slots, datetime.now(timezone.utc) + timedelta(days=1))

    result = get_department_availability(db, clinic.id, "Cardiology")

    names = {d.full_name for d in result.doctors}
    assert names == {"Dr. Has Slots"}
    assert "Dr. No Slots" not in names


def test_clinic_isolation_a_department_in_another_clinic_is_not_matched(db):
    clinic_a = Clinic(name="Clinic A", slug=f"a-{uuid.uuid4().hex[:8]}", timezone="UTC")
    clinic_b = Clinic(name="Clinic B", slug=f"b-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add_all([clinic_a, clinic_b])
    db.flush()

    _department(db, clinic_a, name="Cardiology")

    result = get_department_availability(db, clinic_b.id, "Cardiology")

    assert result.found is False
    assert result.available_department_names == []


# --- list_active_department_names (item 2: symptom-triage context injection) -------


def test_list_active_department_names_returns_only_active_departments_sorted(db, clinic):
    _department(db, clinic, name="Dermatology")
    _department(db, clinic, name="Cardiology")
    _department(db, clinic, name="OldDept", is_active=False)

    names = list_active_department_names(db, clinic.id)

    assert names == ["Cardiology", "Dermatology"]


def test_list_active_department_names_empty_when_no_departments(db, clinic):
    assert list_active_department_names(db, clinic.id) == []


# --- minimum slot guarantee (item 4) -------------------------------------------------


def test_at_least_four_free_slots_shown_when_that_many_exist(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    base = datetime.now(timezone.utc) + timedelta(days=1)
    for i in range(6):
        _slot(db, clinic, doctor, base + timedelta(hours=i))

    result = get_department_availability(db, clinic.id, "Cardiology")

    assert MAX_SLOTS_PER_DOCTOR >= 4, "MAX_SLOTS_PER_DOCTOR must guarantee at least 4 slots whenever that many exist"
    assert len(result.doctors[0].slots) >= 4


# --- earliest_date / day filter (item 4) ---------------------------------------------


def test_earliest_date_filters_out_slots_before_that_date(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    now = datetime.now(timezone.utc)

    soon_slot = _slot(db, clinic, doctor, now + timedelta(days=1))
    friday_date = (now + timedelta(days=5)).date()
    friday_slot = _slot(db, clinic, doctor, datetime.combine(friday_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=10))

    result = get_department_availability(db, clinic.id, "Cardiology", earliest_date=friday_date)

    slot_ids = {s.slot_id for s in result.doctors[0].slots}
    assert friday_slot.id in slot_ids
    assert soon_slot.id not in slot_ids


def test_earliest_date_in_the_past_does_not_resurrect_already_past_slots(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    now = datetime.now(timezone.utc)

    _slot(db, clinic, doctor, now - timedelta(days=1))  # already past
    future_slot = _slot(db, clinic, doctor, now + timedelta(days=1))

    result = get_department_availability(db, clinic.id, "Cardiology", earliest_date=(now - timedelta(days=30)).date())

    slot_ids = {s.slot_id for s in result.doctors[0].slots}
    assert slot_ids == {future_slot.id}


def test_no_earliest_date_behaves_exactly_as_before(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    _slot(db, clinic, doctor, datetime.now(timezone.utc) + timedelta(days=1))

    result = get_department_availability(db, clinic.id, "Cardiology", earliest_date=None)

    assert result.found is True
    assert len(result.doctors[0].slots) == 1


# --- latest_date / bounded window (e.g. "today or tomorrow") -------------------------


def test_latest_date_includes_slots_within_window_and_excludes_after(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    now = datetime.now(timezone.utc)

    in_window = _slot(db, clinic, doctor, now + timedelta(hours=2))
    after_window = _slot(db, clinic, doctor, now + timedelta(days=5))

    today = now.date()
    tomorrow = today + timedelta(days=1)
    result = get_department_availability(db, clinic.id, "Cardiology", earliest_date=today, latest_date=tomorrow)

    slot_ids = {s.slot_id for s in result.doctors[0].slots}
    assert slot_ids == {in_window.id}
    assert after_window.id not in slot_ids
    assert result.next_available_when is None


def test_latest_date_window_with_nothing_open_reports_real_next_available_slot(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    now = datetime.now(timezone.utc)

    friday_slot = _slot(db, clinic, doctor, now + timedelta(days=5))

    today = now.date()
    tomorrow = today + timedelta(days=1)
    result = get_department_availability(db, clinic.id, "Cardiology", earliest_date=today, latest_date=tomorrow)

    assert result.found is True
    assert result.doctors == []
    assert result.next_available_when == friday_slot.start_utc


def test_latest_date_window_with_no_slots_at_all_leaves_next_available_none(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept)
    now = datetime.now(timezone.utc)

    today = now.date()
    result = get_department_availability(db, clinic.id, "Cardiology", earliest_date=today, latest_date=today)

    assert result.found is True
    assert result.doctors == []
    assert result.next_available_when is None


# --- earliest_time / latest_time (time-of-day filter) ---------------------------------
# Reported live: "only show me available slots of dr farhan rehman after 12 pm on
# monday" and "show me his available slots after 12 pm" both silently ignored the
# time-of-day request and returned the same top-5-earliest-of-the-day slots — there
# was no time-filtering mechanism anywhere at all.


def test_earliest_time_filters_out_slots_before_that_local_time(db, clinic):
    from datetime import time

    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    # clinic fixture uses timezone="UTC", so local time-of-day == UTC time-of-day here.
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    morning = _slot(db, clinic, doctor, datetime.combine(tomorrow, time(10, 0), tzinfo=timezone.utc))
    afternoon = _slot(db, clinic, doctor, datetime.combine(tomorrow, time(14, 0), tzinfo=timezone.utc))

    result = get_department_availability(db, clinic.id, "Cardiology", earliest_time=time(12, 0))

    slot_ids = {s.slot_id for s in result.doctors[0].slots}
    assert slot_ids == {afternoon.id}
    assert morning.id not in slot_ids


def test_latest_time_filters_out_slots_after_that_local_time(db, clinic):
    from datetime import time

    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    morning = _slot(db, clinic, doctor, datetime.combine(tomorrow, time(10, 0), tzinfo=timezone.utc))
    afternoon = _slot(db, clinic, doctor, datetime.combine(tomorrow, time(14, 0), tzinfo=timezone.utc))

    result = get_department_availability(db, clinic.id, "Cardiology", latest_time=time(12, 0))

    slot_ids = {s.slot_id for s in result.doctors[0].slots}
    assert slot_ids == {morning.id}
    assert afternoon.id not in slot_ids


def test_time_of_day_filter_still_returns_up_to_max_slots_per_doctor(db, clinic):
    from datetime import time

    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()

    # 3 slots before noon (filtered out), then MAX_SLOTS_PER_DOCTOR + 2 slots after
    # noon — confirms the widened candidate pool finds real matches past the
    # earliest, filtered-out ones, and still caps at MAX_SLOTS_PER_DOCTOR.
    for hour in (8, 9, 10):
        _slot(db, clinic, doctor, datetime.combine(tomorrow, time(hour, 0), tzinfo=timezone.utc))
    afternoon_slots = [
        _slot(db, clinic, doctor, datetime.combine(tomorrow, time(hour, 0), tzinfo=timezone.utc))
        for hour in range(13, 13 + MAX_SLOTS_PER_DOCTOR + 2)
    ]

    result = get_department_availability(db, clinic.id, "Cardiology", earliest_time=time(12, 0))

    assert len(result.doctors[0].slots) == MAX_SLOTS_PER_DOCTOR
    returned_ids = {s.slot_id for s in result.doctors[0].slots}
    assert returned_ids <= {s.id for s in afternoon_slots}


def test_no_time_filter_behaves_exactly_as_before(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    doctor = _doctor(db, clinic, dept)
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    slot = _slot(db, clinic, doctor, datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=9))

    result = get_department_availability(db, clinic.id, "Cardiology")

    assert {s.slot_id for s in result.doctors[0].slots} == {slot.id}


# --- find_doctors_by_name -------------------------------------------------------------


def test_find_doctors_by_name_matches_reversed_word_order(db, clinic):
    # Reproduces the reported bug: patient typed "raza iqra" (reversed) for a real
    # "Dr. Iqra Raza" — word-level, order-independent matching must still find her.
    dept = _department(db, clinic, name="ENT")
    doctor = _doctor(db, clinic, dept, full_name="Dr. Iqra Raza")

    matches = find_doctors_by_name(db, clinic.id, "raza iqra")

    assert len(matches) == 1
    assert matches[0].full_name == "Dr. Iqra Raza"
    assert matches[0].department_name == "ENT"


def test_find_doctors_by_name_single_word_overlap_matches(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Babar Ali")

    matches = find_doctors_by_name(db, clinic.id, "Ali Baber")

    assert len(matches) == 1
    assert matches[0].full_name == "Dr. Babar Ali"


def test_find_doctors_by_name_returns_every_doctor_sharing_a_matched_word(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Ali Raza")
    _doctor(db, clinic, dept, full_name="Dr. Ali Khan")
    _doctor(db, clinic, dept, full_name="Dr. Sana Malik")

    matches = find_doctors_by_name(db, clinic.id, "dr ali")

    names = {m.full_name for m in matches}
    assert names == {"Dr. Ali Raza", "Dr. Ali Khan"}


def test_find_doctors_by_name_exact_full_name_short_circuits_past_unrelated_word_overlap(db, clinic):
    # Reproduces the reported bug: a patient typing the full, exact name "Dr. Ali
    # Raza" got stuck disambiguating against Babar Ali/Fatima Raza/Hina Raza — sharing
    # only a first or last name — instead of resolving directly to the one doctor
    # whose full name is an exact match.
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Ali Raza")
    _doctor(db, clinic, dept, full_name="Dr. Babar Ali")
    _doctor(db, clinic, dept, full_name="Dr. Fatima Raza")
    _doctor(db, clinic, dept, full_name="Dr. Hina Raza")

    matches = find_doctors_by_name(db, clinic.id, "Dr. Ali Raza")

    assert len(matches) == 1
    assert matches[0].full_name == "Dr. Ali Raza"


def test_find_doctors_by_name_exact_match_is_order_independent(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Ali Raza")
    _doctor(db, clinic, dept, full_name="Dr. Babar Ali")

    matches = find_doctors_by_name(db, clinic.id, "Raza Ali")

    assert len(matches) == 1
    assert matches[0].full_name == "Dr. Ali Raza"


def test_find_doctors_by_name_exact_match_works_when_name_is_embedded_in_a_full_sentence(db, clinic):
    # The name rarely arrives as a bare query in practice — it's typically embedded in
    # a full sentence like "I'd like to book with Dr. Ali Raza". The extra words
    # ("book", "with", ...) must not break the exact-match subset check.
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Ali Raza")
    _doctor(db, clinic, dept, full_name="Dr. Babar Ali")
    _doctor(db, clinic, dept, full_name="Dr. Fatima Raza")

    matches = find_doctors_by_name(db, clinic.id, "I'd like to book with Dr. Ali Raza")

    assert len(matches) == 1
    assert matches[0].full_name == "Dr. Ali Raza"


def test_find_doctors_by_name_partial_name_alone_still_falls_back_to_broad_overlap(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Ali Raza")
    _doctor(db, clinic, dept, full_name="Dr. Babar Ali")

    matches = find_doctors_by_name(db, clinic.id, "Ali")

    names = {m.full_name for m in matches}
    assert names == {"Dr. Ali Raza", "Dr. Babar Ali"}


def test_find_doctors_by_name_no_match_returns_empty(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Jane Example")

    assert find_doctors_by_name(db, clinic.id, "Dr. Someone Else") == []


def test_find_doctors_by_name_ignores_inactive_doctors_and_departments(db, clinic):
    active_dept = _department(db, clinic, name="Cardiology")
    inactive_dept = _department(db, clinic, name="OldDept", is_active=False)
    _doctor(db, clinic, active_dept, full_name="Dr. Sana Malik", is_active=False)
    _doctor(db, clinic, inactive_dept, full_name="Dr. Sana Rehman")

    assert find_doctors_by_name(db, clinic.id, "Sana") == []


def test_find_doctors_by_name_bare_query_with_no_real_words_returns_empty(db, clinic):
    dept = _department(db, clinic, name="Cardiology")
    _doctor(db, clinic, dept, full_name="Dr. Jane Example")

    assert find_doctors_by_name(db, clinic.id, "dr") == []
    assert find_doctors_by_name(db, clinic.id, "") == []
