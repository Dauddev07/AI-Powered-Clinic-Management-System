import uuid

import pytest

from app.api.admin_doctors import list_doctors
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.user import User


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def admin(db, clinic):
    a = User(
        clinic_id=clinic.id, role="admin", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Admin User",
    )
    db.add(a)
    db.flush()
    return a


def _doctor(db, clinic, department, full_name, specialization=None):
    d = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:6]}",
        full_name=full_name, specialization=specialization, is_active=True,
    )
    db.add(d)
    db.flush()
    return d


@pytest.fixture
def seeded(db, clinic):
    cardiology = Department(clinic_id=clinic.id, name="Cardiology")
    dermatology = Department(clinic_id=clinic.id, name="Dermatology")
    db.add_all([cardiology, dermatology])
    db.flush()

    jane = _doctor(db, clinic, cardiology, "Dr. Jane Example", specialization="Interventional Cardiology")
    bilal = _doctor(db, clinic, dermatology, "Dr. Bilal Ahmed", specialization="Skin allergy testing")
    return {"cardiology": cardiology, "dermatology": dermatology, "jane": jane, "bilal": bilal}


def test_no_query_returns_every_doctor(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q=None, current_user=admin, db=db)
    assert result.total == 2


def test_search_matches_doctor_name(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q="jane", current_user=admin, db=db)
    assert result.total == 1
    assert result.items[0].full_name == "Dr. Jane Example"


def test_search_matches_department_name(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q="Dermatology", current_user=admin, db=db)
    assert result.total == 1
    assert result.items[0].full_name == "Dr. Bilal Ahmed"


def test_search_matches_specialization(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q="allergy", current_user=admin, db=db)
    assert result.total == 1
    assert result.items[0].full_name == "Dr. Bilal Ahmed"


def test_search_is_case_insensitive_and_partial(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q="cardio", current_user=admin, db=db)
    assert result.total == 1
    assert result.items[0].full_name == "Dr. Jane Example"


def test_search_with_no_match_returns_empty(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q="Neurology", current_user=admin, db=db)
    assert result.total == 0
    assert result.items == []


def test_blank_query_behaves_like_no_query(db, clinic, admin, seeded):
    result = list_doctors(limit=20, offset=0, q="   ", current_user=admin, db=db)
    assert result.total == 2


def test_search_scoped_to_own_clinic_only(db, admin, seeded):
    other_clinic = Clinic(name="Other Clinic", slug=f"other-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(other_clinic)
    db.flush()
    other_dept = Department(clinic_id=other_clinic.id, name="Cardiology")
    db.add(other_dept)
    db.flush()
    _doctor(db, other_clinic, other_dept, "Dr. Jane Example")

    result = list_doctors(limit=20, offset=0, q="jane", current_user=admin, db=db)
    assert result.total == 1
