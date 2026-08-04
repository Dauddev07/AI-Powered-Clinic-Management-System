import uuid

from app.api.auth import get_my_account
from app.models.clinic import Clinic
from app.models.user import User


def _clinic(db, name="Quickcheck Clinic"):
    c = Clinic(name=name, slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _user(db, clinic, role="patient"):
    u = User(
        clinic_id=clinic.id, role=role, email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Test User",
    )
    db.add(u)
    db.flush()
    db.refresh(u)
    return u


def test_get_my_account_includes_the_clinic_name(db):
    clinic = _clinic(db, name="Quickcheck Clinic")
    user = _user(db, clinic)

    result = get_my_account(current_user=user, db=db)

    assert result.clinic_name == "Quickcheck Clinic"
    assert result.id == user.id
    assert result.role == "patient"


def test_get_my_account_works_for_admin_role_too(db):
    clinic = _clinic(db, name="Admin Test Clinic")
    admin = _user(db, clinic, role="admin")

    result = get_my_account(current_user=admin, db=db)

    assert result.clinic_name == "Admin Test Clinic"
    assert result.role == "admin"
