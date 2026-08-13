import uuid

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api.auth import get_my_account, login, register
from app.core.security import hash_password
from app.models.clinic import Clinic
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest


def _clinic(db, name="Quickcheck Clinic"):
    c = Clinic(name=name, slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


def _user(db, clinic, role="patient", email_verified=True):
    u = User(
        clinic_id=clinic.id, role=role, email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Test User", email_verified=email_verified,
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


def test_login_blocks_a_correct_password_on_an_unverified_account(db):
    clinic = _clinic(db)
    user = User(
        clinic_id=clinic.id, role="patient", email="unverified@example.com",
        hashed_password=hash_password("CorrectPass123"), full_name="Test User",
        email_verified=False,
    )
    db.add(user)
    db.flush()

    with pytest.raises(HTTPException) as exc_info:
        login(LoginRequest(email="unverified@example.com", password="CorrectPass123"), db=db)

    assert exc_info.value.status_code == 403


def test_login_succeeds_for_a_verified_account_with_correct_password(db):
    clinic = _clinic(db)
    user = User(
        clinic_id=clinic.id, role="patient", email="verified@example.com",
        hashed_password=hash_password("CorrectPass123"), full_name="Test User",
        email_verified=True,
    )
    db.add(user)
    db.flush()

    result = login(LoginRequest(email="verified@example.com", password="CorrectPass123"), db=db)

    assert result.access_token


def test_register_creates_an_unverified_account(db):
    clinic = _clinic(db)

    payload = RegisterRequest(
        clinic_id=clinic.id,
        full_name="New Patient",
        email=f"{uuid.uuid4().hex}@example.com",
        phone="+923001234567",
        password="NewPass123",
        dob="1995-01-01",
    )
    result = register(payload, background_tasks=BackgroundTasks(), db=db)

    assert result.email_verified is False
