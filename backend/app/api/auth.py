from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.db import get_db
from app.core.security import create_access_token, hash_password, verify_password
from app.models.clinic import Clinic
from app.models.user import User
from app.schemas.auth import ChangePasswordRequest, LoginRequest, RegisterRequest, TokenResponse, UserPublicOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    # Email is unique per clinic, not globally (a patient may register the same email at
    # more than one branch). The login screen has no branch selector, so the password
    # itself disambiguates which clinic account is being authenticated.
    candidates = db.execute(
        select(User).where(User.email == payload.email.lower(), User.is_active.is_(True))
    ).scalars().all()

    matched = next((u for u in candidates if verify_password(payload.password, u.hashed_password)), None)
    if matched is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token(user_id=matched.id, clinic_id=matched.clinic_id, role=matched.role)
    return TokenResponse(access_token=token, must_change_password=matched.must_change_password)


@router.get("/me", response_model=UserPublicOut)
def get_my_account(current_user: User = Depends(get_current_user)) -> User:
    """Any authenticated role (patient or admin) — unlike /patients/me, which is
    patient-only. Read-only; account edits still go through the role-specific flows
    that already exist (patient profile PATCH, superadmin CLI for admins).
    """
    return current_user


@router.post("/register", response_model=UserPublicOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> User:
    clinic = db.get(Clinic, payload.clinic_id)
    if clinic is None or not clinic.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clinic not found")

    existing_phone = db.execute(
        select(User).where(
            User.clinic_id == payload.clinic_id,
            User.role == "patient",
            User.phone == payload.phone,
        )
    ).scalars().first()
    if existing_phone is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This phone number is already registered at this clinic.",
        )

    user = User(
        clinic_id=payload.clinic_id,
        role="patient",  # never read from the request body
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        phone=payload.phone,
        dob=payload.dob,
        gender=payload.gender,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists for this clinic",
        )
    db.refresh(user)
    return user


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not verify_password(payload.old_password, current_user.hashed_password):
        # 400, not 401: this is a client input error (wrong password typed into the
        # form), not an authentication/session failure. The caller's token is still
        # valid — using 401 here would be indistinguishable from an expired/tampered
        # token to the frontend's global 401 handler, which wrongly logs the user out.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is incorrect")

    if verify_password(payload.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from your current password",
        )

    current_user.hashed_password = hash_password(payload.new_password)
    current_user.must_change_password = False
    # Invalidates every token issued before now, including the one used for this
    # very request — get_current_user rejects any token whose iat predates this.
    # Floored to whole seconds: JWT `iat` only has 1-second resolution (jose truncates
    # it on encode), so a microsecond-precise timestamp here could exceed the `iat` of
    # a token freshly issued in the very same second, incorrectly invalidating it too.
    current_user.password_changed_at = datetime.now(timezone.utc).replace(microsecond=0)
    db.commit()
    return {"status": "password changed"}
