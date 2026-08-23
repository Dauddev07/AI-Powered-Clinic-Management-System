from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_clinic_scope, require_role
from app.core.db import get_db
from app.core.rate_limit import limiter
from app.core.tenancy import ClinicScope
from app.models.user import User
from app.schemas.patient import ConfirmDeleteAccountRequest, PatientProfileOut, PatientProfileUpdate
from app.services.account_deletion import (
    InvalidOrExpiredOtp,
    OtpCooldownActive,
    confirm_and_delete_account,
    request_delete_otp,
    send_otp_email,
)

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/me", response_model=PatientProfileOut)
@limiter.limit("60/minute")
def get_my_profile(
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    scope: ClinicScope = Depends(get_clinic_scope),
) -> User:
    return scope(User).get(current_user.id)


@router.patch("/me", response_model=PatientProfileOut)
@limiter.limit("20/minute")
def update_my_profile(
    payload: PatientProfileUpdate,
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    scope: ClinicScope = Depends(get_clinic_scope),
) -> User:
    repo = scope(User)
    user = repo.get(current_user.id)
    updates = payload.model_dump(exclude_unset=True)

    # Once a phone number exists — whether already on file or being set right now — it
    # can't be cleared back to blank. A patient who registered before phone was required
    # (and so still has none on file) is exempt until they add one.
    if "phone" in updates and not updates["phone"] and user.phone:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Phone number cannot be cleared once set.",
        )

    phone = updates.get("phone")
    if phone:
        existing_phone = scope.db.execute(
            select(User).where(
                User.clinic_id == current_user.clinic_id,
                User.role == "patient",
                User.phone == phone,
                User.id != current_user.id,
            )
        ).scalars().first()
        if existing_phone is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This phone number is already registered at this clinic.",
            )

    updated = repo.update(user, **updates)
    scope.db.commit()
    return updated


@router.post("/me/delete/request-otp")
@limiter.limit("5/minute")
def request_delete_account_otp(
    background_tasks: BackgroundTasks,
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> dict:
    try:
        code = request_delete_otp(db, current_user)
    except OtpCooldownActive:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="A code was already sent recently — please wait a minute before requesting another.",
        )
    background_tasks.add_task(send_otp_email, current_user, code)
    return {"status": "otp sent"}


@router.post("/me/delete/confirm")
@limiter.limit("10/minute")
def confirm_delete_account(
    payload: ConfirmDeleteAccountRequest,
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> dict:
    try:
        confirm_and_delete_account(db, current_user, payload.otp)
    except InvalidOrExpiredOtp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That code is invalid or has expired. Please request a new one.",
        )
    return {"status": "account deleted"}
