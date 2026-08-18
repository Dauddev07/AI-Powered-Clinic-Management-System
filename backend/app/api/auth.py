from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.db import get_db
from app.core.rate_limit import limiter
from app.core.security import create_access_token, hash_password, verify_password
from app.models.clinic import Clinic
from app.models.user import User
from app.schemas.auth import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    RefreshTokenRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserPublicOut,
    UserWithClinicOut,
    VerifyEmailRequest,
    VerifyResetOtpRequest,
)
from app.services.email import send_welcome_email
from app.services.email_verification import (
    InvalidOrExpiredOtp as InvalidOrExpiredVerificationOtp,
    OtpCooldownActive as VerificationOtpCooldownActive,
    request_email_verification,
    send_verification_for_new_registration,
    send_verification_otp_email,
    verify_email as apply_email_verification,
)
from app.services.password_reset import (
    InvalidOrExpiredOtp,
    OtpCooldownActive,
    apply_password_reset,
    request_password_reset,
    send_otp_email,
    verify_otp,
)
from app.services.refresh_tokens import (
    InvalidRefreshToken,
    issue_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
)

# Identical wording regardless of whether the email actually has an account, or
# whether a code was already sent moments ago — never lets a caller distinguish
# "no such account" from "check your inbox" (account enumeration).
_FORGOT_PASSWORD_GENERIC_RESPONSE = {
    "status": "If an account exists for this email, a verification code has been sent."
}
_RESEND_VERIFICATION_GENERIC_RESPONSE = {
    "status": "If an unverified account exists for this email, a verification code has been sent."
}

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(payload: LoginRequest, request: Request = None, db: Session = Depends(get_db)) -> TokenResponse:
    # Email is unique per clinic, not globally (a patient may register the same email at
    # more than one branch). The login screen has no branch selector, so the password
    # itself disambiguates which clinic account is being authenticated.
    candidates = db.execute(
        select(User).where(User.email == payload.email.lower(), User.is_active.is_(True))
    ).scalars().all()

    matched = next((u for u in candidates if verify_password(payload.password, u.hashed_password)), None)
    if matched is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not matched.email_verified:
        # 403, not 401: the password is correct (already confirmed above) — this is
        # an account-state block, not an auth failure, and the frontend's global 401
        # handler must not treat it as an expired/invalid session. Distinct status
        # code (not just message text) so the frontend can reliably route to the
        # verify-email screen instead of string-matching the detail text.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before logging in.",
        )

    token = create_access_token(user_id=matched.id, clinic_id=matched.clinic_id, role=matched.role)
    refresh_token = issue_refresh_token(db, matched)
    return TokenResponse(access_token=token, refresh_token=refresh_token, must_change_password=matched.must_change_password)


@router.get("/me", response_model=UserWithClinicOut)
def get_my_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> UserWithClinicOut:
    """Any authenticated role (patient or admin) — unlike /patients/me, which is
    patient-only. Read-only; account edits still go through the role-specific flows
    that already exist (patient profile PATCH, superadmin CLI for admins). Includes
    the clinic's display name (see UserWithClinicOut) for the app header.
    """
    clinic = db.get(Clinic, current_user.clinic_id)
    base = UserPublicOut.model_validate(current_user)
    return UserWithClinicOut(**base.model_dump(), clinic_name=clinic.name if clinic else "")


@router.post("/register", response_model=UserPublicOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> User:
    clinic = db.get(Clinic, payload.clinic_id)
    if clinic is None or not clinic.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clinic not found")

    # Phone is not unique — a household may share one number across several
    # patient accounts. Email (checked via the IntegrityError below) is the only
    # uniqueness constraint on registration.
    user = User(
        clinic_id=payload.clinic_id,
        role="patient",  # never read from the request body
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        phone=payload.phone,
        dob=payload.dob,
        gender=payload.gender,
        # Overrides the column's own server_default "true" (which only exists to
        # grandfather in pre-existing rows) — every fresh self-registration starts
        # unverified and must complete the OTP flow below before it can log in.
        email_verified=False,
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
    # Verification OTP only here, not the welcome email — that goes out once
    # verify_email() below actually confirms they own this address, not before.
    code = send_verification_for_new_registration(db, user)
    background_tasks.add_task(send_verification_otp_email, user, code)
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


@router.post("/forgot-password")
@limiter.limit("5/minute")
def forgot_password(
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    try:
        result = request_password_reset(db, payload.email)
    except OtpCooldownActive:
        # The only place this flow deviates from a flat generic response — but only
        # ever reached for an email that DOES have an account (see
        # request_password_reset's own docstring), so it can't be used to probe for
        # account existence; it just tells a real, already-requesting user to wait.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="A code was already sent recently — please wait a minute before requesting another.",
        )

    if result is not None:
        user, code = result
        background_tasks.add_task(send_otp_email, user, code)

    return _FORGOT_PASSWORD_GENERIC_RESPONSE


@router.post("/verify-reset-otp")
@limiter.limit("10/minute")
def verify_reset_otp(payload: VerifyResetOtpRequest, request: Request = None, db: Session = Depends(get_db)) -> dict:
    """Lets the frontend show its "code verified ✓" step before ever collecting a
    new password — doesn't consume the code (see verify_otp's own docstring); the
    actual /reset-password call re-checks it before applying anything.
    """
    try:
        verify_otp(db, payload.email, payload.otp)
    except InvalidOrExpiredOtp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That code is invalid or has expired. Please request a new one.",
        )
    return {"status": "verified"}


@router.post("/reset-password")
@limiter.limit("10/minute")
def reset_password(payload: ResetPasswordRequest, request: Request = None, db: Session = Depends(get_db)) -> dict:
    try:
        apply_password_reset(db, payload.email, payload.otp, payload.new_password)
    except InvalidOrExpiredOtp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That code is invalid or has expired. Please request a new one.",
        )
    return {"status": "password reset"}


@router.post("/resend-verification-email")
@limiter.limit("5/minute")
def resend_verification_email(
    payload: ResendVerificationRequest,
    background_tasks: BackgroundTasks,
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    try:
        result = request_email_verification(db, payload.email)
    except VerificationOtpCooldownActive:
        # Same reasoning as forgot-password's identical branch: only ever reached for
        # an email that DOES have an unverified account, so it can't be used to
        # probe account existence — just tells a real, already-requesting user to wait.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="A code was already sent recently — please wait a minute before requesting another.",
        )

    if result is not None:
        user, code = result
        background_tasks.add_task(send_verification_otp_email, user, code)

    return _RESEND_VERIFICATION_GENERIC_RESPONSE


@router.post("/verify-email", response_model=TokenResponse)
@limiter.limit("10/minute")
def verify_email(
    payload: VerifyEmailRequest,
    background_tasks: BackgroundTasks,
    request: Request = None,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """On success, logs the patient straight in (same as /login) — they just proved
    ownership of the email a moment ago, so making them log in again immediately
    after would be pure friction with no security benefit. The welcome email — held
    back at registration until this point — goes out now instead.
    """
    try:
        user = apply_email_verification(db, payload.email, payload.otp)
    except InvalidOrExpiredVerificationOtp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That code is invalid or has expired. Please request a new one.",
        )

    background_tasks.add_task(send_welcome_email, to=user.email, full_name=user.full_name)
    token = create_access_token(user_id=user.id, clinic_id=user.clinic_id, role=user.role)
    refresh_token = issue_refresh_token(db, user)
    return TokenResponse(access_token=token, refresh_token=refresh_token, must_change_password=user.must_change_password)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("30/minute")
def refresh(payload: RefreshTokenRequest, request: Request = None, db: Session = Depends(get_db)) -> TokenResponse:
    """Exchanges a still-valid refresh token for a new access token, rotating the
    refresh token itself in the process (see rotate_refresh_token's own docstring for
    why re-presenting an already-rotated token is treated as invalid rather than
    silently accepted).
    """
    try:
        user, new_refresh_token = rotate_refresh_token(db, payload.refresh_token)
    except InvalidRefreshToken:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    access_token = create_access_token(user_id=user.id, clinic_id=user.clinic_id, role=user.role)
    return TokenResponse(
        access_token=access_token, refresh_token=new_refresh_token, must_change_password=user.must_change_password
    )


@router.post("/logout")
def logout(payload: LogoutRequest, db: Session = Depends(get_db)) -> dict:
    """Revokes this one refresh token (i.e. this one device/session) — every other
    session the patient is logged into elsewhere is untouched. The now-short-lived
    access token this device already holds simply expires on its own within
    JWT_ACCESS_EXPIRE_MINUTES; there's nothing further to revoke for it.
    """
    revoke_refresh_token(db, payload.refresh_token)
    return {"status": "logged out"}
