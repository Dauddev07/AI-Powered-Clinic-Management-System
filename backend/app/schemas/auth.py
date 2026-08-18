import re
import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.tld import is_valid_tld
from app.core.validators import (
    FULL_NAME_FORMAT_ERROR,
    FULL_NAME_RE,
    PHONE_FORMAT_ERROR,
    PHONE_RE,
    validate_dob_within_reasonable_range,
)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    # Opaque, long-lived, revocable — exchanged via POST /auth/refresh for a new
    # access token once the short-lived one above expires, instead of forcing a
    # full re-login every JWT_ACCESS_EXPIRE_MINUTES. See app/services/refresh_tokens.py.
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    must_change_password: bool


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class RegisterRequest(BaseModel):
    clinic_id: uuid.UUID
    full_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    # default="" (not a bare required field) so a completely omitted key also reaches
    # phone_required_and_formatted below and gets our clear "required" message, instead
    # of Pydantic's generic "Field required" — the field is still functionally
    # mandatory since the validator rejects blank/None/short values. validate_default
    # is required: Pydantic v2 otherwise skips validators entirely when a default is
    # used, which would silently accept an omitted phone as "".
    phone: str = Field(default="", max_length=32, validate_default=True)
    password: str = Field(min_length=8, max_length=128)
    dob: date
    gender: Literal["male", "female", "other"] | None = None

    @field_validator("email")
    @classmethod
    def email_has_recognized_tld(cls, value: str) -> str:
        tld = value.rsplit(".", 1)[-1]
        if not is_valid_tld(tld):
            raise ValueError(f"'.{tld}' is not a recognized email domain ending")
        return value

    @field_validator("full_name")
    @classmethod
    def full_name_letters_only(cls, value: str) -> str:
        if not FULL_NAME_RE.match(value):
            raise ValueError(FULL_NAME_FORMAT_ERROR)
        return value

    @field_validator("phone", mode="before")
    @classmethod
    def phone_required_and_formatted(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("Phone number is required")
        if not isinstance(value, str) or not PHONE_RE.match(value):
            raise ValueError(PHONE_FORMAT_ERROR)
        return value

    @field_validator("password")
    @classmethod
    def password_complexity(cls, value: str) -> str:
        if not re.search(r"[A-Za-z]", value):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"[0-9]", value):
            raise ValueError("Password must contain at least one number")
        return value

    @field_validator("dob")
    @classmethod
    def dob_within_reasonable_range(cls, value: date) -> date:
        return validate_dob_within_reasonable_range(value)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class VerifyResetOtpRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_complexity(cls, value: str) -> str:
        if not re.search(r"[A-Za-z]", value):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"[0-9]", value):
            raise ValueError("Password must contain at least one number")
        return value


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class UserPublicOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    clinic_id: uuid.UUID
    role: str
    email: str
    full_name: str
    phone: str | None
    dob: date | None
    gender: str | None
    must_change_password: bool
    email_verified: bool
    created_at: datetime


class UserWithClinicOut(UserPublicOut):
    """GET /auth/me only — adds the clinic's display name so the frontend header
    can show it without a second round-trip. Kept as its own subclass rather than
    adding clinic_name to UserPublicOut directly: that base schema is also used by
    POST /register's response, which returns the freshly-created User row straight
    from the ORM (from_attributes) with no clinic name resolved alongside it."""

    clinic_name: str
