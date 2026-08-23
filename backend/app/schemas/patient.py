import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.validators import (
    FULL_NAME_FORMAT_ERROR,
    FULL_NAME_RE,
    PHONE_FORMAT_ERROR,
    PHONE_RE,
    validate_dob_within_reasonable_range,
)


class PatientProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    clinic_id: uuid.UUID
    full_name: str
    email: str
    phone: str | None
    dob: date | None
    gender: str | None


class PatientProfileUpdate(BaseModel):
    # Fields stay Optional at the type level so the request can still omit one to leave it
    # unchanged (see patients.py's exclude_unset) — but full_name/dob may not be explicitly
    # submitted as blank/null; their own validators enforce that. phone keeps its legacy
    # nullable exception, enforced in the endpoint where the current DB value is known.
    full_name: str | None = Field(default=None, max_length=255)
    phone: str | None = Field(default=None, max_length=32)
    dob: date | None = None
    gender: Literal["male", "female", "other"] | None = None

    @field_validator("full_name")
    @classmethod
    def full_name_required_and_valid(cls, value: str | None) -> str:
        if value is None or not value.strip():
            raise ValueError("Full name is required")
        if not FULL_NAME_RE.match(value):
            raise ValueError(FULL_NAME_FORMAT_ERROR)
        return value

    @field_validator("phone")
    @classmethod
    def phone_format_if_provided(cls, value: str | None) -> str | None:
        # Format only here — whether an existing/legacy-null phone may be cleared depends
        # on the patient's current DB value, which this schema can't see; that rule is
        # enforced in patients.py's update_my_profile.
        if value is None or value == "":
            return value
        if not PHONE_RE.match(value):
            raise ValueError(PHONE_FORMAT_ERROR)
        return value

    @field_validator("dob")
    @classmethod
    def dob_required_and_valid(cls, value: date | None) -> date:
        if value is None:
            raise ValueError("Date of birth is required")
        return validate_dob_within_reasonable_range(value)


class ConfirmDeleteAccountRequest(BaseModel):
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")
