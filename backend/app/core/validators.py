import re
from datetime import date, datetime, timezone

# Pakistan-only for now: +92 followed by exactly 10 digits, first digit not 0
# (e.g. +923001234567 valid, +920123456789 invalid). Expand to other countries later.
# Shared by RegisterRequest (required) and PatientProfileUpdate (optional) so both
# enforce the exact same format.
PHONE_RE = re.compile(r"^\+92[1-9]\d{9}$")

PHONE_FORMAT_ERROR = "Phone number must be +92 followed by 10 digits, not starting with 0 (e.g. +923001234567)"

# Shared by RegisterRequest and PatientProfileUpdate so both enforce identical full-name rules.
FULL_NAME_RE = re.compile(r"^[A-Za-z' \-]+$")
FULL_NAME_FORMAT_ERROR = "Full name can only contain letters, spaces, hyphens, and apostrophes"

# Shared by RegisterRequest and PatientProfileUpdate so both enforce identical DOB rules.
MIN_DOB_YEARS = 120


def validate_dob_within_reasonable_range(value: date) -> date:
    today = datetime.now(timezone.utc).date()
    if value >= today:
        raise ValueError("Date of birth must be in the past")
    try:
        earliest = today.replace(year=today.year - MIN_DOB_YEARS)
    except ValueError:
        # today is Feb 29 on a leap year; fall back a day
        earliest = today.replace(month=2, day=28, year=today.year - MIN_DOB_YEARS)
    if value < earliest:
        raise ValueError(f"Date of birth must be within the last {MIN_DOB_YEARS} years")
    return value
