"""One-time internal fix — corrects clinics.timezone away from the DB column's
"UTC" server_default, which seed_clinic.py never overrides. Every clinic-local time
(appointment displays, doctor CSV shift conversion, slot-generation day boundaries)
was silently computed as if the clinic were in UTC instead of Asia/Karachi.

Idempotent: only touches rows still set to "UTC", so re-running this after an admin
has deliberately set a different timezone is a no-op for that clinic rather than
clobbering their choice.

Usage (from backend/, with the venv activated):
    python -m app.scripts.fix_clinic_timezone
"""
from app.core.db import SessionLocal
from app.models.clinic import Clinic

CORRECT_TIMEZONE = "Asia/Karachi"


def main() -> None:
    db = SessionLocal()
    try:
        clinics = db.query(Clinic).filter(Clinic.timezone == "UTC").all()
        if not clinics:
            print("No clinics with timezone='UTC' found — nothing to fix.")
            return

        for clinic in clinics:
            clinic.timezone = CORRECT_TIMEZONE
            print(f"Updated clinic '{clinic.name}' (id={clinic.id}) timezone: UTC -> {CORRECT_TIMEZONE}")

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
