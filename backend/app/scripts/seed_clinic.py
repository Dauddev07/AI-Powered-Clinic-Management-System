"""One-time internal setup script — run once by the platform owner to onboard the
first clinic. CLI only; no arguments, no prompts, no superadmin UI in this phase.

Usage (from backend/, with the venv activated):
    python -m app.scripts.seed_clinic

Creates: the clinic row, exactly one admin user, and the clinic's empty ChromaDB
hospital-info collection, named per clinic so it cannot collide with a future
clinic N+1.

SECURITY: CLINIC_ADMIN_PASSWORD below is used ONLY as input to the same bcrypt hashing
function used everywhere else in the system (app.core.security.hash_password). Only the
resulting hash is ever written to the database. The plaintext value is never logged,
printed, or persisted anywhere — do not add a print/log statement that includes it.
"""

import re
import sys

import chromadb

from app.core.config import settings
from app.core.db import SessionLocal
from app.core.security import hash_password
from app.models.clinic import Clinic
from app.models.user import User

# --- Hardcoded seed constants (per client decision: no argparse/CLI args/prompts) ---
CLINIC_NAME = "Quickcheck Clinic"
CLINIC_ADMIN_EMAIL = "admin@quickcheck.example"
CLINIC_ADMIN_FULL_NAME = "Quickcheck Clinic Admin"
CLINIC_ADMIN_PASSWORD = "QuickCheck#2026"  # noqa: S105 - hashed below, never stored/logged raw
# --------------------------------------------------------------------------------


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug


def main() -> None:
    slug = _slugify(CLINIC_NAME)
    db = SessionLocal()

    try:
        existing = db.query(Clinic).filter(Clinic.slug == slug).first()
        if existing is not None:
            sys.exit(
                f"Refusing to seed: a clinic with slug '{slug}' (from name "
                f"'{CLINIC_NAME}') already exists (id={existing.id}). "
                "Re-running this script must not create a duplicate tenant."
            )

        clinic = Clinic(name=CLINIC_NAME, slug=slug)
        db.add(clinic)
        db.flush()  # assign clinic.id without committing yet

        admin = User(
            clinic_id=clinic.id,
            role="admin",
            email=CLINIC_ADMIN_EMAIL.lower(),
            hashed_password=hash_password(CLINIC_ADMIN_PASSWORD),
            full_name=CLINIC_ADMIN_FULL_NAME,
            must_change_password=False,  # hardcoded credential, not a rotating temp password
        )
        db.add(admin)
        db.commit()
        db.refresh(clinic)
        db.refresh(admin)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    hospital_info_name = f"{slug}__hospital-info"

    chroma = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, ssl=settings.CHROMA_SSL)
    chroma.get_or_create_collection(hospital_info_name)

    print(f"Seeded clinic '{CLINIC_NAME}' (id={clinic.id}, slug={slug}).")
    print(f"Created admin user (id={admin.id}, email={admin.email}).")
    print(f"Created ChromaDB collection: '{hospital_info_name}'.")


if __name__ == "__main__":
    main()
