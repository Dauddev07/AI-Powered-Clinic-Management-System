from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.clinic import Clinic
from app.schemas.clinic import ClinicPublicOut

router = APIRouter(prefix="/clinics", tags=["clinics"])


@router.get("/public-list", response_model=list[ClinicPublicOut])
def public_list(db: Session = Depends(get_db)) -> list[Clinic]:
    """The only unauthenticated cross-clinic endpoint in the system: a patient must pick
    their branch before they have a token. Returns nothing beyond what the picker needs.
    """
    stmt = select(Clinic).where(Clinic.is_active.is_(True)).order_by(Clinic.name)
    return list(db.execute(stmt).scalars().all())
