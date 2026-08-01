from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import get_clinic_scope, require_role
from app.core.tenancy import ClinicScope
from app.models.user import User
from app.schemas.patient import PatientProfileOut, PatientProfileUpdate

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/me", response_model=PatientProfileOut)
def get_my_profile(
    current_user: User = Depends(require_role("patient")),
    scope: ClinicScope = Depends(get_clinic_scope),
) -> User:
    return scope(User).get(current_user.id)


@router.patch("/me", response_model=PatientProfileOut)
def update_my_profile(
    payload: PatientProfileUpdate,
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
