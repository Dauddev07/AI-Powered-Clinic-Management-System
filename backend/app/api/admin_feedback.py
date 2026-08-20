from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.models.appointment_feedback import AppointmentFeedback
from app.models.doctor import Doctor
from app.models.user import User
from app.schemas.appointment_feedback import FeedbackInsightOut, FeedbackListOut, FeedbackOut, FeedbackToneCounts
from app.services.feedback_insights import get_feedback_digest

router = APIRouter(prefix="/admin/feedback", tags=["admin-feedback"])

# Same 1-2 / 3 / 4-5 split used by app.services.feedback to decide the patient-facing
# acknowledgement message — kept in sync so "bad" here means the same thing it does
# at submission time.
_TONE_RANGES = {"bad": (1, 2), "neutral": (3, 3), "good": (4, 5)}


@router.get("", response_model=FeedbackListOut)
def list_feedback(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    tone: str | None = Query(default=None, pattern="^(good|neutral|bad)$"),
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> FeedbackListOut:
    base_filter = (AppointmentFeedback.clinic_id == current_user.clinic_id,)

    # Clinic-wide summary — deliberately computed from `base_filter` only, never
    # `tone`, so the stat tiles above the table always describe every rating on
    # file regardless of which tab the admin currently has selected.
    average_rating = db.execute(
        select(func.avg(AppointmentFeedback.rating)).where(*base_filter)
    ).scalar_one()
    tone_case = case(
        (AppointmentFeedback.rating >= 4, "good"),
        (AppointmentFeedback.rating == 3, "neutral"),
        else_="bad",
    )
    tone_rows = db.execute(
        select(tone_case.label("tone"), func.count()).where(*base_filter).group_by(tone_case)
    ).all()
    tone_counts = {"good": 0, "neutral": 0, "bad": 0}
    for tone_label, count in tone_rows:
        tone_counts[tone_label] = count

    list_filter = list(base_filter)
    if tone is not None:
        low, high = _TONE_RANGES[tone]
        list_filter.append(AppointmentFeedback.rating.between(low, high))

    total = db.execute(select(func.count()).select_from(AppointmentFeedback).where(*list_filter)).scalar_one()

    rows = db.execute(
        select(AppointmentFeedback, User, Doctor)
        .join(User, User.id == AppointmentFeedback.patient_id)
        .join(Doctor, Doctor.id == AppointmentFeedback.doctor_id)
        .where(*list_filter)
        .order_by(AppointmentFeedback.created_at.desc(), AppointmentFeedback.seq.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    items = [
        FeedbackOut(
            id=feedback.id,
            patient_name=patient.full_name,
            doctor_name=doctor.full_name,
            rating=feedback.rating,
            reason=feedback.reason,
            created_at=feedback.created_at,
        )
        for feedback, patient, doctor in rows
    ]
    return FeedbackListOut(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
        average_rating=round(float(average_rating), 2) if average_rating is not None else None,
        tone_counts=FeedbackToneCounts(**tone_counts),
    )


@router.get("/insights", response_model=FeedbackInsightOut)
def get_feedback_insights(
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> FeedbackInsightOut:
    """A short, LLM-generated synthesis of recurring themes in this clinic's
    low-rating feedback (see app.services.feedback_insights) — cached and refreshed
    at most once a week, so this is cheap to call on every feedback-page visit.
    `digest` is None both when there's no low-rating feedback with a reason yet and
    when no digest has ever been successfully generated; the frontend hides the card
    entirely in either case rather than showing an error.
    """
    digest, generated_at = get_feedback_digest(db, current_user.clinic_id)
    db.commit()
    return FeedbackInsightOut(digest=digest, generated_at=generated_at)
