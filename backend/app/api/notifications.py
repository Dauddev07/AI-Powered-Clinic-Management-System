import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.core.rate_limit import limiter
from app.models.notification import Notification
from app.models.user import User
from app.schemas.notification import NotificationListOut, NotificationOut

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=NotificationListOut)
@limiter.limit("60/minute")
def list_my_notifications(
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> NotificationListOut:
    """Own-only, clinic-scoped, newest first — ordered by seq (see the model's own
    tiebreaker note) rather than created_at alone, so notifications written in the
    same transaction (e.g. a scheduler tick auto-completing several appointments at
    once) never come back in an arbitrary order on a created_at tie.
    """
    clinic_id = current_user.clinic_id

    notifications = db.execute(
        select(Notification)
        .where(Notification.clinic_id == clinic_id, Notification.user_id == current_user.id)
        .order_by(Notification.seq.desc())
    ).scalars().all()

    unread_count = db.execute(
        select(func.count())
        .select_from(Notification)
        .where(
            Notification.clinic_id == clinic_id,
            Notification.user_id == current_user.id,
            Notification.read_at.is_(None),
        )
    ).scalar_one()

    return NotificationListOut(notifications=notifications, unread_count=unread_count)


@router.patch("/{notification_id}/read", response_model=NotificationOut)
@limiter.limit("30/minute")
def mark_notification_read(
    notification_id: uuid.UUID,
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> Notification:
    notification = db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.clinic_id == current_user.clinic_id,
            Notification.user_id == current_user.id,
        )
    ).scalar_one_or_none()
    if notification is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found.")

    if notification.read_at is None:
        notification.read_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(notification)
    return notification


@router.patch("/mark-all-read", response_model=NotificationListOut)
@limiter.limit("20/minute")
def mark_all_notifications_read(
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> NotificationListOut:
    clinic_id = current_user.clinic_id
    now = datetime.now(timezone.utc)

    unread = db.execute(
        select(Notification).where(
            Notification.clinic_id == clinic_id,
            Notification.user_id == current_user.id,
            Notification.read_at.is_(None),
        )
    ).scalars().all()
    for notification in unread:
        notification.read_at = now
    db.commit()

    notifications = db.execute(
        select(Notification)
        .where(Notification.clinic_id == clinic_id, Notification.user_id == current_user.id)
        .order_by(Notification.seq.desc())
    ).scalars().all()
    return NotificationListOut(notifications=notifications, unread_count=0)
