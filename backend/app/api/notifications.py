import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.config import settings
from app.core.db import get_db
from app.models.notification import Notification
from app.models.push_subscription import PushSubscription
from app.models.user import User
from app.schemas.notification import (
    NotificationListOut,
    NotificationOut,
    PushPublicKeyOut,
    PushSubscriptionIn,
    PushUnsubscribeIn,
)

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=NotificationListOut)
def list_my_notifications(
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
def mark_notification_read(
    notification_id: uuid.UUID,
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
def mark_all_notifications_read(
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


@router.get("/push-public-key", response_model=PushPublicKeyOut)
def get_push_public_key() -> PushPublicKeyOut:
    """No auth dependency — the VAPID public key is not a secret (it's handed to
    every subscribing browser anyway, same trust level as a public TLS cert), and
    the frontend needs it before it can know whether to even offer the "enable
    notifications" control at all."""
    return PushPublicKeyOut(public_key=settings.VAPID_PUBLIC_KEY)


@router.post("/push-subscribe", status_code=status.HTTP_204_NO_CONTENT)
def subscribe_to_push(
    payload: PushSubscriptionIn,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> None:
    """Upsert by endpoint (see PushSubscription's own unique constraint) — calling
    this again for the same browser (e.g. the patient re-enables notifications
    after having disabled them) replaces the row's keys rather than erroring or
    accumulating a duplicate."""
    existing = db.execute(
        select(PushSubscription).where(PushSubscription.endpoint == payload.endpoint)
    ).scalar_one_or_none()
    if existing is not None:
        existing.user_id = current_user.id
        existing.clinic_id = current_user.clinic_id
        existing.p256dh_key = payload.keys.p256dh
        existing.auth_key = payload.keys.auth
    else:
        db.add(
            PushSubscription(
                clinic_id=current_user.clinic_id,
                user_id=current_user.id,
                endpoint=payload.endpoint,
                p256dh_key=payload.keys.p256dh,
                auth_key=payload.keys.auth,
            )
        )
    db.commit()


@router.post("/push-unsubscribe", status_code=status.HTTP_204_NO_CONTENT)
def unsubscribe_from_push(
    payload: PushUnsubscribeIn,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db),
) -> None:
    db.execute(
        delete(PushSubscription).where(
            PushSubscription.endpoint == payload.endpoint,
            PushSubscription.user_id == current_user.id,
        )
    )
    db.commit()
