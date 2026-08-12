"""Delivers a Web Push notification for a Notification row that's already been
committed to the database — see app.models.push_subscription's own docstring for
what a subscription row represents, and app.services.notifications for how
Notification rows themselves get created.

Deliberately NOT called from inside create_notification(): that function is
documented as "adds and flushes but never commits," meaning it can still be part
of a transaction that later rolls back (see booking_engine.py's IntegrityError
backstops). Sending a real push is an irreversible external side effect — it must
only happen once the caller's own db.commit() has actually succeeded, so this is
called explicitly, right after that commit, at each call site that wants push
delivery (not every one — see the product decision below).

Product decision: only appointment_rescheduled, appointment_cancelled, and the
four reminder types (appointment_reminder_60m/30m/5m, appointment_starting) get a
push. appointment_booked does NOT — a patient who just booked is, by definition,
already looking at the confirmation on screen; a phone notification for something
they're already looking at is redundant noise, not a genuine "you'd otherwise miss
this" alert. appointment_auto_completed is a housekeeping event, not something a
patient needs to act on right now, so it stays bell-icon-only too.

Never allowed to raise or block the caller — same "a bolt-on convenience must
never break the core flow" pattern as app.services.nearby_hospitals. Missing/
unconfigured VAPID keys (a clinic that hasn't run generate_vapid_keys.py yet)
means every call here is a silent, safe no-op.
"""
import base64
import json
import logging

from pywebpush import WebPushException, webpush
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.notification import Notification
from app.models.push_subscription import PushSubscription

logger = logging.getLogger(__name__)

# Same clinic name shown everywhere else patient-facing (chat header, etc.) — no
# per-clinic customization exists yet in this single-clinic-per-deploy setup, so a
# constant is the honest reflection of that, not a hardcoded shortcut around a
# feature that should exist.
_NOTIFICATION_TITLE = "Quick Check Clinic"

# Where tapping the notification lands — the patient's upcoming-appointments list,
# since every push type here (reminder/rescheduled/cancelled) is about an
# appointment they'd want to see, not any one specific screen per type.
_CLICK_URL = "/patient/appointments"


def _vapid_private_pem() -> str | None:
    if not settings.VAPID_PRIVATE_KEY_B64:
        return None
    return base64.b64decode(settings.VAPID_PRIVATE_KEY_B64).decode()


def send_push_for_notification(db: Session, notification: Notification) -> None:
    """Sends `notification` to every device the patient has subscribed on. Reads
    notification.user_id/message/type/related_appointment_id/clinic_id — safe to
    call right after db.commit() (SQLAlchemy transparently reloads expired
    attributes on access), no manual db.refresh() needed by the caller."""
    private_pem = _vapid_private_pem()
    if not private_pem or not settings.VAPID_PUBLIC_KEY or not settings.VAPID_CONTACT_EMAIL:
        return

    subscriptions = list(
        db.execute(
            select(PushSubscription).where(PushSubscription.user_id == notification.user_id)
        ).scalars()
    )
    if not subscriptions:
        return

    payload = {
        "title": _NOTIFICATION_TITLE,
        "body": notification.message,
        "url": _CLICK_URL,
        # Same-tag pushes replace each other in the OS notification tray instead
        # of stacking — the 60m/30m/5m/starting reminder sequence for one
        # appointment shares a tag, so an un-dismissed "60 min" reminder gets
        # replaced by "30 min" rather than leaving stale reminders piled up.
        "tag": str(notification.related_appointment_id or notification.id),
    }
    vapid_claims = {"sub": f"mailto:{settings.VAPID_CONTACT_EMAIL}"}

    dead_subscription_ids = []
    for subscription in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": subscription.endpoint,
                    "keys": {"p256dh": subscription.p256dh_key, "auth": subscription.auth_key},
                },
                data=json.dumps(payload),
                vapid_private_key=private_pem,
                vapid_claims=dict(vapid_claims),
            )
        except WebPushException as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code in (404, 410):
                # The browser itself has revoked/expired this subscription (site
                # uninstalled, permission revoked, etc.) — the push service is
                # telling us it will NEVER succeed again, so stop storing it
                # rather than retrying forever on every future notification.
                dead_subscription_ids.append(subscription.id)
            else:
                logger.warning("push_notifications: send failed for subscription %s: %s", subscription.id, exc)

    if dead_subscription_ids:
        db.execute(delete(PushSubscription).where(PushSubscription.id.in_(dead_subscription_ids)))
        db.commit()
