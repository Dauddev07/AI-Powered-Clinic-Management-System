import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class NotificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: str
    message: str
    related_appointment_id: uuid.UUID | None
    read_at: datetime | None
    created_at: datetime


class NotificationListOut(BaseModel):
    notifications: list[NotificationOut]
    unread_count: int


class PushSubscriptionKeysIn(BaseModel):
    p256dh: str
    auth: str


class PushSubscriptionIn(BaseModel):
    # Field names match the raw shape of the browser's own PushSubscription.toJSON()
    # output exactly, so the frontend can forward it with no reshaping.
    endpoint: str
    keys: PushSubscriptionKeysIn


class PushUnsubscribeIn(BaseModel):
    endpoint: str


class PushPublicKeyOut(BaseModel):
    # Empty string when this clinic hasn't generated VAPID keys yet (see
    # app.scripts.generate_vapid_keys) — the frontend treats that as "push isn't
    # available," never a broken/missing value to retry.
    public_key: str
