import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.db import Base


class PushSubscription(Base):
    """One row per browser/device a patient has enabled Web Push notifications on
    (see app.services.push_notifications) — a patient with two devices subscribed
    has two rows, both get every push. Always created via the patient's own
    explicit "enable notifications" action (POST /notifications/push-subscribe);
    never created server-side on their behalf, unlike Notification rows.

    endpoint is the unique identity of a subscription (one per browser
    install/PushManager.subscribe() call) — re-subscribing the same browser
    replaces its row rather than accumulating duplicates.
    """

    __tablename__ = "push_subscriptions"
    __table_args__ = (UniqueConstraint("endpoint", name="uq_push_subscriptions_endpoint"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clinics.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    # The two keys the Web Push protocol needs to encrypt a push payload for this
    # specific subscription (RFC 8291) — opaque, browser-generated, base64url.
    p256dh_key: Mapped[str] = mapped_column(String(255), nullable=False)
    auth_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
