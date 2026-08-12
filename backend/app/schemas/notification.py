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
