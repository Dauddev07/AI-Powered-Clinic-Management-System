import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: uuid.UUID | None = None
    # Optional patient coordinates, sent by the frontend only at the moment a message
    # is about to be classified as an emergency (see app.services.nearby_hospitals) —
    # never required, and silently unused for every non-emergency turn.
    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    reply: str
    red_flag: bool = False


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime
    red_flag: bool = False

    model_config = {"from_attributes": True}


class ChatHistoryOut(BaseModel):
    session_id: uuid.UUID | None
    messages: list[ChatMessageOut]


class ChatSessionOut(BaseModel):
    session_id: uuid.UUID
    title: str
    last_message_at: datetime

    model_config = {"from_attributes": True}
