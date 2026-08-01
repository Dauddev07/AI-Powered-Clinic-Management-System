import uuid

from pydantic import BaseModel, ConfigDict


class ClinicPublicOut(BaseModel):
    """Only what a patient needs to pick a branch before they have a token.
    This is the only unauthenticated cross-clinic endpoint in the system."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
