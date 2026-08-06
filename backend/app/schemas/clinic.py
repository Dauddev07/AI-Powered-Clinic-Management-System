import uuid

from pydantic import BaseModel, ConfigDict


class ClinicPublicOut(BaseModel):
    """Only what a patient needs to pick a branch before they have a token."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str


class PublicTopRatedDoctorOut(BaseModel):
    """Public, cross-clinic version of admin_dashboard.TopRatedDoctorOut — shown on
    the landing page, which is reached before a visitor has picked a clinic or
    logged in, so it also carries clinic_name (the admin dashboard version doesn't
    need it, since it's already scoped to one admin's clinic).
    """

    doctor_id: str
    doctor_name: str
    department_name: str
    clinic_name: str
    average_rating: float
    rating_count: int
    visit_count: int
