from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.rate_limit import limiter
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.schemas.clinic import ClinicPublicOut, PublicTopRatedDoctorOut

TOP_RATED_DOCTORS_LIMIT = 3

router = APIRouter(prefix="/clinics", tags=["clinics"])


@router.get("/public-list", response_model=list[ClinicPublicOut])
@limiter.limit("30/minute")
def public_list(request: Request = None, db: Session = Depends(get_db)) -> list[Clinic]:
    """A patient must pick their branch before they have a token. Returns nothing
    beyond what the picker needs.
    """
    stmt = select(Clinic).where(Clinic.is_active.is_(True)).order_by(Clinic.name)
    return list(db.execute(stmt).scalars().all())


@router.get("/top-rated-doctors", response_model=list[PublicTopRatedDoctorOut])
@limiter.limit("30/minute")
def public_top_rated_doctors(request: Request = None, db: Session = Depends(get_db)) -> list[PublicTopRatedDoctorOut]:
    """Top 3 active doctors by average AppointmentFeedback rating, across every
    active clinic (ties broken by rating count, more ratings first) — powers the
    landing page's "top rated doctors" section, reached before a visitor has
    picked a clinic or logged in, so this is intentionally unscoped by clinic_id
    (unlike admin_dashboard.get_top_rated_doctors, which is scoped to one admin's
    clinic). Always computed live, never cached. A doctor with zero ratings can't
    appear at all (inner join on the rating subquery).
    """
    rating_subq = (
        select(
            AppointmentFeedback.doctor_id.label("doctor_id"),
            func.avg(AppointmentFeedback.rating).label("avg_rating"),
            func.count(AppointmentFeedback.id).label("rating_count"),
        )
        .group_by(AppointmentFeedback.doctor_id)
        .subquery()
    )
    visit_subq = (
        select(
            Appointment.doctor_id.label("doctor_id"),
            func.count(Appointment.id).label("visit_count"),
        )
        .where(Appointment.status == "completed")
        .group_by(Appointment.doctor_id)
        .subquery()
    )

    rows = db.execute(
        select(
            Doctor.id,
            Doctor.full_name,
            Department.name,
            Clinic.name,
            rating_subq.c.avg_rating,
            rating_subq.c.rating_count,
            func.coalesce(visit_subq.c.visit_count, 0),
        )
        .select_from(Doctor)
        .join(rating_subq, rating_subq.c.doctor_id == Doctor.id)
        .join(Department, Department.id == Doctor.department_id)
        .join(Clinic, Clinic.id == Doctor.clinic_id)
        .outerjoin(visit_subq, visit_subq.c.doctor_id == Doctor.id)
        .where(Doctor.is_active.is_(True), Clinic.is_active.is_(True))
        .order_by(rating_subq.c.avg_rating.desc(), rating_subq.c.rating_count.desc())
        .limit(TOP_RATED_DOCTORS_LIMIT)
    ).all()

    return [
        PublicTopRatedDoctorOut(
            doctor_id=str(doctor_id),
            doctor_name=doctor_name,
            department_name=department_name,
            clinic_name=clinic_name,
            average_rating=round(float(avg_rating), 1),
            rating_count=rating_count,
            visit_count=visit_count,
        )
        for doctor_id, doctor_name, department_name, clinic_name, avg_rating, rating_count, visit_count in rows
    ]
