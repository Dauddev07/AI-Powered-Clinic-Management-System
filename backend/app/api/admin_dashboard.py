from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.core.db import get_db
from app.core.rate_limit import limiter
from app.models.appointment import Appointment
from app.models.appointment_feedback import AppointmentFeedback
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.slot import Slot
from app.models.user import User
from app.schemas.admin_dashboard import (
    AdminDashboardStatsOut,
    AdminWeeklyDigestOut,
    DailyAppointmentCountOut,
    DoctorAppointmentCountOut,
    SlotUtilizationOut,
    TopRatedDoctorOut,
)

TOP_RATED_DOCTORS_LIMIT = 3

router = APIRouter(prefix="/admin/dashboard", tags=["admin-dashboard"])


@router.get("/stats", response_model=AdminDashboardStatsOut)
@limiter.limit("60/minute")
def get_admin_dashboard_stats(
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> AdminDashboardStatsOut:
    """Three read-only rollups for the admin dashboard, all scoped to this admin's
    own clinic_id (never client input): active vs. total doctor counts, and
    today's slot utilization. Nothing here writes anything.
    """
    clinic_id = current_user.clinic_id
    clinic = db.get(Clinic, clinic_id)

    active_doctors_count = db.execute(
        select(func.count())
        .select_from(Doctor)
        .where(Doctor.clinic_id == clinic_id, Doctor.is_active.is_(True))
    ).scalar_one()
    total_doctors_count = db.execute(
        select(func.count()).select_from(Doctor).where(Doctor.clinic_id == clinic_id)
    ).scalar_one()

    # "Today" is the clinic's own local calendar day, not UTC's — the same
    # local-midnight-to-local-midnight boundary list_slots uses for its date_from/
    # date_to filters, so this matches what an admin would see filtering slots to
    # today there.
    tz = ZoneInfo(clinic.timezone)
    today_local = datetime.now(tz).date()
    day_start_utc = datetime.combine(today_local, time.min, tzinfo=tz).astimezone(timezone.utc)
    day_end_utc = datetime.combine(today_local + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)

    # Total slots today and how many of them are booked — a slot stays 'booked'
    # once its appointment is confirmed and remains so even after that
    # appointment auto-completes (only a cancellation reopens it, see
    # cancel_appointment in appointments.py), so this counts every slot with an
    # appointment against it today regardless of whether that appointment has
    # since completed or is still upcoming.
    total_today = db.execute(
        select(func.count())
        .select_from(Slot)
        .where(Slot.clinic_id == clinic_id, Slot.start_utc >= day_start_utc, Slot.start_utc < day_end_utc)
    ).scalar_one()
    booked_today = db.execute(
        select(func.count())
        .select_from(Slot)
        .where(
            Slot.clinic_id == clinic_id,
            Slot.start_utc >= day_start_utc,
            Slot.start_utc < day_end_utc,
            Slot.status == "booked",
        )
    ).scalar_one()
    percentage = round((booked_today / total_today) * 100, 1) if total_today else 0.0

    # Per-doctor appointment counts for today, busiest first — powers the dashboard's
    # busiest-doctors pie chart. Only 'confirmed' (booked) and 'completed' count: a
    # cancelled appointment freed its slot back up and never happened, so it isn't
    # part of how busy a doctor actually was today. Joined against
    # Appointment.doctor_id/Doctor.id rather than Slot.doctor_id so a doctor only
    # shows up here for slots that actually have a live/completed appointment against
    # them today, not merely for having open slots today.
    busiest_doctors_rows = db.execute(
        select(Doctor.id, Doctor.full_name, func.count(Appointment.id))
        .select_from(Appointment)
        .join(Slot, Slot.id == Appointment.slot_id)
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .where(
            Appointment.clinic_id == clinic_id,
            Slot.start_utc >= day_start_utc,
            Slot.start_utc < day_end_utc,
            Appointment.status.in_(("confirmed", "completed")),
        )
        .group_by(Doctor.id, Doctor.full_name)
        .order_by(func.count(Appointment.id).desc())
    ).all()
    busiest_doctors_today = [
        DoctorAppointmentCountOut(doctor_id=str(doctor_id), doctor_name=doctor_name, count=count)
        for doctor_id, doctor_name, count in busiest_doctors_rows
    ]

    return AdminDashboardStatsOut(
        active_doctors_count=active_doctors_count,
        total_doctors_count=total_doctors_count,
        slot_utilization_today=SlotUtilizationOut(booked=booked_today, total=total_today, percentage=percentage),
        busiest_doctors_today=busiest_doctors_today,
    )


@router.get("/appointments-trend", response_model=list[DailyAppointmentCountOut])
@limiter.limit("60/minute")
def get_appointments_trend(
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> list[DailyAppointmentCountOut]:
    """Live booking-activity volume for the last 7 clinic-local calendar days,
    oldest to newest, ending today — powers the dashboard's weekly chart and its
    "booked today" headline (the last entry). Counts by created_at (when the
    booking was made), never by the appointment's own slot date — a patient
    booking today for a visit next week still counts against *today* here, not
    next week. Excludes cancelled appointments, so this is a live count of
    still-active bookings made that day: booking one bumps the count, and
    cancelling it later drops the count back down again, even if the
    cancellation happens on a later day than the booking did. One COUNT query
    per day (fixed at 7, never client-controlled) keeps each day's boundary as
    explicit and easy to verify as the slot-utilization boundary above.
    """
    clinic_id = current_user.clinic_id
    clinic = db.get(Clinic, clinic_id)
    tz = ZoneInfo(clinic.timezone)
    today_local = datetime.now(tz).date()

    trend: list[DailyAppointmentCountOut] = []
    for days_ago in range(6, -1, -1):
        day = today_local - timedelta(days=days_ago)
        day_start_utc = datetime.combine(day, time.min, tzinfo=tz).astimezone(timezone.utc)
        day_end_utc = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)

        count = db.execute(
            select(func.count())
            .select_from(Appointment)
            .where(
                Appointment.clinic_id == clinic_id,
                Appointment.created_at >= day_start_utc,
                Appointment.created_at < day_end_utc,
                Appointment.status != "cancelled",
            )
        ).scalar_one()
        trend.append(DailyAppointmentCountOut(date=day, count=count))

    return trend


@router.get("/top-rated-doctors", response_model=list[TopRatedDoctorOut])
@limiter.limit("60/minute")
def get_top_rated_doctors(
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> list[TopRatedDoctorOut]:
    """Top 3 active doctors by average AppointmentFeedback rating (ties broken by
    rating count, more ratings first) — always computed live from current rows,
    never cached, so it moves the moment a new rating comes in. A doctor with zero
    ratings can't appear at all (inner join on the rating subquery) — "top rated"
    inherently requires at least one real rating to rank by. `visit_count` is this
    doctor's all-time 'completed' appointment count (a visit that actually
    happened), independent of and not filtered by the rating subquery.
    """
    clinic_id = current_user.clinic_id

    rating_subq = (
        select(
            AppointmentFeedback.doctor_id.label("doctor_id"),
            func.avg(AppointmentFeedback.rating).label("avg_rating"),
            func.count(AppointmentFeedback.id).label("rating_count"),
        )
        .where(AppointmentFeedback.clinic_id == clinic_id)
        .group_by(AppointmentFeedback.doctor_id)
        .subquery()
    )
    visit_subq = (
        select(
            Appointment.doctor_id.label("doctor_id"),
            func.count(Appointment.id).label("visit_count"),
        )
        .where(Appointment.clinic_id == clinic_id, Appointment.status == "completed")
        .group_by(Appointment.doctor_id)
        .subquery()
    )

    rows = db.execute(
        select(
            Doctor.id,
            Doctor.full_name,
            Department.name,
            rating_subq.c.avg_rating,
            rating_subq.c.rating_count,
            func.coalesce(visit_subq.c.visit_count, 0),
        )
        .select_from(Doctor)
        .join(rating_subq, rating_subq.c.doctor_id == Doctor.id)
        .join(Department, Department.id == Doctor.department_id)
        .outerjoin(visit_subq, visit_subq.c.doctor_id == Doctor.id)
        .where(Doctor.clinic_id == clinic_id, Doctor.is_active.is_(True))
        .order_by(rating_subq.c.avg_rating.desc(), rating_subq.c.rating_count.desc())
        .limit(TOP_RATED_DOCTORS_LIMIT)
    ).all()

    return [
        TopRatedDoctorOut(
            doctor_id=str(doctor_id),
            doctor_name=doctor_name,
            department_name=department_name,
            average_rating=round(float(avg_rating), 1),
            rating_count=rating_count,
            visit_count=visit_count,
        )
        for doctor_id, doctor_name, department_name, avg_rating, rating_count, visit_count in rows
    ]


@router.get("/weekly-digest", response_model=AdminWeeklyDigestOut)
@limiter.limit("30/minute")
def get_weekly_digest(
    request: Request = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> AdminWeeklyDigestOut:
    """A short, LLM-generated plain-English summary of this clinic's own dashboard
    numbers (see app.services.admin_insights) — cached and refreshed at most once a
    week, so this is cheap to call on every dashboard visit. `digest` is None only
    when no digest has ever been successfully generated for this clinic (no LLM key
    configured, or every attempt so far has failed); the frontend hides the card
    entirely in that case rather than showing an error.

    Imported locally (not at module level) because app.services.admin_insights
    itself imports the three functions above from this module — a module-level
    import here would be circular.
    """
    from app.services.admin_insights import get_weekly_digest as compute_weekly_digest

    digest, generated_at = compute_weekly_digest(db, current_user.clinic_id, current_user)
    db.commit()
    return AdminWeeklyDigestOut(digest=digest, generated_at=generated_at)
