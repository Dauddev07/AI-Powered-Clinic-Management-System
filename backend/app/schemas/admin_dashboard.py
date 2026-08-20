from datetime import date, datetime

from pydantic import BaseModel


class SlotUtilizationOut(BaseModel):
    booked: int
    total: int
    # 0.0 when total is 0 (no slots generated for today at all), never a
    # division-by-zero — the frontend can treat 0/0 and a real 0% the same way.
    percentage: float


class DoctorAppointmentCountOut(BaseModel):
    doctor_id: str
    doctor_name: str
    count: int


class AdminDashboardStatsOut(BaseModel):
    active_doctors_count: int
    total_doctors_count: int
    slot_utilization_today: SlotUtilizationOut
    # Doctors with at least one appointment today (any status), busiest first —
    # powers the admin dashboard's busiest-doctors pie chart. Empty when nothing's
    # booked for today yet.
    busiest_doctors_today: list[DoctorAppointmentCountOut]


class DailyAppointmentCountOut(BaseModel):
    date: date
    count: int


class TopRatedDoctorOut(BaseModel):
    doctor_id: str
    doctor_name: str
    department_name: str
    # Mean of AppointmentFeedback.rating (1-5), rounded to 1 decimal — only
    # doctors with at least one rating can appear here at all (see the query in
    # admin_dashboard.py), so this is never null.
    average_rating: float
    rating_count: int
    # All-time count of this doctor's 'completed' appointments — a real visit
    # that actually happened, not merely booked.
    visit_count: int


class AdminWeeklyDigestOut(BaseModel):
    # None only when nothing has ever been generated for this clinic (no LLM key
    # configured, or every generation attempt so far has failed) — the frontend
    # hides the digest card entirely in that case rather than showing an error.
    digest: str | None
    generated_at: datetime | None
