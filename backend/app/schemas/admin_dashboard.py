from datetime import date

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
