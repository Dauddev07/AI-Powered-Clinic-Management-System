import { apiFetch } from "./client";

export function fetchDepartmentsWithSlots() {
  return apiFetch("/slots/departments", { auth: true });
}

export function fetchDoctorsWithSlots(departmentId, { dateFrom, dateTo } = {}) {
  const params = new URLSearchParams();
  if (departmentId) params.set("department_id", departmentId);
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  const query = params.toString();
  return apiFetch(`/slots/doctors${query ? `?${query}` : ""}`, { auth: true });
}

export function fetchSlots(departmentId, { doctorId, dateFrom, dateTo, timeOfDay, limit, offset } = {}) {
  const params = new URLSearchParams();
  if (departmentId) params.set("department_id", departmentId);
  if (doctorId) params.set("doctor_id", doctorId);
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  if (timeOfDay) params.set("time_of_day", timeOfDay);
  if (limit != null) params.set("limit", limit);
  if (offset != null) params.set("offset", offset);
  const query = params.toString();
  return apiFetch(`/slots${query ? `?${query}` : ""}`, { auth: true });
}

export function bookAppointment(slotId, reason) {
  return apiFetch("/appointments", {
    method: "POST",
    auth: true,
    body: { slot_id: slotId, reason: reason || null },
  });
}

export function fetchMyAppointments() {
  return apiFetch("/appointments", { auth: true });
}

export function fetchMyAppointmentSummary() {
  return apiFetch("/appointments/summary", { auth: true });
}

export function fetchMyAppointmentHistory() {
  return apiFetch("/appointments/history", { auth: true });
}

export function cancelAppointment(appointmentId) {
  return apiFetch(`/appointments/${appointmentId}/cancel`, { method: "POST", auth: true });
}

export function rescheduleAppointment(appointmentId, newSlotId) {
  return apiFetch(`/appointments/${appointmentId}/reschedule`, {
    method: "POST",
    auth: true,
    body: { new_slot_id: newSlotId },
  });
}
