import { apiFetch } from "./client";

export function fetchAdminDashboardStats() {
  return apiFetch("/admin/dashboard/stats", { auth: true });
}

export function fetchAppointmentsTrend() {
  return apiFetch("/admin/dashboard/appointments-trend", { auth: true });
}
