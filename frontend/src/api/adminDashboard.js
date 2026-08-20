import { apiFetch } from "./client";

export function fetchAdminDashboardStats() {
  return apiFetch("/admin/dashboard/stats", { auth: true });
}

export function fetchAppointmentsTrend() {
  return apiFetch("/admin/dashboard/appointments-trend", { auth: true });
}

export function fetchTopRatedDoctors() {
  return apiFetch("/admin/dashboard/top-rated-doctors", { auth: true });
}

export function fetchWeeklyDigest() {
  return apiFetch("/admin/dashboard/weekly-digest", { auth: true });
}
