import { apiFetch } from "./client";

export function previewDoctorCsv(file, headerMapping) {
  const formData = new FormData();
  formData.append("file", file);
  if (headerMapping && Object.keys(headerMapping).length > 0) {
    formData.append("header_mapping", JSON.stringify(headerMapping));
  }
  return apiFetch("/admin/doctors/csv/preview", { method: "POST", auth: true, formData });
}

export function confirmDoctorCsv(file, headerMapping) {
  const formData = new FormData();
  formData.append("file", file);
  if (headerMapping && Object.keys(headerMapping).length > 0) {
    formData.append("header_mapping", JSON.stringify(headerMapping));
  }
  return apiFetch("/admin/doctors/csv/confirm", { method: "POST", auth: true, formData });
}

export function fetchIngestionLogs({ limit, offset } = {}) {
  const params = new URLSearchParams();
  if (limit != null) params.set("limit", limit);
  if (offset != null) params.set("offset", offset);
  const query = params.toString();
  return apiFetch(`/admin/doctors/csv/ingestion-logs${query ? `?${query}` : ""}`, { auth: true });
}

export function fetchDoctors({ limit, offset } = {}) {
  const params = new URLSearchParams();
  if (limit != null) params.set("limit", limit);
  if (offset != null) params.set("offset", offset);
  const query = params.toString();
  return apiFetch(`/admin/doctors${query ? `?${query}` : ""}`, { auth: true });
}

export function updateDoctorStatus(doctorId, isActive) {
  return apiFetch(`/admin/doctors/${doctorId}/status`, {
    method: "PATCH",
    auth: true,
    body: { is_active: isActive },
  });
}
