import { apiFetch } from "./client";

export function fetchMyNotifications() {
  return apiFetch("/notifications", { auth: true });
}

export function markNotificationRead(notificationId) {
  return apiFetch(`/notifications/${notificationId}/read`, { method: "PATCH", auth: true });
}

export function markAllNotificationsRead() {
  return apiFetch("/notifications/mark-all-read", { method: "PATCH", auth: true });
}
