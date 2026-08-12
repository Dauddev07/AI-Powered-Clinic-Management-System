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

// No auth: the VAPID public key isn't a secret (see the backend endpoint's own
// docstring) — the app needs it before it knows whether push is even configured.
export function fetchPushPublicKey() {
  return apiFetch("/notifications/push-public-key");
}

// `subscription` is the raw object returned by PushSubscription.toJSON() — see
// pushNotifications.js, which forwards it here with no reshaping.
export function subscribeToPush(subscription) {
  return apiFetch("/notifications/push-subscribe", { method: "POST", auth: true, body: subscription });
}

export function unsubscribeFromPush(endpoint) {
  return apiFetch("/notifications/push-unsubscribe", { method: "POST", auth: true, body: { endpoint } });
}
