import { apiFetch } from "./client";

// coords (optional) is { lat, lng } — the patient's device location, only ever used
// server-side to attach a "nearest emergency hospitals" block when this particular
// message turns out to be classified as an emergency (see
// app.services.nearby_hospitals). Harmless to send on every message: the backend
// silently ignores it otherwise.
export function sendChatMessage(message, sessionId, coords) {
  return apiFetch("/chat", {
    method: "POST",
    auth: true,
    body: {
      message,
      session_id: sessionId || undefined,
      lat: coords?.lat ?? undefined,
      lng: coords?.lng ?? undefined,
    },
  });
}

export function fetchChatHistory(sessionId) {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  return apiFetch(`/chat/history${query}`, { auth: true });
}

export function fetchChatSessions() {
  return apiFetch("/chat/sessions", { auth: true });
}

export function deleteChatSession(sessionId) {
  return apiFetch(`/chat/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE", auth: true });
}

export function fetchPendingFeedback() {
  return apiFetch("/chat/pending-feedback", { auth: true });
}

export function submitFeedback(appointmentIds, rating, reason) {
  return apiFetch("/chat/feedback", {
    method: "POST",
    auth: true,
    body: { appointment_ids: appointmentIds, rating, reason: reason || undefined },
  });
}
