import { apiFetch } from "./client";

export function sendChatMessage(message, sessionId) {
  return apiFetch("/chat", {
    method: "POST",
    auth: true,
    body: { message, session_id: sessionId || undefined },
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
