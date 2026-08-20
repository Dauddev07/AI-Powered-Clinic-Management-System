import { apiFetch } from "./client";

export function fetchFeedback({ limit = 20, offset = 0, tone } = {}) {
  const params = new URLSearchParams({ limit, offset });
  if (tone) params.set("tone", tone);
  return apiFetch(`/admin/feedback?${params.toString()}`, { auth: true });
}
