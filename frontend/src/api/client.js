const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed with status ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

let onUnauthorized = () => {};

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

export async function apiFetch(path, { method = "GET", body, auth = false, formData } = {}) {
  const headers = {};

  if (auth) {
    const token = localStorage.getItem("access_token");
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  let requestBody;
  if (formData) {
    // Let the browser set Content-Type (with the multipart boundary) itself.
    requestBody = formData;
  } else {
    headers["Content-Type"] = "application/json";
    requestBody = body !== undefined ? JSON.stringify(body) : undefined;
  }

  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers,
    body: requestBody,
  });

  if (res.status === 401 && auth) {
    localStorage.removeItem("access_token");
    onUnauthorized();
  }

  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }

  if (!res.ok) {
    const detail = data && typeof data.detail === "string" ? data.detail : null;
    throw new ApiError(res.status, detail);
  }

  return data;
}
