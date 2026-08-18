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

function clearTokens() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

// Access tokens are short-lived now (see backend JWT_ACCESS_EXPIRE_MINUTES) — a 401
// from an expired-but-otherwise-valid session is the common case, not the
// exception, so it's worth one silent refresh-and-retry before treating it as a
// real logout. Concurrent 401s (several requests in flight when the token expires)
// must share a single in-flight refresh rather than each firing their own — a
// second call would try to rotate an already-just-rotated refresh token and fail
// (see rotate_refresh_token's docstring on the backend), logging the patient out
// for no reason.
let refreshInFlight = null;

export async function refreshAccessToken() {
  if (refreshInFlight) return refreshInFlight;

  const refreshToken = localStorage.getItem("refresh_token");
  if (!refreshToken) return null;

  refreshInFlight = (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!res.ok) {
        clearTokens();
        return null;
      }
      const data = await res.json();
      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("refresh_token", data.refresh_token);
      return data.access_token;
    } catch {
      return null;
    } finally {
      refreshInFlight = null;
    }
  })();

  return refreshInFlight;
}

export async function apiFetch(path, { method = "GET", body, auth = false, formData, _isRetry = false } = {}) {
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

  if (res.status === 401 && auth && !_isRetry) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      return apiFetch(path, { method, body, auth, formData, _isRetry: true });
    }
    onUnauthorized();
    throw new ApiError(401, "Invalid or expired token");
  }

  if (res.status === 401 && auth && _isRetry) {
    clearTokens();
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
