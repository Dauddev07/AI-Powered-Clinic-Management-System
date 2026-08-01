export function decodeJwtPayload(token) {
  try {
    const [, payloadB64] = token.split(".");
    const base64 = payloadB64.replace(/-/g, "+").replace(/_/g, "/");
    const json = decodeURIComponent(
      atob(base64)
        .split("")
        .map((c) => "%" + c.charCodeAt(0).toString(16).padStart(2, "0"))
        .join("")
    );
    return JSON.parse(json);
  } catch {
    return null;
  }
}

export function isExpired(payload) {
  if (!payload || !payload.exp) return true;
  return Date.now() >= payload.exp * 1000;
}
