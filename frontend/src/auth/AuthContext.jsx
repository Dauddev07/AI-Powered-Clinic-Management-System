import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as authApi from "../api/auth";
import { refreshAccessToken, setUnauthorizedHandler } from "../api/client";
import { decodeJwtPayload, isExpired } from "./jwt";

const AuthContext = createContext(null);

function loadUserFromStorage() {
  const token = localStorage.getItem("access_token");
  if (!token) return { token: null, user: null };

  const payload = decodeJwtPayload(token);
  if (!payload || isExpired(payload)) {
    // Don't clear refresh_token here — an expired access token with a still-valid
    // refresh token is the ordinary case now (access tokens live for only
    // JWT_ACCESS_EXPIRE_MINUTES; a page reload after that window is routine, not a
    // real logout). AuthProvider's mount effect below tries a silent refresh
    // before giving up.
    localStorage.removeItem("access_token");
    return { token: null, user: null };
  }

  return {
    token,
    user: { id: payload.sub, clinicId: payload.clinic_id, role: payload.role },
  };
}

export function AuthProvider({ children }) {
  const [{ token, user }, setAuthState] = useState(loadUserFromStorage);
  const [sessionMessage, setSessionMessage] = useState(null);
  // True only while a page-load silent refresh (see the mount effect below) is
  // still in flight — RequireAuth holds off redirecting to /login until this
  // settles, so a merely-expired-but-refreshable access token doesn't bounce the
  // patient out of a page they're still validly logged into.
  const [isBootstrapping, setIsBootstrapping] = useState(() => !token && !!localStorage.getItem("refresh_token"));
  const navigate = useNavigate();

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setAuthState({ token: null, user: null });
      setSessionMessage("Your session has expired. Please log in again.");
      navigate("/login");
    });
  }, [navigate]);

  useEffect(() => {
    if (!isBootstrapping) return;
    refreshAccessToken().then((newToken) => {
      if (newToken) {
        const payload = decodeJwtPayload(newToken);
        setAuthState({
          token: newToken,
          user: { id: payload.sub, clinicId: payload.clinic_id, role: payload.role },
        });
      }
      setIsBootstrapping(false);
    });
    // Runs once, on mount, only when there was a refresh token to try.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Shared by login() below and VerifyEmail.jsx's post-verification auto-login —
  // both end up with the exact same TokenResponse shape ({access_token,
  // refresh_token, must_change_password}) from the backend, just via a different
  // endpoint.
  const applyAuthResponse = (data) => {
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    const payload = decodeJwtPayload(data.access_token);
    setAuthState({
      token: data.access_token,
      user: { id: payload.sub, clinicId: payload.clinic_id, role: payload.role },
    });
    setSessionMessage(null);
    return data;
  };

  const login = async (email, password) => {
    const data = await authApi.login(email, password);
    return applyAuthResponse(data);
  };

  // Takes an already-fetched TokenResponse (e.g. from POST /auth/verify-email,
  // which logs the patient in directly on success) rather than calling the login
  // endpoint itself.
  const loginWithToken = (data) => applyAuthResponse(data);

  const logout = () => {
    // Best-effort — revokes this one device's refresh token server-side so it
    // can't be replayed later, but the local logout below happens regardless of
    // whether this call succeeds (e.g. the network is down, or the token was
    // already expired/revoked).
    const refreshToken = localStorage.getItem("refresh_token");
    if (refreshToken) {
      authApi.logout(refreshToken).catch(() => {});
    }
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("chat_session_id");
    setAuthState({ token: null, user: null });
    navigate("/login");
  };

  const value = useMemo(
    () => ({
      token,
      user,
      isAuthenticated: !!token,
      isBootstrapping,
      login,
      loginWithToken,
      logout,
      sessionMessage,
      setSessionMessage,
    }),
    [token, user, sessionMessage, isBootstrapping]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
