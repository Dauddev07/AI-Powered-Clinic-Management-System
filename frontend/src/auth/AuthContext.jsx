import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as authApi from "../api/auth";
import { setUnauthorizedHandler } from "../api/client";
import { decodeJwtPayload, isExpired } from "./jwt";

const AuthContext = createContext(null);

function loadUserFromStorage() {
  const token = localStorage.getItem("access_token");
  if (!token) return { token: null, user: null };

  const payload = decodeJwtPayload(token);
  if (!payload || isExpired(payload)) {
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
  const navigate = useNavigate();

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setAuthState({ token: null, user: null });
      setSessionMessage("Your session has expired. Please log in again.");
      navigate("/login");
    });
  }, [navigate]);

  const login = async (email, password) => {
    const data = await authApi.login(email, password);
    localStorage.setItem("access_token", data.access_token);
    const payload = decodeJwtPayload(data.access_token);
    setAuthState({
      token: data.access_token,
      user: { id: payload.sub, clinicId: payload.clinic_id, role: payload.role },
    });
    setSessionMessage(null);
    return data;
  };

  const logout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("chat_session_id");
    setAuthState({ token: null, user: null });
    navigate("/login");
  };

  const value = useMemo(
    () => ({ token, user, isAuthenticated: !!token, login, logout, sessionMessage, setSessionMessage }),
    [token, user, sessionMessage]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
