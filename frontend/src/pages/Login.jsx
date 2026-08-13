import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { ApiError } from "../api/client";
import { decodeJwtPayload } from "../auth/jwt";
import Logo from "../components/Logo";
import PasswordInput from "../components/PasswordInput";
import { useReveal } from "../hooks/useReveal";
import styles from "./Login.module.css";

export default function Login() {
  const { login, sessionMessage, setSessionMessage } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const register = useReveal();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // Set by ForgotPassword's navigate() after a successful reset — a one-time
  // router-state flag rather than sessionMessage, so it can't survive a refresh
  // and get shown again out of context.
  useEffect(() => {
    if (location.state?.resetSuccess) {
      setSessionMessage("Password reset — please log in with your new password.");
    }
  }, [location.state, setSessionMessage]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const data = await login(email, password);
      const payload = decodeJwtPayload(data.access_token);
      const destination = data.must_change_password
        ? "/change-password"
        : payload?.role === "admin"
          ? "/admin"
          : "/patient";
      navigate(destination, { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={`${styles.card} reveal`} ref={register}>
        <div className={styles.intro}>
          <Logo wrap />
          <h2 className={styles.introTitle}>Welcome back</h2>
          <p className={styles.introText}>
            Log in to book appointments, message our assistant, and manage your visits online.
          </p>
        </div>

        <div className={styles.formSide}>
          <h1 className={styles.title}>Log in</h1>
          <p className={styles.subtitle}>Access your clinic account</p>

          {sessionMessage && (
            <div className={styles.error} role="alert">
              {sessionMessage}
            </div>
          )}
          {error && (
            <div className={styles.error} role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className={styles.field}>
              <label htmlFor="email">
                Email<span className={styles.requiredMark}>*</span>
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                onFocus={() => setSessionMessage(null)}
                autoComplete="username"
              />
            </div>
            <div className={styles.field}>
              <label htmlFor="password">
                Password<span className={styles.requiredMark}>*</span>
              </label>
              <PasswordInput
                id="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
              <span className={styles.fieldHint}>
                <Link to="/forgot-password">Forgot password?</Link>
              </span>
            </div>
            <button className={styles.submit} type="submit" disabled={submitting}>
              {submitting ? "Logging in…" : "Log in"}
            </button>
          </form>

          <div className={styles.footer}>
            New patient? <Link to="/register">Register here</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
