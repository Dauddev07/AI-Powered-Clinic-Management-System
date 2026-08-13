import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { resendVerificationEmail, verifyEmail } from "../api/auth";
import { ApiError } from "../api/client";
import OtpVerificationBlock from "../components/OtpVerificationBlock";
import { useAuth } from "../auth/AuthContext";
import { decodeJwtPayload } from "../auth/jwt";
import { useReveal } from "../hooks/useReveal";
import styles from "./ForgotPassword.module.css";

// Matches the backend's EMAIL_VERIFICATION_OTP_TTL_MINUTES / _RESEND_COOLDOWN_SECONDS
// (app/core/config.py) — purely cosmetic countdowns, the backend's own expiry/
// cooldown checks are the real enforcement either way.
const OTP_TTL_SECONDS = 5 * 60;
const RESEND_COOLDOWN_SECONDS = 60;

export default function VerifyEmail() {
  const navigate = useNavigate();
  const location = useLocation();
  const { loginWithToken } = useAuth();
  const revealRef = useReveal();

  // Registration (see Register.jsx) already triggers the first code as part of
  // creating the account, so arriving here with an email already known skips
  // straight to entering it — asking again / re-sending here too would just
  // trip the backend's own resend cooldown for no reason. Only a direct,
  // state-less visit to this URL needs the plain email-entry step at all.
  const knownEmail = location.state?.email?.trim() || "";
  const [step, setStep] = useState(knownEmail ? "otp" : "email");
  const [email, setEmail] = useState(knownEmail);
  const [otp, setOtp] = useState("");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(OTP_TTL_SECONDS);
  const [resendCooldown, setResendCooldown] = useState(RESEND_COOLDOWN_SECONDS);

  useEffect(() => {
    if (step !== "otp" || secondsLeft <= 0) return undefined;
    const id = setInterval(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearInterval(id);
  }, [step, secondsLeft]);

  useEffect(() => {
    if (step !== "otp" || resendCooldown <= 0) return undefined;
    const id = setInterval(() => setResendCooldown((s) => s - 1), 1000);
    return () => clearInterval(id);
  }, [step, resendCooldown]);

  const codeExpired = step === "otp" && secondsLeft <= 0;
  const canResend = step === "otp" && resendCooldown <= 0;

  const restart = () => {
    setStep("email");
    setOtp("");
    setNotice(null);
    setError(null);
  };

  const handleRequestCode = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await resendVerificationEmail(email);
      // Same generic wording the backend itself uses — never confirms or denies
      // that this email has an unverified account (see app/api/auth.py).
      setNotice("If an unverified account exists for this email, a verification code has been sent.");
      setSecondsLeft(OTP_TTL_SECONDS);
      setResendCooldown(RESEND_COOLDOWN_SECONDS);
      setStep("otp");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleResendCode = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await resendVerificationEmail(email);
      setNotice("A new code has been sent.");
      setOtp("");
      setSecondsLeft(OTP_TTL_SECONDS);
      setResendCooldown(RESEND_COOLDOWN_SECONDS);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  const verifyCode = async (code) => {
    setError(null);
    setSubmitting(true);
    try {
      const data = await verifyEmail(email, code);
      loginWithToken(data);
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

  // Auto-verifies the moment all 6 boxes are filled — no separate "Verify" button.
  // Depends on the otp string itself, not just its length, so a failed attempt
  // (code unchanged) doesn't loop-retry; it only fires again once the patient
  // actually edits a digit, producing a new 6-character value.
  useEffect(() => {
    if (step !== "otp" || codeExpired || submitting || otp.length !== 6) return;
    verifyCode(otp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [otp, step, codeExpired]);

  return (
    <div className={styles.page}>
      <div className={`${styles.card} reveal`} ref={revealRef}>
        {step === "email" && (
          <>
            <div className={styles.icon}>
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="3" y="5" width="18" height="14" rx="2.5" />
                <path d="m3.5 6 8 6.2L19.5 6" />
              </svg>
            </div>
            <h1 className={styles.title}>Verify your email</h1>
            <p className={styles.subtitle}>
              Enter your account email and we&apos;ll send a 6-digit verification code to it.
            </p>

            {error && (
              <div className={styles.error} role="alert">
                {error}
              </div>
            )}

            <form onSubmit={handleRequestCode}>
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
                  autoComplete="username"
                  placeholder="you@example.com"
                />
              </div>
              <button className={styles.submit} type="submit" disabled={submitting}>
                {submitting ? "Sending…" : "Send code"}
              </button>
            </form>

            <div className={styles.footer}>
              <Link to="/login">Back to log in</Link>
            </div>
          </>
        )}

        {step === "otp" && (
          <>
            <div className={styles.icon}>
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="4" y="10" width="16" height="10" rx="2.5" />
                <path d="M8 10V7a4 4 0 0 1 8 0v3" />
              </svg>
            </div>
            <h1 className={styles.title}>Enter your code</h1>
            <p className={styles.subtitle}>
              We sent a 6-digit code to <strong>{email}</strong>
            </p>

            {notice && (
              <div className={styles.success} role="status">
                {notice}
              </div>
            )}
            {error && (
              <div className={styles.error} role="alert">
                {error}
              </div>
            )}

            <OtpVerificationBlock
              otp={otp}
              onOtpChange={setOtp}
              secondsLeft={secondsLeft}
              resendCooldown={resendCooldown}
              codeExpired={codeExpired}
              canResend={canResend}
              submitting={submitting}
              onResend={handleResendCode}
            />

            <div className={styles.footer}>
              <button type="button" onClick={restart} className={styles.linkButton}>
                Use a different email
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
