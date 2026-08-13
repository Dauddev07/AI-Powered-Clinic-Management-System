import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { forgotPassword, resetPassword } from "../api/auth";
import { ApiError } from "../api/client";
import PasswordInput from "../components/PasswordInput";
import { useReveal } from "../hooks/useReveal";
import styles from "./ForgotPassword.module.css";

// Matches the backend's PASSWORD_RESET_OTP_TTL_MINUTES / _RESEND_COOLDOWN_SECONDS
// (app/core/config.py) — purely cosmetic countdowns, the backend's own expiry/
// cooldown checks are the real enforcement either way.
const OTP_TTL_SECONDS = 5 * 60;
const RESEND_COOLDOWN_SECONDS = 60;

function formatCountdown(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function OtpInput({ value, onChange, disabled }) {
  const digits = value.split("");
  const inputRefs = useRef([]);

  const setDigit = (index, char) => {
    const next = value.split("");
    next[index] = char;
    onChange(next.join("").slice(0, 6));
  };

  const handleChange = (index) => (e) => {
    const raw = e.target.value.replace(/\D/g, "");
    if (!raw) {
      setDigit(index, "");
      return;
    }
    // Handles a full paste landing in one box (raw can be multiple digits) as well
    // as a single keystroke — either way, spill remaining digits into the boxes
    // that follow and land focus just past the last one filled.
    const chars = raw.split("");
    const next = value.split("");
    let cursor = index;
    for (const char of chars) {
      if (cursor > 5) break;
      next[cursor] = char;
      cursor += 1;
    }
    onChange(next.join("").slice(0, 6));
    const focusIndex = Math.min(cursor, 5);
    inputRefs.current[focusIndex]?.focus();
  };

  const handleKeyDown = (index) => (e) => {
    if (e.key === "Backspace" && !digits[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  return (
    <div className={styles.otpRow}>
      {Array.from({ length: 6 }).map((_, i) => (
        <input
          key={i}
          ref={(el) => (inputRefs.current[i] = el)}
          className={styles.otpDigit}
          type="text"
          inputMode="numeric"
          maxLength={6}
          disabled={disabled}
          value={digits[i] || ""}
          onChange={handleChange(i)}
          onKeyDown={handleKeyDown(i)}
          autoComplete={i === 0 ? "one-time-code" : "off"}
          aria-label={`Digit ${i + 1} of verification code`}
        />
      ))}
    </div>
  );
}

export default function ForgotPassword() {
  const navigate = useNavigate();
  const revealRef = useReveal();

  // "request" — just an email field, sends the code.
  // "reset" — code + new password fields, shown once a code has been sent.
  const [step, setStep] = useState("request");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(OTP_TTL_SECONDS);
  const [resendCooldown, setResendCooldown] = useState(RESEND_COOLDOWN_SECONDS);

  useEffect(() => {
    if (step !== "reset" || secondsLeft <= 0) return undefined;
    const id = setInterval(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearInterval(id);
  }, [step, secondsLeft]);

  useEffect(() => {
    if (step !== "reset" || resendCooldown <= 0) return undefined;
    const id = setInterval(() => setResendCooldown((s) => s - 1), 1000);
    return () => clearInterval(id);
  }, [step, resendCooldown]);

  const codeExpired = step === "reset" && secondsLeft <= 0;
  const canResend = step === "reset" && resendCooldown <= 0;

  const handleRequestCode = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await forgotPassword(email);
      // Same generic wording the backend itself uses — never confirms or denies
      // that this email actually has an account (see app/api/auth.py).
      setNotice("If an account exists for this email, a verification code has been sent.");
      setSecondsLeft(OTP_TTL_SECONDS);
      setResendCooldown(RESEND_COOLDOWN_SECONDS);
      setStep("reset");
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
      await forgotPassword(email);
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

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError(null);

    if (codeExpired) {
      setError("This code has expired. Request a new one.");
      return;
    }
    if (otp.length < 6) {
      setError("Enter the full 6-digit code.");
      return;
    }
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("New password and confirmation do not match.");
      return;
    }

    setSubmitting(true);
    try {
      await resetPassword(email, otp, newPassword);
      navigate("/login", { replace: true, state: { resetSuccess: true } });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={`${styles.card} reveal`} ref={revealRef}>
        {step === "request" ? (
          <>
            <div className={styles.icon}>
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="3" y="5" width="18" height="14" rx="2.5" />
                <path d="m3.5 6 8 6.2L19.5 6" />
              </svg>
            </div>
            <h1 className={styles.title}>Forgot password</h1>
            <p className={styles.subtitle}>
              Enter your account email and we&apos;ll send you a 6-digit verification code.
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
          </>
        ) : (
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

            {codeExpired && (
              <div className={styles.expiredBanner} role="alert">
                <span>This code has expired.</span>
                <button
                  type="button"
                  onClick={handleResendCode}
                  className={styles.linkButton}
                  disabled={submitting}
                >
                  Send a new code
                </button>
              </div>
            )}

            <form onSubmit={handleResetPassword}>
              <div className={styles.field}>
                <label htmlFor="otp-0">
                  Verification code<span className={styles.requiredMark}>*</span>
                </label>
                <OtpInput value={otp} onChange={setOtp} disabled={codeExpired} />

                <div className={styles.otpMeta}>
                  {!codeExpired && (
                    <span
                      className={`${styles.expiryPill} ${secondsLeft <= 60 ? styles.expiryPillUrgent : ""}`}
                    >
                      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <circle cx="12" cy="12" r="9" />
                        <path d="M12 7v5l3.5 2" />
                      </svg>
                      Expires in {formatCountdown(secondsLeft)}
                    </span>
                  )}
                  <span className={styles.resendText}>
                    {canResend ? (
                      <button
                        type="button"
                        onClick={handleResendCode}
                        className={styles.linkButton}
                        disabled={submitting}
                      >
                        Resend code
                      </button>
                    ) : (
                      <>Resend code in {formatCountdown(resendCooldown)}</>
                    )}
                  </span>
                </div>
              </div>

              <div className={styles.divider} />

              <div className={styles.field}>
                <label htmlFor="newPassword">
                  New password<span className={styles.requiredMark}>*</span>
                </label>
                <PasswordInput
                  id="newPassword"
                  required
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  autoComplete="new-password"
                />
                <span className={styles.fieldHint}>At least 8 characters.</span>
              </div>
              <div className={styles.field}>
                <label htmlFor="confirmPassword">
                  Confirm new password<span className={styles.requiredMark}>*</span>
                </label>
                <PasswordInput
                  id="confirmPassword"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  autoComplete="new-password"
                />
              </div>
              <button className={styles.submit} type="submit" disabled={submitting || codeExpired}>
                {submitting ? "Resetting…" : "Reset password"}
              </button>
            </form>

            <div className={styles.footer}>
              <button
                type="button"
                onClick={() => {
                  setStep("request");
                  setNotice(null);
                  setError(null);
                  setOtp("");
                }}
                className={styles.linkButton}
              >
                Use a different email
              </button>
            </div>
          </>
        )}

        <div className={styles.footer}>
          <Link to="/login">Back to log in</Link>
        </div>
      </div>
    </div>
  );
}
