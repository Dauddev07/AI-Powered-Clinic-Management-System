import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { forgotPassword, resetPassword, verifyResetOtp } from "../api/auth";
import { ApiError } from "../api/client";
import PasswordInput from "../components/PasswordInput";
import { useReveal } from "../hooks/useReveal";
import styles from "./ForgotPassword.module.css";

// Matches the backend's PASSWORD_RESET_OTP_TTL_MINUTES / _RESEND_COOLDOWN_SECONDS
// (app/core/config.py) — purely cosmetic countdowns, the backend's own expiry/
// cooldown checks are the real enforcement either way.
const OTP_TTL_SECONDS = 5 * 60;
const RESEND_COOLDOWN_SECONDS = 60;
// How long the "code verified" checkmark stays on screen before advancing to the
// password step — long enough to register as a deliberate confirmation, short
// enough not to feel like a stall.
const VERIFIED_PAUSE_MS = 1100;

function formatCountdown(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function OtpInput({ value, onChange, disabled, verifying }) {
  const digits = value.split("");
  const inputRefs = useRef([]);

  const handleChange = (index) => (e) => {
    const raw = e.target.value.replace(/\D/g, "");
    if (!raw) {
      const next = value.split("");
      next[index] = "";
      onChange(next.join("").slice(0, 6));
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
    inputRefs.current[Math.min(cursor, 5)]?.focus();
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
          className={`${styles.otpDigit} ${verifying ? styles.otpDigitVerifying : ""}`}
          style={verifying ? { animationDelay: `${i * 90}ms` } : undefined}
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

  // email -> otp -> verified (brief, auto-advances) -> reset
  const [step, setStep] = useState("email");
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
    if (step !== "otp" || secondsLeft <= 0) return undefined;
    const id = setInterval(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearInterval(id);
  }, [step, secondsLeft]);

  useEffect(() => {
    if (step !== "otp" || resendCooldown <= 0) return undefined;
    const id = setInterval(() => setResendCooldown((s) => s - 1), 1000);
    return () => clearInterval(id);
  }, [step, resendCooldown]);

  // "verified" is a transitional beat, not a screen the patient acts on — it
  // always moves itself forward after a short pause.
  useEffect(() => {
    if (step !== "verified") return undefined;
    const id = setTimeout(() => setStep("reset"), VERIFIED_PAUSE_MS);
    return () => clearTimeout(id);
  }, [step]);

  const codeExpired = step === "otp" && secondsLeft <= 0;
  const canResend = step === "otp" && resendCooldown <= 0;

  const restart = () => {
    setStep("email");
    setOtp("");
    setNewPassword("");
    setConfirmPassword("");
    setNotice(null);
    setError(null);
  };

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

  const verifyCode = async (code) => {
    setError(null);
    setSubmitting(true);
    try {
      await verifyResetOtp(email, code);
      setStep("verified");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  // Auto-verifies the moment all 6 boxes are filled — no separate "Verify code"
  // button to press. Depends on the otp string itself, not just its length, so a
  // failed attempt (code unchanged) doesn't loop-retry; it only fires again once
  // the patient actually edits a digit, producing a new 6-character value.
  useEffect(() => {
    if (step !== "otp" || codeExpired || submitting || otp.length !== 6) return;
    verifyCode(otp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [otp, step, codeExpired]);

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError(null);

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
      // otp was already confirmed valid in the "verify" step — re-submitted here
      // (not re-entered by the patient) because the backend re-checks it before
      // ever touching the password, rather than trusting an earlier, separate call.
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
        {step === "email" && (
          <>
            <div className={styles.icon}>
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="3" y="5" width="18" height="14" rx="2.5" />
                <path d="m3.5 6 8 6.2L19.5 6" />
              </svg>
            </div>
            <h1 className={styles.title}>Forgot password</h1>
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

            <div className={styles.field}>
              <label htmlFor="otp-0">
                Verification code<span className={styles.requiredMark}>*</span>
              </label>
              <OtpInput
                value={otp}
                onChange={setOtp}
                disabled={codeExpired || submitting}
                verifying={submitting}
              />

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

              {submitting && (
                <div className={styles.verifyingRow}>
                  <span className={styles.verifyingCube} aria-hidden="true" />
                  <span className={styles.fieldHint}>Verifying…</span>
                </div>
              )}
            </div>

            <div className={styles.footer}>
              <button type="button" onClick={restart} className={styles.linkButton}>
                Use a different email
              </button>
            </div>
          </>
        )}

        {step === "verified" && (
          <div className={styles.verifiedWrap}>
            <div className={styles.verifiedCheck}>
              <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M5 12.5 9.5 17 19 7" />
              </svg>
            </div>
            <p className={styles.verifiedText}>Code verified</p>
          </div>
        )}

        {step === "reset" && (
          <>
            <div className={styles.icon}>
              <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="5" y="10.5" width="14" height="9" rx="2.2" />
                <path d="M8 10.5V7.5a4 4 0 0 1 8 0v3" />
                <circle cx="12" cy="15" r="1.4" />
              </svg>
            </div>
            <h1 className={styles.title}>Set a new password</h1>
            <p className={styles.subtitle}>Choose a new password for your account.</p>

            {error && (
              <div className={styles.error} role="alert">
                {error}
              </div>
            )}

            <form onSubmit={handleResetPassword}>
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
              <button className={styles.submit} type="submit" disabled={submitting}>
                {submitting ? "Resetting…" : "Reset password"}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
