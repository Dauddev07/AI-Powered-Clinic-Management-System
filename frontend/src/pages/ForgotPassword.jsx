import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { forgotPassword, resetPassword } from "../api/auth";
import { ApiError } from "../api/client";
import PasswordInput from "../components/PasswordInput";
import { useReveal } from "../hooks/useReveal";
import styles from "./Login.module.css";

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

  const handleRequestCode = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await forgotPassword(email);
      // Same generic wording the backend itself uses — never confirms or denies
      // that this email actually has an account (see app/api/auth.py).
      setNotice("If an account exists for this email, a verification code has been sent.");
      setStep("reset");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

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
                />
              </div>
              <button className={styles.submit} type="submit" disabled={submitting}>
                {submitting ? "Sending…" : "Send code"}
              </button>
            </form>
          </>
        ) : (
          <>
            <h1 className={styles.title}>Enter your code</h1>
            <p className={styles.subtitle}>
              Check {email} for a 6-digit code. It expires in 5 minutes.
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

            <form onSubmit={handleResetPassword}>
              <div className={styles.field}>
                <label htmlFor="otp">
                  Verification code<span className={styles.requiredMark}>*</span>
                </label>
                <input
                  id="otp"
                  type="text"
                  inputMode="numeric"
                  pattern="\d{6}"
                  maxLength={6}
                  required
                  value={otp}
                  onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  autoComplete="one-time-code"
                />
              </div>
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
