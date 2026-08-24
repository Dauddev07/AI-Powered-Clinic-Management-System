import { useState } from "react";
import { changePassword } from "../api/auth";
import { ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import PasswordInput from "../components/PasswordInput";
import PasswordRequirements from "../components/PasswordRequirements";
import { useReveal } from "../hooks/useReveal";
import styles from "./Login.module.css";

export default function ChangePassword() {
  const { logout, setSessionMessage } = useAuth();
  const revealRef = useReveal();
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
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
    if (newPassword === oldPassword) {
      setError("New password must be different from your current password.");
      return;
    }

    setSubmitting(true);
    try {
      await changePassword(oldPassword, newPassword);
      // The backend invalidates the current token the moment the password changes,
      // so the stored token is already dead — clear it and force a fresh login rather
      // than letting the user believe they're still signed in.
      setSessionMessage("Password changed — please log in again.");
      logout();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={`${styles.card} reveal`} ref={revealRef}>
        <h1 className={styles.title}>Change password</h1>
        <p className={styles.subtitle}>Update your account password</p>

        {error && (
          <div className={styles.error} role="alert">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className={styles.field}>
            <label htmlFor="oldPassword">
              Current password<span className={styles.requiredMark}>*</span>
            </label>
            <PasswordInput
              id="oldPassword"
              required
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
              autoComplete="current-password"
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
            <PasswordRequirements password={newPassword} />
            <span className={styles.fieldHint}>Different from your current password.</span>
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
            {submitting ? "Updating…" : "Update password"}
          </button>
        </form>

        <div className={styles.footer}>
          <button type="button" onClick={logout} className={styles.linkButton}>
            Log out
          </button>
        </div>
      </div>
    </div>
  );
}
