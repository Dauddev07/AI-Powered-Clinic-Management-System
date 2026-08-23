import { useEffect, useState } from "react";
import { confirmDeleteAccount, requestDeleteAccountOtp } from "../api/auth";
import { ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import styles from "../pages/patient/PatientScreens.module.css";
import OtpVerificationBlock from "./OtpVerificationBlock";
import Modal from "./Modal";

// Mirrors the backend's ACCOUNT_DELETE_OTP_TTL_MINUTES / _RESEND_COOLDOWN_SECONDS
// (app/core/config.py) — purely cosmetic countdowns, same as ForgotPassword.jsx's own
// copy of this pattern; the backend's own expiry/cooldown checks are the real
// enforcement either way.
const OTP_TTL_SECONDS = 5 * 60;
const RESEND_COOLDOWN_SECONDS = 60;

// Reached only from SettingsMenu's patient-only "Delete account" row. Unlike
// ForgotPassword's OTP step, filling in the 6th digit here does NOT auto-submit —
// this is a permanent, irreversible action, so it always waits for an explicit
// "Delete my account" press rather than firing the moment the code is complete.
export default function DeleteAccountModal({ open, onClose }) {
  const { logout, setSessionMessage } = useAuth();
  const [step, setStep] = useState("warn");
  const [otp, setOtp] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(OTP_TTL_SECONDS);
  const [resendCooldown, setResendCooldown] = useState(RESEND_COOLDOWN_SECONDS);

  // Reset back to the warning step every time the modal is reopened, rather than
  // reopening wherever it was left mid-flow.
  useEffect(() => {
    if (!open) return;
    setStep("warn");
    setOtp("");
    setError(null);
    setSubmitting(false);
  }, [open]);

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

  const handleSendCode = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await requestDeleteAccountOtp();
      setOtp("");
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
      await requestDeleteAccountOtp();
      setOtp("");
      setSecondsLeft(OTP_TTL_SECONDS);
      setResendCooldown(RESEND_COOLDOWN_SECONDS);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleConfirmDelete = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await confirmDeleteAccount(otp);
      setSessionMessage("Your account has been deleted.");
      onClose();
      logout();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong.");
      setSubmitting(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Delete account">
      {step === "warn" && (
        <>
          <p className={styles.modalIntro}>
            This permanently deletes your account, appointments, and chat history. This
            cannot be undone. We&apos;ll email a verification code to confirm it&apos;s you.
          </p>
          {error && <p className={styles.errorText}>{error}</p>}
          <div className={styles.modalActions}>
            <button type="button" className={styles.modalCancelBtn} onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button type="button" className={styles.modalDangerBtn} onClick={handleSendCode} disabled={submitting}>
              {submitting ? "Sending…" : "Send verification code"}
            </button>
          </div>
        </>
      )}

      {step === "otp" && (
        <>
          <p className={styles.modalIntro}>We sent a 6-digit code to your registered email.</p>
          {error && <p className={styles.errorText}>{error}</p>}

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

          <div className={styles.modalActions}>
            <button type="button" className={styles.modalCancelBtn} onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button
              type="button"
              className={styles.modalDangerBtn}
              onClick={handleConfirmDelete}
              disabled={submitting || codeExpired || otp.length !== 6}
            >
              {submitting ? "Deleting…" : "Delete my account"}
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}
