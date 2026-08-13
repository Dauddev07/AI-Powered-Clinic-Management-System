import { useRef } from "react";
import styles from "./OtpVerificationBlock.module.css";

function formatCountdown(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function OtpDigits({ value, onChange, disabled, verifying }) {
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
          autoComplete="off"
          aria-label={`Digit ${i + 1} of verification code`}
        />
      ))}
    </div>
  );
}

// Shared by ForgotPassword.jsx and VerifyEmail.jsx — both are "enter the 6-digit
// code we emailed you" screens with the same expiry countdown, resend cooldown, and
// auto-verify-on-completion behavior; only what happens on a valid code differs
// (move to a password-reset step vs. log the patient straight in), which stays in
// each page's own onComplete callback rather than in here.
export default function OtpVerificationBlock({
  otp,
  onOtpChange,
  secondsLeft,
  resendCooldown,
  codeExpired,
  canResend,
  submitting,
  onResend,
}) {
  return (
    <>
      {codeExpired && (
        <div className={styles.expiredBanner} role="alert">
          <span>This code has expired.</span>
          <button type="button" onClick={onResend} className={styles.linkButton} disabled={submitting}>
            Send a new code
          </button>
        </div>
      )}

      <div className={styles.field}>
        <label htmlFor="otp-0">
          Verification code<span className={styles.requiredMark}>*</span>
        </label>
        <OtpDigits value={otp} onChange={onOtpChange} disabled={codeExpired || submitting} verifying={submitting} />

        <div className={styles.otpMeta}>
          {!codeExpired && (
            <span className={`${styles.expiryPill} ${secondsLeft <= 60 ? styles.expiryPillUrgent : ""}`}>
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 7v5l3.5 2" />
              </svg>
              Expires in {formatCountdown(secondsLeft)}
            </span>
          )}
          <span className={styles.resendText}>
            {canResend ? (
              <button type="button" onClick={onResend} className={styles.linkButton} disabled={submitting}>
                Resend code
              </button>
            ) : (
              <>Resend code in {formatCountdown(resendCooldown)}</>
            )}
          </span>
        </div>

        {submitting && (
          <div className={styles.verifyingRow}>
            <div className={styles.cubeScene} aria-hidden="true">
              <div className={styles.cube}>
                <span className={`${styles.cubeFace} ${styles.cubeFront}`} />
                <span className={`${styles.cubeFace} ${styles.cubeBack}`} />
                <span className={`${styles.cubeFace} ${styles.cubeRight}`} />
                <span className={`${styles.cubeFace} ${styles.cubeLeft}`} />
                <span className={`${styles.cubeFace} ${styles.cubeTop}`} />
                <span className={`${styles.cubeFace} ${styles.cubeBottom}`} />
              </div>
            </div>
            <span className={styles.fieldHint}>Verifying…</span>
          </div>
        )}
      </div>
    </>
  );
}
