import styles from "./PhoneInput.module.css";

// Pakistan-only for now (expandable later): +92 followed by exactly 10 digits,
// first digit not 0. This mirrors backend/app/schemas/auth.py's _PHONE_RE — the
// backend is the authoritative check; this only gives instant UI feedback.
export function sanitizePhoneDigits(raw) {
  let digits = raw.replace(/\D/g, "");
  digits = digits.replace(/^0+/, ""); // block/strip a leading 0 (typed or pasted)
  return digits.slice(0, 10);
}

export function isPhoneDigitsComplete(digits) {
  return digits.length === 10;
}

export function toE164(digits) {
  return digits ? `+92${digits}` : null;
}

export default function PhoneInput({ id, digits, onDigitsChange, required = false }) {
  const touched = digits.length > 0;
  const invalid = touched && !isPhoneDigitsComplete(digits);

  return (
    <div>
      <div className={`${styles.wrapper} ${invalid ? styles.invalid : ""}`}>
        <span className={styles.prefix}>+92</span>
        <input
          id={id}
          className={styles.input}
          type="tel"
          inputMode="numeric"
          placeholder="3XX XXXXXXX"
          value={digits}
          onChange={(e) => onDigitsChange(sanitizePhoneDigits(e.target.value))}
          maxLength={10}
          required={required}
        />
      </div>
      <div className={`${styles.hint} ${invalid ? styles.hintError : ""}`}>
        Enter 10 digits, not starting with 0 (e.g. 3001234567).
      </div>
    </div>
  );
}
