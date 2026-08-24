import styles from "./PasswordRequirements.module.css";

const RULES = [
  { key: "length", label: "8+ characters", test: (v) => v.length >= 8 },
  { key: "letter", label: "A letter", test: (v) => /[A-Za-z]/.test(v) },
  { key: "digit", label: "A number", test: (v) => /[0-9]/.test(v) },
];

const STRENGTH_LABEL = ["Too short", "Weak", "Good", "Strong"];

// Live checklist + strength meter shown while typing a new password — each
// chip fills in and checks off as its rule is met, matching the same rules
// Register/ChangePassword already validate on submit.
export default function PasswordRequirements({ password }) {
  const metFlags = RULES.map((rule) => rule.test(password));
  const metCount = metFlags.filter(Boolean).length;
  const strength = password.length === 0 ? 0 : metCount;

  return (
    <div className={styles.wrap}>
      <div className={styles.meter} aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <span key={i} className={i < strength ? `${styles.bar} ${styles[`level${strength}`]}` : styles.bar} />
        ))}
      </div>

      <ul className={styles.list}>
        {RULES.map(({ key, label }, i) => {
          const met = metFlags[i];
          return (
            <li key={key} className={met ? styles.met : styles.unmet}>
              <span className={styles.icon} aria-hidden="true">
                {met ? (
                  <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                ) : (
                  <span className={styles.dot} />
                )}
              </span>
              {label}
            </li>
          );
        })}
      </ul>

      {password.length > 0 && (
        <span className={`${styles.strengthLabel} ${styles[`level${strength}`]}`}>{STRENGTH_LABEL[strength]}</span>
      )}
    </div>
  );
}
