import styles from "./PasswordRequirements.module.css";

const RULES = [
  { key: "length", label: "8+ characters", test: (v) => v.length >= 8 },
  { key: "letter", label: "A letter", test: (v) => /[A-Za-z]/.test(v) },
  { key: "digit", label: "A number", test: (v) => /[0-9]/.test(v) },
];

// Live checklist shown while typing a new password — each rule turns green
// as soon as it's satisfied, matching the same rules Register/ChangePassword
// already validate on submit.
export default function PasswordRequirements({ password }) {
  return (
    <ul className={styles.list}>
      {RULES.map(({ key, label, test }) => {
        const met = test(password);
        return (
          <li key={key} className={met ? styles.met : styles.unmet}>
            <span className={styles.icon} aria-hidden="true">
              {met ? (
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 6 9 17l-5-5" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="9" />
                </svg>
              )}
            </span>
            {label}
          </li>
        );
      })}
    </ul>
  );
}
