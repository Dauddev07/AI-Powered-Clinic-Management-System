import { useState } from "react";
import styles from "./PasswordInput.module.css";

// Drop-in replacement for a plain <input type="password">, with a show/hide
// toggle. Deliberately a bare <input> (no wrapper-specific className) inside
// the returned wrapper so each page's own `.field input` CSS rule (Login/
// Register/ChangePassword all style plain descendant inputs, not a specific
// class) still applies exactly as before — only the padding-right below is
// added inline, to make room for the toggle button without fighting each
// page's separately-hashed CSS module on specificity.
export default function PasswordInput({ id, value, onChange, autoComplete, required, placeholder }) {
  const [visible, setVisible] = useState(false);
  const hasContent = value.length > 0;

  const handleChange = (e) => {
    // Emptying the field hides the toggle (see below) — also reset visible so a
    // field that's typed into again afterward starts back at hidden/password,
    // instead of silently resuming in plain-text from whatever it was left at.
    if (!e.target.value) setVisible(false);
    onChange(e);
  };

  return (
    <div className={styles.wrapper}>
      <input
        id={id}
        type={visible ? "text" : "password"}
        required={required}
        value={value}
        onChange={handleChange}
        autoComplete={autoComplete}
        placeholder={placeholder}
        style={{ paddingRight: "2.5rem" }}
      />
      {hasContent && (
        <button
          type="button"
          className={styles.toggle}
          onClick={() => setVisible((v) => !v)}
          aria-label={visible ? "Hide password" : "Show password"}
          aria-pressed={visible}
          tabIndex={-1}
        >
          {visible ? (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 3l18 18" />
              <path d="M10.58 10.58a2 2 0 0 0 2.83 2.83" />
              <path d="M9.88 5.09A9.77 9.77 0 0 1 12 5c5 0 9 4.5 10 7-.36.92-1.02 2-1.94 3M6.1 6.1C4.16 7.42 2.68 9.34 2 12c1 2.5 5 7 10 7 1.19 0 2.3-.25 3.31-.68" />
            </svg>
          )}
        </button>
      )}
    </div>
  );
}
