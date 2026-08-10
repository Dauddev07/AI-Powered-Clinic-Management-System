import styles from "./DateInput.module.css";

// Drop-in wrapper for <input type="date"> that adds a visible calendar icon.
// Several mobile browsers (notably iOS Safari) render a native date input with
// no calendar icon at all — the whole field is tappable to open the picker, but
// nothing on screen signals that, unlike desktop Chrome's built-in indicator.
// The icon here is purely decorative (pointer-events: none in the CSS) — clicks
// pass straight through to the input underneath, which already opens the native
// picker on tap/click on every browser that matters, so there's no separate
// click handler to keep in sync with that native behavior.
export default function DateInput({ id, value, onChange, required, className, min, max }) {
  return (
    <div className={styles.wrapper}>
      <input
        id={id}
        type="date"
        required={required}
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        className={className}
        style={{ paddingRight: "2.25rem" }}
      />
      <span className={styles.icon} aria-hidden="true">
        <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="5" width="18" height="16" rx="2.5" />
          <path d="M8 3v4M16 3v4M3 10h18" />
        </svg>
      </span>
    </div>
  );
}
