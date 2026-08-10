import { useRef } from "react";
import styles from "./DateInput.module.css";

// Drop-in wrapper for <input type="date"> that adds a visible calendar icon.
// Several mobile browsers (notably iOS Safari) render a native date input with
// no calendar icon at all — the whole field is tappable to open the picker, but
// nothing on screen signals that, unlike desktop Chrome's built-in icon. This
// draws a decorative icon that also calls the input's native showPicker() (the
// standard way to open a date input's picker programmatically) so it's a real
// affordance, not just decoration, on browsers that support it — elsewhere it
// just focuses the input, same as tapping the field itself already does.
export default function DateInput({ id, value, onChange, required, className, min, max }) {
  const inputRef = useRef(null);

  const openPicker = () => {
    const el = inputRef.current;
    if (!el) return;
    if (typeof el.showPicker === "function") {
      el.showPicker();
    } else {
      el.focus();
    }
  };

  return (
    <div className={styles.wrapper}>
      <input
        ref={inputRef}
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
      <button
        type="button"
        className={styles.icon}
        onClick={openPicker}
        aria-label="Open date picker"
        tabIndex={-1}
      >
        <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="3" y="5" width="18" height="16" rx="2.5" />
          <path d="M8 3v4M16 3v4M3 10h18" />
        </svg>
      </button>
    </div>
  );
}
