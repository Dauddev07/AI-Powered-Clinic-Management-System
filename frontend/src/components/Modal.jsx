import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import styles from "./Modal.module.css";

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Portaled to <body> and positioned fixed relative to the viewport — same reasoning
// as the doctor search dropdown: a card living inside a normal document-flow ancestor
// can never reliably out-stack a later sibling's own stacking context, so a modal
// (which must always sit above everything) can't safely be a plain in-place child.
export default function Modal({ open, onClose, title, children }) {
  const contentRef = useRef(null);
  const previouslyFocusedRef = useRef(null);

  useEffect(() => {
    if (!open) return;

    // role="dialog" aria-modal="true" implies focus stays inside this dialog while
    // it's open — without actually moving focus in and trapping Tab, a keyboard user
    // opening this modal keeps tabbing through the page behind it instead.
    previouslyFocusedRef.current = document.activeElement;
    const firstFocusable = contentRef.current?.querySelector(FOCUSABLE_SELECTOR);
    (firstFocusable || contentRef.current)?.focus();

    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab" || !contentRef.current) return;
      const focusable = Array.from(contentRef.current.querySelectorAll(FOCUSABLE_SELECTOR));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      // Return focus to whatever triggered the modal (e.g. the "Delete" button) so a
      // keyboard user isn't dropped back at the top of the page when it closes.
      previouslyFocusedRef.current?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      className={styles.overlay}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={styles.content}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modalTitle"
        ref={contentRef}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <h2 id="modalTitle" className={styles.title}>
            {title}
          </h2>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className={styles.body}>{children}</div>
      </div>
    </div>,
    document.body,
  );
}
