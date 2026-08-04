import { createContext, useCallback, useContext, useRef, useState } from "react";
import { createPortal } from "react-dom";
import styles from "./Toast.module.css";

const ToastContext = createContext(null);
let idCounter = 0;

// Shared toast infrastructure — previously every page that needed transient
// success/error feedback rolled its own inline banner (e.g. UpcomingAppointments.jsx's
// local "message"/"error" state rendered as a box above the table, only visible if the
// patient happened to still be looking at that section). This centralizes it: any page
// under this provider can call useToast() and get a consistent, auto-dismissing
// notification instead of reinventing the pattern.
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const timers = useRef(new Map());

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const showToast = useCallback(
    (message, { tone = "success", duration = 4000 } = {}) => {
      const id = ++idCounter;
      // Prepended, not appended: the viewport is anchored to the top edge (see
      // Toast.module.css), so the newest toast should render as the first DOM
      // child to land closest to that edge, pushing older ones down beneath it.
      setToasts((prev) => [{ id, message, tone }, ...prev]);
      if (duration > 0) {
        const timer = setTimeout(() => dismiss(id), duration);
        timers.current.set(id, timer);
      }
      return id;
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={showToast}>
      {children}
      {createPortal(
        <div className={styles.viewport} role="status" aria-live="polite">
          {toasts.map((t) => (
            <div key={t.id} className={`${styles.toast} ${styles[t.tone] || ""}`}>
              <span className={styles.toastMessage}>{t.message}</span>
              <button type="button" className={styles.closeBtn} onClick={() => dismiss(t.id)} aria-label="Dismiss">
                ✕
              </button>
            </div>
          ))}
        </div>,
        document.body,
      )}
    </ToastContext.Provider>
  );
}

// Returns a showToast(message, { tone, duration }) function — tone is
// "success" | "error" | "info", duration is ms (0 disables auto-dismiss).
export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within a ToastProvider");
  return ctx;
}
