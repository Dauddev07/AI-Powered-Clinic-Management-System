import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./FloatingChatButton.module.css";

// Entry point into the AI chat, fixed bottom-left on the patient dashboard. The
// tooltip bubble is permanent (not hover-only) — it fades in shortly after mount and
// then stays up, so the invitation to chat is always legible, not just discoverable.
export default function FloatingChatButton() {
  const navigate = useNavigate();
  const [tooltipVisible, setTooltipVisible] = useState(false);

  useEffect(() => {
    const showTimer = setTimeout(() => setTooltipVisible(true), 500);
    return () => clearTimeout(showTimer);
  }, []);

  return (
    <div className={styles.wrapper}>
      <div className={`${styles.tooltip} ${tooltipVisible ? styles.tooltipVisible : ""}`} role="status">
        <span className={styles.tooltipSparkle} aria-hidden="true">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true">
            <path d="M12 2 13.8 8.9 21 12l-7.2 3.1L12 22l-1.8-6.9L3 12l7.2-3.1L12 2Z" />
          </svg>
        </span>
        Need help? I'm here
      </div>

      <button type="button" className={styles.button} aria-label="Open AI assistant chat" onClick={() => navigate("/patient/chat")}>
        <span className={styles.ring} aria-hidden="true" />
        <span className={styles.pulse} aria-hidden="true" />
        <span className={styles.glow} aria-hidden="true" />
        <svg
          className={styles.icon}
          viewBox="0 0 24 24"
          width="27"
          height="27"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M4 12.5C4 7.81 8.03 4 13 4s9 3.81 9 8.5-4.03 8.5-9 8.5c-1.09 0-2.13-.19-3.1-.53L4 21l1.2-4.02A8.16 8.16 0 0 1 4 12.5Z" />
          <circle cx="9.5" cy="12.5" r="1" fill="currentColor" stroke="none" />
          <circle cx="13" cy="12.5" r="1" fill="currentColor" stroke="none" />
          <circle cx="16.5" cy="12.5" r="1" fill="currentColor" stroke="none" />
        </svg>
        <svg className={styles.sparkle} viewBox="0 0 24 24" width="12" height="12" fill="currentColor" aria-hidden="true">
          <path d="M12 2 13.8 8.9 21 12l-7.2 3.1L12 22l-1.8-6.9L3 12l7.2-3.1L12 2Z" />
        </svg>
      </button>
    </div>
  );
}
