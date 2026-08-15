import styles from "./Logo.module.css";

// Text-based wordmark for "Quick Check Clinic" paired with a chat-bubble glyph
// containing a pulse-into-checkmark stroke — a single icon rather than several
// competing symbols, chosen specifically to stay legible down to favicon size.
//
// `wrap` lets the wordmark break onto a second line instead of truncating with
// an ellipsis — for narrow-but-tall placements (e.g. Register's fixed-width
// intro sidebar) where there's plenty of vertical room and truncating "Clinic"
// down to "Cli…" reads worse than just wrapping it.
//
// `compact` drops the wordmark entirely below a narrow width, leaving just the
// glyph — for AppHeader specifically, where the brand shares a single row with
// the home/bell/avatar icon group on the authenticated header, and a fixed
// ellipsis-truncated "Quick Che…" reads as broken rather than just tight on
// space. Landing/Login/Register don't compete for that row, so they keep the
// full wordmark at every width.
export default function Logo({ size = "md", wrap = false, compact = false }) {
  return (
    <span
      className={`${styles.logo} ${size === "lg" ? styles.lg : ""} ${wrap ? styles.wrap : ""} ${
        compact ? styles.compact : ""
      }`}
    >
      {/* A chat bubble (ties directly to the AI assistant) with one continuous
          stroke inside it — a heartbeat trace that rises straight into a
          checkmark's long stroke, so "pulse" (clinic) and "check" (quick,
          confirmed) read as a single unbroken gesture rather than two icons
          sharing one shape. */}
      <svg
        className={styles.glyph}
        viewBox="0 0 24 24"
        width="1em"
        height="1em"
        aria-hidden="true"
        focusable="false"
      >
        <rect x="2" y="3" width="20" height="13" rx="6.5" ry="6.5" fill="var(--app-accent)" />
        <path d="M7.5 16 5.5 20 11 16Z" fill="var(--app-accent)" />
        <path
          d="M5.3 10.3h2.3l1.7-3.6 2 7.8L18.5 6.3"
          fill="none"
          stroke="var(--app-accent-text)"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className={styles.wordmark}>
        Quick Check <span className={styles.clinic}>Clinic</span>
      </span>
    </span>
  );
}
