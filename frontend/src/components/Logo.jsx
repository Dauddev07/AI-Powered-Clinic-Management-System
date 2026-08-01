import styles from "./Logo.module.css";

// Text-based wordmark for "Quick Check Clinic" with a small pulse/check glyph
// standing in for the dot on the "i" in "Quick" — keeps the mark simple and
// legible at header size while still reading as a distinct brand icon.
//
// `wrap` lets the wordmark break onto a second line instead of truncating with
// an ellipsis — for narrow-but-tall placements (e.g. Register's fixed-width
// intro sidebar) where there's plenty of vertical room and truncating "Clinic"
// down to "Cli…" reads worse than just wrapping it.
export default function Logo({ size = "md", wrap = false }) {
  return (
    <span className={`${styles.logo} ${size === "lg" ? styles.lg : ""} ${wrap ? styles.wrap : ""}`}>
      <svg
        className={styles.glyph}
        viewBox="0 0 24 24"
        width="1em"
        height="1em"
        aria-hidden="true"
        focusable="false"
      >
        <circle cx="12" cy="12" r="11" fill="var(--app-accent)" />
        <path
          d="M5.5 12.5L9 16l3.2-4.2M12.5 12h2.5l1.3-3 1.4 6 1.3-3H21"
          fill="none"
          stroke="var(--app-accent-text)"
          strokeWidth="1.6"
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
