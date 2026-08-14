import styles from "./Logo.module.css";

// Text-based wordmark for "Quick Check Clinic" with a small pulse/check glyph
// standing in for the dot on the "i" in "Quick" — keeps the mark simple and
// legible at header size while still reading as a distinct brand icon.
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
      {/* One continuous stroke, not two separate glyphs: a heartbeat trace that
          rises straight into a checkmark's long stroke, so "pulse" (clinic) and
          "check" (quick, confirmed) read as a single unbroken gesture rather than
          a pulse icon and a check icon awkwardly sharing one circle. */}
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
          d="M5 12.5h3.3l2-5.3 2.4 9.8L20 7.5"
          fill="none"
          stroke="var(--app-accent-text)"
          strokeWidth="1.7"
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
