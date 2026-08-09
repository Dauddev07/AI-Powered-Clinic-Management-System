import { useId, useState } from "react";
import { useTheme } from "../theme/ThemeContext";
import styles from "./ThemeToggle.module.css";

// Icon shows the mode a click will SWITCH TO, not the current one — a sun
// while in dark mode (click for daylight), a moon while in light mode (click
// for night). Each glyph carries its own gradient fill (warm gold for the
// sun, deep indigo for the moon) rather than a flat currentColor stroke like
// the header's other icons — this button doubles as a small colorful badge,
// not just another line-icon blending into the row. Gradient ids are unique
// per mounted instance (useId) so two toggles on screen at once (unlikely
// today, but AppHeader renders one per header branch) never collide.
function SunIcon({ gradientId }) {
  return (
    <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true">
      <defs>
        <radialGradient id={gradientId} cx="35%" cy="30%" r="75%">
          <stop offset="0%" stopColor="#fff3d6" />
          <stop offset="55%" stopColor="#ffcf5c" />
          <stop offset="100%" stopColor="#ff9d3d" />
        </radialGradient>
      </defs>
      <g fill="none" stroke={`url(#${gradientId})`} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="4.6" fill={`url(#${gradientId})`} stroke="none" />
        <path d="M12 2.5v2.15M12 19.35v2.15M4.22 4.22l1.52 1.52M18.26 18.26l1.52 1.52M2.5 12h2.15M19.35 12h2.15M4.22 19.78l1.52-1.52M18.26 5.74l1.52-1.52" />
      </g>
    </svg>
  );
}

function MoonIcon({ gradientId }) {
  return (
    <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true">
      <defs>
        <linearGradient id={gradientId} x1="20%" y1="0%" x2="85%" y2="100%">
          <stop offset="0%" stopColor="#dfe4ff" />
          <stop offset="55%" stopColor="#8f9cf0" />
          <stop offset="100%" stopColor="#4c56b8" />
        </linearGradient>
      </defs>
      <path
        d="M20.5 14.2A8.5 8.5 0 1 1 9.8 3.5a7 7 0 0 0 10.7 10.7Z"
        fill={`url(#${gradientId})`}
      />
      {/* Two small stars for a touch of night-sky charm — twinkle on their own
          idle cycle, the moon's counterpart to the sun icon's breathing glow
          below (previously asymmetric: only the sun animated at rest). */}
      <path
        className={styles.moonStarBig}
        d="M18.6 4.3l.55 1.35 1.35.55-1.35.55-.55 1.35-.55-1.35-1.35-.55 1.35-.55Z"
        fill="#dfe4ff"
      />
      <circle className={styles.moonStarSmall} cx="15.3" cy="8.6" r="0.55" fill="#dfe4ff" opacity="0.85" />
    </svg>
  );
}

// Brief, direction-specific particle burst that plays once on every toggle,
// alongside the existing icon-swap spin — gold rays radiating outward for the
// sun (daylight "switching on"), small indigo sparks scattering for the moon
// (night settling in). Purely decorative (aria-hidden), unmounted between
// bursts via the `active` gate so it never lingers as a static shape behind
// the icon once its animation ends.
function SunRayBurst() {
  return (
    <span className={styles.burstWrap} aria-hidden="true">
      {Array.from({ length: 8 }).map((_, i) => (
        <span key={i} className={styles.sunRay} style={{ "--i": i }} />
      ))}
    </span>
  );
}

function MoonSparkBurst() {
  return (
    <span className={styles.burstWrap} aria-hidden="true">
      {Array.from({ length: 6 }).map((_, i) => (
        <span key={i} className={styles.moonSpark} style={{ "--i": i }} />
      ))}
    </span>
  );
}

export default function ThemeToggle({ className }) {
  const { isDark, toggleTheme } = useTheme();
  const [spinning, setSpinning] = useState(false);
  const gradientId = useId();

  return (
    <button
      type="button"
      className={`${styles.toggleBtn} ${isDark ? styles.toggleBtn_sun : styles.toggleBtn_moon} ${className || ""}`}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      onClick={() => {
        toggleTheme();
        setSpinning(false);
        requestAnimationFrame(() => setSpinning(true));
      }}
    >
      <span className={styles.glowRing} aria-hidden="true" />
      {spinning && (isDark ? <SunRayBurst /> : <MoonSparkBurst />)}
      <span
        key={isDark ? "sun" : "moon"}
        className={`${styles.iconWrap} ${spinning ? styles.iconSpin : ""}`}
        onAnimationEnd={() => setSpinning(false)}
      >
        {isDark ? <SunIcon gradientId={`${gradientId}-sun`} /> : <MoonIcon gradientId={`${gradientId}-moon`} />}
      </span>
    </button>
  );
}
