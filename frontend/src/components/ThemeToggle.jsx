import { useState } from "react";
import { useTheme } from "../theme/ThemeContext";
import styles from "./ThemeToggle.module.css";

// Icon shows the mode a click will SWITCH TO, not the current one — a sun
// while in dark mode (click for daylight), a moon while in light mode (click
// for night). Same stroke-icon language as the header's other buttons
// (AppHeader's home icon, StarIcon, etc.): viewBox 24, currentColor stroke.
function SunIcon(props) {
  return (
    <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
      <circle cx="12" cy="12" r="4.5" />
      <path d="M12 2.5v2.25M12 19.25v2.25M4.22 4.22l1.6 1.6M18.18 18.18l1.6 1.6M2.5 12h2.25M19.25 12h2.25M4.22 19.78l1.6-1.6M18.18 5.82l1.6-1.6" />
    </svg>
  );
}

function MoonIcon(props) {
  return (
    <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
      <path d="M20.5 14.2A8.5 8.5 0 1 1 9.8 3.5a7 7 0 0 0 10.7 10.7Z" />
    </svg>
  );
}

export default function ThemeToggle({ className }) {
  const { isDark, toggleTheme } = useTheme();
  const [spinning, setSpinning] = useState(false);

  return (
    <button
      type="button"
      className={`${styles.toggleBtn} ${className || ""}`}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      onClick={() => {
        toggleTheme();
        setSpinning(false);
        requestAnimationFrame(() => setSpinning(true));
      }}
    >
      <span
        key={isDark ? "sun" : "moon"}
        className={`${styles.iconWrap} ${spinning ? styles.iconSpin : ""}`}
        onAnimationEnd={() => setSpinning(false)}
      >
        {isDark ? <SunIcon /> : <MoonIcon />}
      </span>
    </button>
  );
}
