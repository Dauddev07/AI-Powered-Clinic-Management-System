import { useReveal } from "../hooks/useReveal";
import styles from "./WelcomeBanner.module.css";

function greetingForHour(hour) {
  if (hour < 5) return "Still up";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

// Shared hero-style banner for both dashboard hubs (patient + admin) — same
// visual treatment (accent glow, brand pulse motif, time-aware greeting) driven
// entirely by props, so the two dashboards read as one consistent product
// rather than two separately-designed screens.
export default function WelcomeBanner({ name, tagline }) {
  const revealRef = useReveal();
  const greeting = greetingForHour(new Date().getHours());

  return (
    <div className={`${styles.banner} reveal`} ref={revealRef}>
      <div className={styles.glow} aria-hidden="true" />
      <svg
        className={styles.motif}
        viewBox="0 0 24 24"
        aria-hidden="true"
        focusable="false"
      >
        <path
          d="M5.5 12.5L9 16l3.2-4.2M12.5 12h2.5l1.3-3 1.4 6 1.3-3H21"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      <div className={styles.text}>
        <span className={styles.eyebrow}>{greeting}</span>
        <h1 className={styles.heading}>
          Welcome{name && (
            <>
              , <span className={styles.name}>{name}</span>
            </>
          )}
        </h1>
        {tagline && <p className={styles.tagline}>{tagline}</p>}
      </div>
    </div>
  );
}
