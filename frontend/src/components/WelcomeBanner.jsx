import { useReveal } from "../hooks/useReveal";
import styles from "./WelcomeBanner.module.css";

function greetingForHour(hour) {
  if (hour < 5) return "Still up";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

// Shared banner for both dashboard hubs (patient + admin) — same visual
// treatment driven entirely by props, so the two dashboards read as one
// consistent product rather than two separately-designed screens.
export default function WelcomeBanner({ name, tagline }) {
  const revealRef = useReveal();
  const greeting = greetingForHour(new Date().getHours());

  return (
    <div className={`${styles.banner} reveal`} ref={revealRef}>
      <span className={styles.dotTexture} aria-hidden="true" />
      <div className={styles.text}>
        <span className={styles.eyebrow}>{greeting}</span>
        {/* No watermark/icon — instructed live to drop both. A static
            gradient underline (see .heading::after) sits under "Welcome,
            Name" — the animated grow-in/shimmer version was instructed live
            to be removed. */}
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
