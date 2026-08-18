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
        {/* No watermark/icon — instructed live to drop both. The "Welcome,
            Name" line itself is the one thing this card is built to draw the
            eye to now: a gradient underline grows in under it (see
            .heading::after) right after the text settles, then a single
            light sweep passes across it once (.heading::before) — animated,
            but a one-time flourish rather than a permanent loop, same
            restraint the .name shimmer just above it already uses. */}
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
