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
      {/* Instructed live: no icon or image at all — the card's depth comes
          from typography instead. A huge, very faint echo of "Welcome"
          itself bleeding off the card's own edge, the same "oversized
          ghost-text" flourish an editorial poster/masthead uses, built
          entirely from the same word already on the card rather than a new
          decorative asset. */}
      <span className={styles.watermarkText} aria-hidden="true">
        Welcome
      </span>
      <div className={styles.text}>
        <span className={styles.eyebrow}>{greeting}</span>
        <h1 className={styles.heading}>
          Welcome{name && (
            <>
              , <span className={styles.name}>{name}</span>
            </>
          )}
        </h1>
        <span className={styles.headingRule} aria-hidden="true" />
        {tagline && <p className={styles.tagline}>{tagline}</p>}
      </div>
    </div>
  );
}
