import { useReveal } from "../hooks/useReveal";
import styles from "./WelcomeBanner.module.css";

function greetingForHour(hour) {
  if (hour < 5) return "Still up";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

// Sun for daylight hours, moon for night — reported live: the plain neutral
// card that replaced the old green-washed one read as "just like every other
// card" with nothing to anchor it. A real icon (not decoration for its own
// sake — it reflects the same hour-based greeting logic right next to it)
// gives the banner its own focal point the way a stat card's icon badge
// gives that card one, instead of being three lines of text alone.
function iconForHour(hour) {
  return hour < 6 || hour >= 18 ? "moon" : "sun";
}

// Shared banner for both dashboard hubs (patient + admin) — same visual
// treatment driven entirely by props, so the two dashboards read as one
// consistent product rather than two separately-designed screens.
export default function WelcomeBanner({ name, tagline }) {
  const revealRef = useReveal();
  const hour = new Date().getHours();
  const greeting = greetingForHour(hour);
  const icon = iconForHour(hour);

  return (
    <div className={`${styles.banner} reveal`} ref={revealRef}>
      <span className={styles.dotTexture} aria-hidden="true" />
      <span className={styles.iconBadge} aria-hidden="true">
        {icon === "sun" ? (
          <svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="4.5" />
            <path d="M12 2.5v2.5M12 19v2.5M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2.5 12H5M19 12h2.5M4.2 19.8 6 18M18 6l1.8-1.8" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" />
          </svg>
        )}
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
        {tagline && <p className={styles.tagline}>{tagline}</p>}
      </div>
    </div>
  );
}
