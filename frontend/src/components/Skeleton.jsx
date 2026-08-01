import styles from "./Skeleton.module.css";

// Pulsing placeholder rows shown while a table/list is fetching, instead of a
// blank gap or a plain "Loading…" line.
export default function Skeleton({ rows = 3 }) {
  return (
    <div className={styles.skeleton} aria-hidden="true">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className={styles.row}>
          <span className={styles.bar} style={{ width: "22%" }} />
          <span className={styles.bar} style={{ width: "28%" }} />
          <span className={styles.bar} style={{ width: "18%" }} />
          <span className={styles.bar} style={{ width: "14%" }} />
        </div>
      ))}
    </div>
  );
}
