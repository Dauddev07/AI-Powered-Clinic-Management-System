import styles from "./StatusBadge.module.css";

const TONE_CLASS = {
  success: styles.success,
  error: styles.error,
  warning: styles.warning,
  info: styles.info,
  neutral: styles.neutral,
};

// Shared status pill used anywhere a status value needs a consistent color:
// appointments, ingestion log rows, doctor active/inactive, slot availability.
// `pulse` adds an animated dot for states that are actively happening right
// now (e.g. an appointment in progress) rather than a static/settled status.
export default function StatusBadge({ tone = "neutral", label, pulse = false }) {
  return (
    <span className={`${styles.badge} ${TONE_CLASS[tone] || styles.neutral}`}>
      {pulse && <span className={styles.dot} aria-hidden="true" />}
      {label}
    </span>
  );
}
