import { Link } from "react-router-dom";
import styles from "./EmptyState.module.css";

const ICONS = {
  calendar: (
    <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" />
  ),
  inbox: (
    <path d="M4 12h4l2 3h4l2-3h4M4 12l1.5-6.5A1 1 0 0 1 6.47 4.75h11.06a1 1 0 0 1 .97.75L20 12M4 12v6a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-6" />
  ),
  search: <path d="M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14ZM21 21l-4.35-4.35" />,
  star: <path d="M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z" />,
};

// Small icon + message + optional direct-action button, used for every "nothing
// here yet" state instead of a dead-end line of gray text.
export default function EmptyState({ icon = "inbox", message, actionLabel, actionTo, onAction }) {
  return (
    <div className={styles.emptyState}>
      <svg
        className={styles.icon}
        viewBox="0 0 24 24"
        width="40"
        height="40"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {ICONS[icon] || ICONS.inbox}
      </svg>
      <p className={styles.message}>{message}</p>
      {actionLabel && actionTo && (
        <Link to={actionTo} className={styles.action}>
          {actionLabel}
        </Link>
      )}
      {actionLabel && onAction && !actionTo && (
        <button type="button" className={styles.action} onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}
