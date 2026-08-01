import styles from "./SuccessCheck.module.css";

// Small checkmark that scales/fades in next to a success message.
export default function SuccessCheck() {
  return (
    <svg
      className={styles.check}
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 13l4 4L19 7" />
    </svg>
  );
}
