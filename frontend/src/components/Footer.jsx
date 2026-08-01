import styles from "./Footer.module.css";

// Minimal, sitewide footer — just the rights line, rendered once at the app
// level so every route gets it consistently.
export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.rights}>© 2026 Quick Check Clinic. All rights reserved.</div>
    </footer>
  );
}
