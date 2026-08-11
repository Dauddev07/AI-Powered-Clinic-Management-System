import { Outlet } from "react-router-dom";
import styles from "./AdminLayout.module.css";

// Navigation for the admin area lives in PrimaryNavDesktop (a row inside the
// global AppHeader, ≥720px) and PrimaryNavMobile (a fixed bottom tab bar
// below that width) — this layout is just the content wrapper.
export default function AdminLayout() {
  return (
    <main className={styles.content}>
      <div className={styles.contentInner}>
        <Outlet />
      </div>
    </main>
  );
}
