import { Outlet } from "react-router-dom";
import styles from "./AdminLayout.module.css";

// Navigation for the admin area lives in the global AppHeader (role-aware,
// collapsing into the same toggle as everywhere else) — this layout is just
// the content wrapper.
export default function AdminLayout() {
  return (
    <main className={styles.content}>
      <div className={styles.contentInner}>
        <Outlet />
      </div>
    </main>
  );
}
