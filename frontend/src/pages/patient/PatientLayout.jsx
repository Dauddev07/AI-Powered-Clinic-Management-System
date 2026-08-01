import { Outlet, useLocation } from "react-router-dom";
import styles from "./PatientLayout.module.css";

// Navigation for the patient area lives in the global AppHeader (role-aware,
// collapsing into the same toggle as everywhere else) — this layout is just
// the content wrapper.
//
// The chat page is the one exception to the shared centered-column/padded
// layout every other patient screen uses: it's meant to fill the viewport
// edge-to-edge like a dedicated app surface, not sit in a card within a
// 1100px reading column, so it opts out of both the padding and the max-width
// here rather than fighting them from inside ChatPage.
export default function PatientLayout() {
  const location = useLocation();
  const isFullBleed = location.pathname.startsWith("/patient/chat");

  return (
    <main className={`${styles.content} ${isFullBleed ? styles.contentFullBleed : ""}`}>
      <div className={`${styles.contentInner} ${isFullBleed ? styles.contentInnerFullBleed : ""}`}>
        <Outlet />
      </div>
    </main>
  );
}
