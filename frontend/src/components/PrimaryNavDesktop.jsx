import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { getIconPath, getNavItems } from "./primaryNavItems";
import styles from "./PrimaryNavDesktop.module.css";

// The row of always-visible destinations under the top header bar, ≥720px
// (below that, PrimaryNavMobile's fixed bottom tab bar takes over — see its
// own comment). Reported live: Book Appointment, Upcoming Appointments and
// Chat all used to live one tap deeper inside the account menu (SettingsMenu)
// — this promotes the handful of screens patients/admins reach for
// constantly to a single tap, while the account menu keeps the occasional
// stuff (Profile, Settings, Logout, CSV import, etc).
//
// Rendered from inside AppHeader's own <header ref={headerRef}> (not as a
// separate element) so its height folds into --app-header-height
// automatically via the ResizeObserver already there — nothing extra needed
// for the chat page's full-bleed sizing or the landing hero to stay correct.
export default function PrimaryNavDesktop() {
  const { isAuthenticated, user } = useAuth();
  const location = useLocation();
  const [homeSpinning, setHomeSpinning] = useState(false);

  // Hidden on the landing page — a signed-in patient/admin who navigates
  // there (via the header's brand link) is meant to see it as the public
  // marketing page, with its own "Back to Dashboard" link (see Landing.jsx),
  // not as another app screen wearing the app's own persistent nav.
  if (!isAuthenticated || location.pathname === "/") return null;

  const dashboardPath = user?.role === "admin" ? "/admin" : "/patient";
  const items = getNavItems(user?.role, dashboardPath);

  return (
    <nav className={styles.navRow} aria-label="Primary">
      {/* Instructed live: a centered rounded "capsule" instead of a flat,
          edge-to-edge strip — reads as its own distinct control rather than
          just another toolbar row, and matches the floating-dock language
          PrimaryNavMobile's bottom tab bar already uses on small screens. */}
      <div className={styles.navCapsule}>
      {items.map((item) => {
        const active = location.pathname === item.to;
        const isHome = item.key === "home";
        const isChat = item.key === "chat";
        return (
          <Link
            key={item.key}
            to={item.to}
            className={`${styles.navItem} ${active ? styles.active : ""} ${isChat ? styles.chat : ""}`}
            aria-current={active ? "page" : undefined}
            onClick={
              isHome
                ? () => {
                    // Clicking Home while already on the dashboard still
                    // navigates — react-router mints a fresh location.key for
                    // a Link to the current path, which remounts the routed
                    // page (see App.jsx's page-transition wrapper) and
                    // re-runs its data fetch, so this doubles as a refresh
                    // button rather than doing nothing. Same behavior (and
                    // spin feedback) the old header home button had.
                    setHomeSpinning(false);
                    requestAnimationFrame(() => setHomeSpinning(true));
                  }
                : undefined
            }
          >
            <svg
              className={isHome && homeSpinning ? styles.iconSpin : ""}
              onAnimationEnd={isHome ? () => setHomeSpinning(false) : undefined}
              viewBox="0 0 24 24"
              width="15"
              height="15"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.1"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d={getIconPath(item.icon)} />
            </svg>
            {item.label}
          </Link>
        );
      })}
      </div>
    </nav>
  );
}
