import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { getIconPath, getNavItems } from "./primaryNavItems";
import styles from "./PrimaryNavMobile.module.css";

// Fixed bottom tab bar, <720px only (see PrimaryNavDesktop.module.css's own
// breakpoint) — same destinations as the desktop nav row, just the mobile-app
// convention instead of a top row once there's no room for a horizontal bar
// next to the header. Hidden on the chat page: it's already a full-bleed,
// edge-to-edge surface with its own bottom input bar (see PatientLayout.jsx),
// and a second fixed bottom bar would sit on top of it.
export default function PrimaryNavMobile() {
  const { isAuthenticated, user } = useAuth();
  const location = useLocation();
  const isChatPage = location.pathname === "/patient/chat";

  // Page content (and the footer, which sits right after it — see App.jsx)
  // needs room reserved at the bottom so this fixed bar never covers the
  // last bit of it. Toggled as a body class rather than threading a prop
  // through every layout, same direct document.body approach SettingsMenu
  // already uses for its own scroll lock. The actual padding only applies
  // under 720px (see index.css) — this class is otherwise inert above that.
  useEffect(() => {
    const shouldReserveSpace = isAuthenticated && !isChatPage;
    document.body.classList.toggle("has-primary-nav-mobile", shouldReserveSpace);
    return () => document.body.classList.remove("has-primary-nav-mobile");
  }, [isAuthenticated, isChatPage]);

  if (!isAuthenticated || isChatPage) return null;

  const dashboardPath = user?.role === "admin" ? "/admin" : "/patient";
  const items = getNavItems(user?.role, dashboardPath);

  return (
    <nav className={styles.tabBar} aria-label="Primary">
      {items.map((item) => {
        const active = location.pathname === item.to;
        const isChat = item.key === "chat";
        return (
          <Link
            key={item.key}
            to={item.to}
            className={`${styles.tabItem} ${active ? styles.active : ""}`}
            aria-current={active ? "page" : undefined}
          >
            {/* Chat used to be its own floating action button (see the removed
                FloatingChatButton) — the filled accent badge keeps that same
                "this one's special" weight now that it's a tab like the rest. */}
            <span className={isChat ? styles.tabIconChat : styles.tabIcon}>
              <svg viewBox="0 0 24 24" width={isChat ? 17 : 20} height={isChat ? 17 : 20} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d={getIconPath(item.icon)} />
              </svg>
            </span>
            <span>{item.shortLabel}</span>
          </Link>
        );
      })}
    </nav>
  );
}
