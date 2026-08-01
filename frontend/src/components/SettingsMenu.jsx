import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { fetchMyAccount } from "../api/auth";
import { useAuth } from "../auth/AuthContext";
import styles from "./SettingsMenu.module.css";

const ICONS = {
  profile: <path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4.5 20.5c0-4.14 3.36-7.5 7.5-7.5s7.5 3.36 7.5 7.5" />,
  calendar: <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" />,
  clock: <path d="M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />,
  settings: <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM19.4 13a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V19a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H4a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H10a1.65 1.65 0 0 0 1-1.51V4a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V10a1.65 1.65 0 0 0 1.51 1H20a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />,
  logout: <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4M10 17l5-5-5-5M15 12H3" />,
  lock: <path d="M6 11V8a6 6 0 0 1 12 0v3M5 11h14a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Z" />,
  back: <path d="M15 18l-6-6 6-6" />,
  chevron: <path d="M9 18l6-6-6-6" />,
  doctors: (
    <path d="M8 21v-2a4 4 0 0 1 4-4h0a4 4 0 0 1 4 4v2M12 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM3 21v-1a3 3 0 0 1 3-3M21 21v-1a3 3 0 0 0-3-3M5.5 10a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5ZM18.5 10a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
  ),
  upload: <path d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />,
  log: (
    <path d="M4 12h4l2 3h4l2-3h4M4 12l1.5-6.5A1 1 0 0 1 6.47 4.75h11.06a1 1 0 0 1 .97.75L20 12M4 12v6a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-6" />
  ),
  book: (
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V4a2 2 0 0 0-2-2H6.5A2.5 2.5 0 0 0 4 4.5v15Z" />
  ),
  chat: (
    <path d="M4 12.5C4 7.81 8.03 4 13 4s9 3.81 9 8.5-4.03 8.5-9 8.5c-1.09 0-2.13-.19-3.1-.53L4 21l1.2-4.02A8.16 8.16 0 0 1 4 12.5Z" />
  ),
  star: (
    <path d="M12 2.5l2.9 6.06 6.6.86-4.85 4.63 1.24 6.6L12 17.6l-5.89 3.05 1.24-6.6-4.85-4.63 6.6-.86L12 2.5Z" />
  ),
};

// Nested sub-panels reached from a top-level menu item, keyed by the `view`
// state below — each swaps the panel's contents in place (same pattern as
// the original Settings/Change-Password nesting) rather than routing away,
// so drilling into "Manage Doctors", "Upload Documents", or "Appointments"
// never leaves the overlay.
const SUBVIEWS = {
  appointments: {
    heading: "Appointments",
    icon: "calendar",
    description: "Book a new visit or manage your upcoming appointments.",
    items: [
      { to: "/patient/book", icon: "calendar", section: "Book", label: "Book appointment" },
      { to: "/patient/appointments", icon: "clock", section: "Upcoming", label: "Upcoming appointments" },
      { to: "/patient/appointments/history", icon: "log", section: "History", label: "Appointment history" },
    ],
  },
  manageDoctors: {
    heading: "Manage Doctors",
    icon: "doctors",
    description: "Review your roster, import doctors via CSV, and check past ingestion runs.",
    items: [
      { to: "/admin/doctors", icon: "doctors", section: "Roster", label: "Doctors" },
      { to: "/admin/doctors/import", icon: "upload", section: "Import", label: "Import doctors" },
      { to: "/admin/doctors/ingestion-log", icon: "log", section: "History", label: "Ingestion log" },
    ],
  },
  uploadDocuments: {
    heading: "Upload Documents",
    icon: "upload",
    description: "Keep the AI chatbot's knowledge base current for your patients.",
    items: [{ to: "/admin/knowledge-base", icon: "book", section: "Knowledge Base", label: "Knowledge base" }],
  },
  settings: {
    heading: "Settings",
    icon: "settings",
    description: "Manage your profile and account security.",
    // "View Profile" isn't listed here since its target path is role-aware
    // (/admin/profile vs /patient/profile) — it's prepended at render time,
    // see subviewItems below.
    items: [{ to: "/change-password", icon: "lock", section: "Security", label: "Change password" }],
  },
};

function ItemIcon({ name }) {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}

// Every menu row shares this shape — an icon in a chip, a label, and (for
// rows that open a nested sub-panel rather than navigating) a trailing
// chevron so it reads as "opens more," not "goes somewhere else."
function MenuRow({ icon, children, chevron }) {
  return (
    <>
      <span className={styles.menuIcon} aria-hidden="true">
        <ItemIcon name={icon} />
      </span>
      <span className={styles.menuLabel}>{children}</span>
      {chevron && (
        <span className={styles.menuChevron} aria-hidden="true">
          <ItemIcon name="chevron" />
        </span>
      )}
    </>
  );
}

// Account launcher rendered as the header's rightmost item (see AppHeader.jsx),
// opening a fixed, full-height left-docked side panel (overlay + dimmed
// backdrop, not a push layout — pushing the whole app's content sideways for
// an account menu would fight every page's own responsive layout far more
// than an overlay does). The panel/backdrop are position: fixed regardless of
// where this trigger lives in the DOM, so they stay correct on scroll and
// independent of any page's own stacking context.
export default function SettingsMenu() {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  // "account" is the default menu; any other value is a key into SUBVIEWS
  // above, reached via that item's button in the account menu — swapped in
  // place rather than routed to, so drilling in never leaves this overlay.
  const [view, setView] = useState("account");
  // Which way the panel body should slide in: "forward" drilling into a
  // sub-panel, "back" returning to the account view — purely cosmetic, so a
  // wrong guess never breaks anything, only looks slightly off.
  const [direction, setDirection] = useState("forward");
  const [account, setAccount] = useState(null);
  const panelRef = useRef(null);
  const launcherRef = useRef(null);

  // Re-fetches whenever the signed-in identity changes (id or role), not just once on
  // mount — this component lives at the app root for the whole session (never
  // unmounts across login/logout), so without this a patient's name/email would keep
  // showing after logging out and back in as a different admin, until a full reload.
  // Clearing first means the previous account never flashes for the new identity while
  // the fetch for it is still in flight.
  useEffect(() => {
    setAccount(null);
    if (!user) return;
    fetchMyAccount()
      .then(setAccount)
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, user?.role]);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (e) => {
      if (
        panelRef.current &&
        !panelRef.current.contains(e.target) &&
        launcherRef.current &&
        !launcherRef.current.contains(e.target)
      ) {
        setOpen(false);
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  // Always land back on the main account view next time the panel opens,
  // rather than reopening wherever it was left (e.g. mid-Settings).
  useEffect(() => {
    if (!open) setView("account");
  }, [open]);

  if (!user) return null;

  const initial = (account?.full_name || user.role || "?").trim().charAt(0).toUpperCase();
  const profilePath = user.role === "admin" ? "/admin/profile" : "/patient/profile";
  // "View Profile" lives inside the Settings sub-panel — its target is role-aware,
  // so it's prepended here rather than baked into the static SUBVIEWS config above.
  const subviewItems =
    view === "settings"
      ? [{ to: profilePath, icon: "profile", section: "Profile", label: "View profile" }, ...SUBVIEWS.settings.items]
      : SUBVIEWS[view]?.items;

  return (
    <>
      <div className={`${styles.backdrop} ${open ? styles.open : ""}`} onClick={() => setOpen(false)} aria-hidden="true" />

      <div className={`${styles.panel} ${open ? styles.open : ""}`} role="menu" aria-label="Account menu" aria-hidden={!open} ref={panelRef}>
        <div className={styles.panelHeader}>
          {view === "account" ? (
            <span className={styles.panelHeading}>Account</span>
          ) : (
            <>
              <button
                type="button"
                className={styles.backBtn}
                onClick={() => {
                  setDirection("back");
                  setView("account");
                }}
                aria-label="Back to account menu"
              >
                <ItemIcon name="back" />
              </button>
              <span className={styles.panelHeading}>{SUBVIEWS[view].heading}</span>
            </>
          )}
          <button type="button" className={styles.closeBtn} onClick={() => setOpen(false)} aria-label="Close account menu">
            ✕
          </button>
        </div>

        <div
          key={view}
          className={`${styles.panelBody} ${direction === "forward" ? styles.slideForward : styles.slideBack}`}
        >
          {view === "account" ? (
            <>
              <div className={styles.accountInfo}>
                <span className={styles.accountAvatar} aria-hidden="true">
                  {initial}
                </span>
                <span className={styles.accountText}>
                  <span className={styles.accountName}>{account?.full_name || "…"}</span>
                  <span className={styles.accountEmail}>{account?.email || ""}</span>
                </span>
              </div>

              <nav className={styles.menuList} aria-label="Account actions">
                {user.role === "patient" && (
                  <>
                    <button
                      type="button"
                      className={styles.menuItem}
                      role="menuitem"
                      onClick={() => {
                        setDirection("forward");
                        setView("appointments");
                      }}
                    >
                      <MenuRow icon="calendar" chevron>
                        Appointments
                      </MenuRow>
                    </button>
                    <Link to="/patient/chat" className={styles.menuItem} role="menuitem" onClick={() => setOpen(false)}>
                      <MenuRow icon="chat">AI Assistant</MenuRow>
                    </Link>
                  </>
                )}
                {user.role === "admin" && (
                  <>
                    <button
                      type="button"
                      className={styles.menuItem}
                      role="menuitem"
                      onClick={() => {
                        setDirection("forward");
                        setView("manageDoctors");
                      }}
                    >
                      <MenuRow icon="doctors" chevron>
                        Manage Doctors
                      </MenuRow>
                    </button>
                    <button
                      type="button"
                      className={styles.menuItem}
                      role="menuitem"
                      onClick={() => {
                        setDirection("forward");
                        setView("uploadDocuments");
                      }}
                    >
                      <MenuRow icon="upload" chevron>
                        Upload Documents
                      </MenuRow>
                    </button>
                    <Link to="/admin/feedback" className={styles.menuItem} role="menuitem" onClick={() => setOpen(false)}>
                      <MenuRow icon="star">Patient Feedback</MenuRow>
                    </Link>
                  </>
                )}
                <button
                  type="button"
                  className={styles.menuItem}
                  role="menuitem"
                  onClick={() => {
                    setDirection("forward");
                    setView("settings");
                  }}
                >
                  <MenuRow icon="settings" chevron>
                    Settings
                  </MenuRow>
                </button>
                <button
                  type="button"
                  className={`${styles.menuItem} ${styles.logoutItem}`}
                  role="menuitem"
                  onClick={() => {
                    setOpen(false);
                    logout();
                  }}
                >
                  <MenuRow icon="logout">Log Out</MenuRow>
                </button>
              </nav>
            </>
          ) : (
            <>
              <div className={styles.subViewIntro}>
                <span className={styles.subViewIcon} aria-hidden="true">
                  <ItemIcon name={SUBVIEWS[view].icon} />
                </span>
                <p className={styles.subViewDescription}>{SUBVIEWS[view].description}</p>
              </div>
              <nav className={styles.subMenuList} aria-label={SUBVIEWS[view].heading}>
                {subviewItems.map((item) => (
                  <div key={item.to} className={styles.subMenuGroup}>
                    <span className={styles.subMenuSectionLabel}>{item.section}</span>
                    <Link
                      to={item.to}
                      className={styles.subMenuButton}
                      role="menuitem"
                      onClick={() => setOpen(false)}
                    >
                      <ItemIcon name={item.icon} />
                      <span className={styles.subMenuButtonLabel}>{item.label}</span>
                    </Link>
                  </div>
                ))}
              </nav>
            </>
          )}
        </div>
      </div>

      <div className={styles.launcher} ref={launcherRef}>
        <button
          type="button"
          className={styles.avatarBtn}
          onClick={() => setOpen((v) => !v)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={open ? "Close account menu" : "Open account menu"}
        >
          {initial}
        </button>
      </div>
    </>
  );
}
