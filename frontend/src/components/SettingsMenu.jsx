import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { fetchMyAccount } from "../api/auth";
import { useAuth } from "../auth/AuthContext";
import ThemeToggle from "./ThemeToggle";
import { useTheme } from "../theme/ThemeContext";
import styles from "./SettingsMenu.module.css";

const ICONS = {
  profile: <path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4.5 20.5c0-4.14 3.36-7.5 7.5-7.5s7.5 3.36 7.5 7.5" />,
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
  theme: <path d="M12 3a9 9 0 1 0 9 9c0-.46-.03-.92-.09-1.36A5.5 5.5 0 0 1 12 3Z" />,
};

// Nested sub-panels reached from a top-level menu item, keyed by the `view`
// state below — each swaps the panel's contents in place (same pattern as
// the original Settings/Change-Password nesting) rather than routing away,
// so drilling into "Manage Doctors", "Upload Documents", or "Appointments"
// never leaves the overlay.
const SUBVIEWS = {
  // "Book appointment" and "Upcoming appointments" moved to the persistent
  // nav (PrimaryNavDesktop/Mobile) — Appointment history is the one thing
  // left here, reached directly as a top-level link instead (see below)
  // rather than a whole subview for a single item.
  manageDoctors: {
    heading: "Manage Doctors",
    icon: "doctors",
    description: "Import doctors via CSV and check past ingestion runs.",
    items: [
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
// opening a fixed, full-height right-docked side panel (overlay + dimmed
// backdrop, not a push layout — pushing the whole app's content sideways for
// an account menu would fight every page's own responsive layout far more
// than an overlay does). The panel/backdrop are position: fixed regardless of
// where this trigger lives in the DOM, so they stay correct on scroll and
// independent of any page's own stacking context.
export default function SettingsMenu() {
  const { user, logout } = useAuth();
  const { isDark } = useTheme();
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
                  // Book appointment / Upcoming appointments / Chat now live in
                  // the persistent nav (see PrimaryNavDesktop/Mobile) —
                  // Appointment history is the one appointment-related screen
                  // that's not frequent enough to promote there, so it's a
                  // direct link here instead of a whole subview for one item.
                  <Link
                    to="/patient/appointments/history"
                    className={styles.menuItem}
                    role="menuitem"
                    onClick={() => setOpen(false)}
                  >
                    <MenuRow icon="log">Appointment History</MenuRow>
                  </Link>
                )}
                {user.role === "admin" && (
                  <>
                    {/* Doctors (roster) and Patient Feedback now live in the
                        persistent nav (see PrimaryNavDesktop/Mobile) — Manage
                        Doctors here now only covers the less-frequent CSV import
                        + ingestion history (see SUBVIEWS.manageDoctors above). */}
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
                {view === "settings" && (
                  <div className={styles.subMenuGroup}>
                    <span className={styles.subMenuSectionLabel}>Appearance</span>
                    <div className={styles.subMenuButton}>
                      <ItemIcon name="theme" />
                      <span className={styles.subMenuButtonLabelStacked}>
                        <span>Theme</span>
                        <span className={styles.subMenuButtonCaption}>
                          {isDark ? "Set to Dark by default — tap to switch to Light" : "Set to Light — tap to switch back to Dark"}
                        </span>
                      </span>
                      <ThemeToggle />
                    </div>
                  </div>
                )}
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
