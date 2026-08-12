import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import Logo from "./Logo";
import NotificationBell from "./NotificationBell";
import PrimaryNavDesktop from "./PrimaryNavDesktop";
import SettingsMenu from "./SettingsMenu";
import ThemeToggle from "./ThemeToggle";
import styles from "./AppHeader.module.css";

const SECTION_IDS = ["departments-heading", "why-us-heading", "how-it-works-heading", "about-heading"];

// Public landing nav only now - the authenticated patient/admin nav lives in
// PrimaryNavDesktop, a second row rendered inside this same <header> (see
// below) so its height still folds into --app-header-height automatically.
// The brand mark (Logo) at the header's left stays the same everywhere -
// login/register, landing, and the authenticated patient/admin dashboards -
// rather than swapping to a clinic-name text for signed-in users.
export default function AppHeader() {
  const location = useLocation();
  const { isAuthenticated, user } = useAuth();
  const isLanding = location.pathname === "/";
  const showPublicNav = isLanding && !isAuthenticated;

  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [activeSection, setActiveSection] = useState(null);
  const headerRef = useRef(null);

  // Closes the collapsible nav panel on route change.
  useEffect(() => {
    setMobileNavOpen(false);
  }, [location.pathname]);

  // Publishes the header's real rendered height as a CSS var so the landing
  // hero (Landing.module.css) can size itself to exactly fill the rest of
  // the viewport, rather than guessing a fixed value that drifts whenever
  // the header's own layout changes (e.g. wrapping to two rows on mobile).
  useEffect(() => {
    const el = headerRef.current;
    if (!el) return undefined;
    const setVar = () => {
      document.documentElement.style.setProperty("--app-header-height", `${el.offsetHeight}px`);
    };
    setVar();
    const observer = new ResizeObserver(setVar);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Highlights whichever on-page section is currently in view while scrolling
  // the landing page, so the header nav reflects where the visitor actually is.
  // Tracks the last section heading that has scrolled past the sticky header
  // rather than watching a thin band around the viewport's center - a
  // center-band approach goes stale for tall sections (e.g. the department
  // grid), whose heading exits the band long before its content does.
  useEffect(() => {
    if (!showPublicNav) return undefined;
    const elements = SECTION_IDS.map((id) => document.getElementById(id)).filter(Boolean);
    if (elements.length === 0) return undefined;

    const OFFSET = 120; // roughly clears the sticky header

    let ticking = false;
    const updateActive = () => {
      ticking = false;
      // A short last section (e.g. About) can run out of page to scroll
      // before its heading ever reaches the offset line below - snap to it
      // once the page itself is scrolled to the bottom.
      const atBottom =
        window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
      if (atBottom) {
        setActiveSection(elements[elements.length - 1].id);
        return;
      }
      let current = null;
      for (const el of elements) {
        if (el.getBoundingClientRect().top - OFFSET <= 0) {
          current = el.id;
        }
      }
      setActiveSection(current);
    };

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(updateActive);
    };

    updateActive();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [showPublicNav, location.pathname]);

  const anchorLinkClass = (id) =>
    `${styles.navAnchor} ${activeSection === id ? styles.navAnchorActive : ""}`;

  const closeMobileNav = () => setMobileNavOpen(false);

  // Scrolls directly on click rather than relying solely on the location-hash
  // effect on the Landing page - that effect only fires when the hash string
  // actually changes, so clicking the same anchor twice in a row (or landing
  // on it via a different path) would otherwise silently do nothing.
  const handleAnchorClick = (id) => (e) => {
    closeMobileNav();
    const el = document.getElementById(id);
    if (el) {
      e.preventDefault();
      el.scrollIntoView({ behavior: "smooth" });
      window.history.replaceState(null, "", `/#${id}`);
    }
  };

  return (
    <header className={styles.header} ref={headerRef}>
      <div className={styles.headerTop}>
        <div className={styles.headerStart}>
          {/* Clicking this while authenticated takes the patient/admin out to
              the public landing page (see the "Back to Dashboard" button
              there, Landing.jsx) rather than back to their own dashboard,
              which PrimaryNavDesktop's own Home/Dashboard item covers. */}
          <Link to="/" className={styles.brand}>
            <Logo compact />
          </Link>
        </div>

        {isAuthenticated && (
          <div className={styles.headerEnd}>
            <ThemeToggle />
            {/* Reported live: on the chat page under 720px, BOTH PrimaryNavDesktop
                (hidden below 720px) and PrimaryNavMobile (deliberately hidden on
                the chat page itself, see its own comment — it's a full-bleed
                surface with its own bottom input bar) are gone, leaving no way
                back to the patient's actual dashboard short of the header's brand
                logo, which intentionally goes to the public landing page instead
                (see the Link above). This is the one remaining way back to Home
                in that specific gap — hidden everywhere else (CSS, see
                .chatHomeBtn) since every other screen already has one of the two
                nav surfaces above. */}
            {user?.role === "patient" && location.pathname === "/patient/chat" && (
              <Link to="/patient" className={styles.chatHomeBtn} aria-label="Home">
                <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M4 11.5 12 4l8 7.5M6 10v9a1 1 0 0 0 1 1h3v-6h4v6h3a1 1 0 0 0 1-1v-9" />
                </svg>
              </Link>
            )}
            {/* Notifications only exist for patient-initiated booking events (see
                app/services/notifications.py) - there's no admin notification type
                yet, so the bell is patient-only rather than shown for every role. */}
            {user?.role === "patient" && <NotificationBell />}
            {/* Account launcher - rightmost element in the header. */}
            <SettingsMenu />
          </div>
        )}

        {showPublicNav && (
          <div className={styles.publicActions}>
            {/* Before everything else on the right side of the landing header. */}
            <ThemeToggle />

            <button
              type="button"
              className={styles.navToggle}
              aria-label={mobileNavOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileNavOpen}
              aria-controls="site-nav"
              onClick={() => setMobileNavOpen((open) => !open)}
            >
              <span className={styles.navToggleBar} />
              <span className={styles.navToggleBar} />
              <span className={styles.navToggleBar} />
            </button>

            {mobileNavOpen && (
              <button
                type="button"
                className={styles.navBackdrop}
                aria-label="Close menu"
                onClick={() => setMobileNavOpen(false)}
              />
            )}

            <nav
              id="site-nav"
              className={`${styles.landingNav} ${mobileNavOpen ? styles.navOpen : ""}`}
              aria-label="Site"
            >
              <Link
                to="/#departments-heading"
                className={anchorLinkClass("departments-heading")}
                onClick={handleAnchorClick("departments-heading")}
              >
                Departments
              </Link>
              <Link
                to="/#why-us-heading"
                className={anchorLinkClass("why-us-heading")}
                onClick={handleAnchorClick("why-us-heading")}
              >
                Why us
              </Link>
              <Link
                to="/#how-it-works-heading"
                className={anchorLinkClass("how-it-works-heading")}
                onClick={handleAnchorClick("how-it-works-heading")}
              >
                How it works
              </Link>
              <Link
                to="/#about-heading"
                className={anchorLinkClass("about-heading")}
                onClick={handleAnchorClick("about-heading")}
              >
                About
              </Link>
            </nav>
          </div>
        )}

        {/* Neither authenticated nor the landing page (e.g. Login/Register) —
            still a global, app-wide setting, so it stays available here too
            rather than only existing on two of the app's several screens. */}
        {!isAuthenticated && !showPublicNav && (
          <div className={styles.publicActions}>
            <ThemeToggle />
          </div>
        )}
      </div>

      {/* Second row — the always-visible destinations (Book Appointment,
          Upcoming Appointments, Chat for patients; Doctors, Feedback for
          admins). Nested inside this same <header> (not a sibling element)
          so its height is included in the ResizeObserver above without any
          extra wiring. */}
      <PrimaryNavDesktop />
    </header>
  );
}
