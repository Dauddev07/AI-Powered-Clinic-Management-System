import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { fetchPublicTopRatedDoctors } from "../api/clinics";
import { useAuth } from "../auth/AuthContext";
import StarIcon from "../components/StarIcon";
import { useReveal, revealDelayClass } from "../hooks/useReveal";
import { DEPARTMENTS } from "./landingDepartments";
import styles from "./Landing.module.css";

// Rank-badge tone per position — gold/silver/bronze is the universal "top 3"
// convention, so it reads at a glance without needing the "#1/#2/#3" label to
// do all the work on its own. Mirrors AdminHome's own top-rated-doctors card.
const RANK_TONES = ["gold", "silver", "bronze"];

function getInitials(fullName) {
  const parts = fullName.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  const first = parts[0][0] || "";
  const last = parts.length > 1 ? parts[parts.length - 1][0] || "" : "";
  return (first + last).toUpperCase();
}

function DoctorStars({ rating }) {
  const rounded = Math.round(rating);
  return (
    <span className={styles.doctorStars} aria-label={`${rating} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <StarIcon
          key={n}
          filled={rounded >= n}
          size={15}
          className={rounded >= n ? styles.starFilled : styles.starEmpty}
        />
      ))}
    </span>
  );
}

function TopRatedDoctorCard({ doctor, rank }) {
  const tone = RANK_TONES[rank] || "bronze";
  return (
    <div className={`${styles.doctorCard} ${styles[`doctorCard_${tone}`]}`}>
      <span className={`${styles.rankBadge} ${styles[`rankBadge_${tone}`]}`}>#{rank + 1}</span>
      <div className={styles.doctorCardHeader}>
        <span className={styles.doctorAvatar} aria-hidden="true">
          {getInitials(doctor.doctor_name)}
        </span>
        <div className={styles.doctorCardHeaderText}>
          <div className={styles.doctorName}>{doctor.doctor_name}</div>
          <div className={styles.doctorDepartment}>{doctor.department_name}</div>
        </div>
      </div>
      <div className={styles.doctorRatingRow}>
        <DoctorStars rating={doctor.average_rating} />
        <span className={styles.doctorRatingValue}>{doctor.average_rating.toFixed(1)}</span>
      </div>
      <div className={styles.doctorStatRow}>
        <div className={styles.doctorStat}>
          <span className={styles.doctorStatValue}>{doctor.rating_count}</span>
          <span className={styles.doctorStatLabel}>{doctor.rating_count === 1 ? "rating" : "ratings"}</span>
        </div>
        <div className={styles.doctorStatDivider} aria-hidden="true" />
        <div className={styles.doctorStat}>
          <span className={styles.doctorStatValue}>{doctor.visit_count}</span>
          <span className={styles.doctorStatLabel}>{doctor.visit_count === 1 ? "visit" : "visits"}</span>
        </div>
      </div>
      <div className={styles.doctorClinic}>{doctor.clinic_name}</div>
    </div>
  );
}

const WHY_CARDS = [
  {
    title: "Skip the waiting room",
    description:
      "No more standing in line just to find out if a doctor is free. Book your slot online in seconds and walk in right on time for your appointment.",
    icon: "M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  },
  {
    title: "See real availability",
    description:
      "Every slot you see is live — filtered by department, doctor, date, or time of day — so you're only ever booking against doctors who are actually available.",
    icon: "M8 3v3M16 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1ZM9 13.5l1.8 1.8L15 11",
  },
  {
    title: "Built for speed",
    description:
      "From choosing a department to confirming your appointment, the whole flow takes minutes — no hold music, no calling around, no guessing.",
    icon: "M13 2 4 14h6l-1 8 9-12h-6l1-8Z",
  },
];

// The four floating badges around the hero visual — real account/booking
// capabilities (mirrors HERO_HIGHLIGHTS' own claims below), not decoration
// picked for shape alone. One filled (the brand's own heart-pulse mark, same
// motif as Logo.jsx) and three outline, echoing the reference layout without
// copying its exact icon set. Each path is deliberately drawn small/simple
// (not borrowed from HERO_HIGHLIGHTS) since these render 3-4x larger.
const HERO_VISUAL_BADGES = [
  {
    key: "care",
    fill: true,
    icon: "M12 20.5s-7.2-4.4-9.6-8.6C.8 8.4 3 5 6.4 5c2 0 3.6 1.4 4.1 2.4.5-1 2.1-2.4 4.1-2.4 3.4 0 5.6 3.4 4 6.9C19.2 16.1 12 20.5 12 20.5Z M5.3 12h3l1.6-3.2 2 6 1.6-3.8H18.7",
  },
  {
    key: "schedule",
    fill: false,
    icon: "M8 3v3M16 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z",
  },
  {
    key: "patients",
    fill: false,
    icon: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4.5 20c0-3.6 3.4-6 7.5-6s7.5 2.4 7.5 6",
  },
  {
    key: "trust",
    fill: false,
    icon: "M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3Z M9 12l2 2 4-4",
  },
];

// A short row of real, site-specific facts (not filler) directly under the
// hero description — instructed live: the hero read as too empty once it
// went back to plain text with no card/imagery. DEPARTMENTS.length is the
// actual live count from landingDepartments.js (reported live: was labeled
// "12+" when it's an exact, known count — corrected to state it plainly);
// the rest mirror WHY_CARDS' own claims below plus the app's real account
// features, so nothing here is a new promise the rest of the page doesn't
// already back up.
const HERO_HIGHLIGHTS = [
  {
    icon: "M8 3v3M16 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z",
    label: `${DEPARTMENTS.length} departments`,
  },
  {
    icon: "M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
    label: "Live slot availability",
  },
  {
    icon: "M13 2 4 14h6l-1 8 9-12h-6l1-8Z",
    label: "Book in minutes",
  },
  {
    icon: "M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11",
    label: "Manage visits anytime",
  },
];

const STEPS = [
  {
    title: "Describe your symptoms",
    description:
      "Tell our assistant how you're feeling, and it will suggest the right department, an available doctor, and open slots for you.",
  },
  {
    title: "Browse",
    description:
      "Explore departments yourself or see available departments according to your symptoms and see open slots.",
  },
  {
    title: "Book",
    description:
      "Reserve your appointment through our assistant or manually — no phone calls needed.",
  },
  {
    title: "Manage",
    description:
      "Reschedule, cancel, or check your upcoming visits anytime from your account.",
  },
];

export default function Landing() {
  const { isAuthenticated, user } = useAuth();
  const location = useLocation();
  const register = useReveal();
  const [topRatedDoctors, setTopRatedDoctors] = useState(null);

  // Client-side route changes don't trigger the browser's native hash-scroll,
  // so a cross-page nav link (e.g. from the footer) landing here with a hash
  // needs a manual scroll-into-view once the page has rendered.
  useEffect(() => {
    if (!location.hash) return;
    const el = document.querySelector(location.hash);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  }, [location.hash]);

  // Public, unauthenticated, cross-clinic — always live, so a new rating
  // anywhere can move this list before the visitor's next reload. No error
  // state surfaced here: a failed fetch (or genuinely zero ratings yet) just
  // means the section quietly doesn't render, same as an empty array does —
  // this is a marketing page, not a dashboard that owes the visitor a status.
  useEffect(() => {
    let cancelled = false;
    fetchPublicTopRatedDoctors()
      .then((data) => {
        if (!cancelled) setTopRatedDoctors(data);
      })
      .catch(() => {
        if (!cancelled) setTopRatedDoctors([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={styles.page}>
      {/* Reported live: this had been folded into a small pill inside the
          hero content (matching a reference layout's own treatment), but
          that took real vertical space away from the actual pitch/CTA on
          short screens once it wrapped to two lines. Back to a full-width
          bar at the very top of the page, above the hero entirely, the way
          it worked before that redesign. */}
      <p className={styles.emergencyBanner} role="note">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 3.5 22 20H2L12 3.5Z" />
          <line x1="12" y1="10" x2="12" y2="14.5" />
          <circle cx="12" cy="17.25" r="0.9" fill="currentColor" stroke="none" />
        </svg>
        Routine checkups only — not for medical emergencies.
      </p>

      <section className={styles.hero}>
        {/* Instructed live to match a specific reference layout: a
            two-column card (text left, a graphic "visual" right) instead of
            the full-bleed atmospheric background this section went through
            many rounds of (gradient wash, particle field, traveling pulse
            line) — all of that is gone in favor of a clean elevated card,
            closer to a product landing page than an ambient scene. No real
            photography is used (nothing to license/fabricate) — the visual
            is a line-art panel in the app's own brand language instead: a
            large stethoscope glyph inside a soft accent-tinted circle, with
            four small floating badges (see HERO_VISUAL_BADGES) that mirror
            the app's real capabilities rather than being decoration for its
            own sake. */}
        <div className={styles.heroCard}>
          <div className={styles.heroTop}>
            <div className={styles.heroContent}>
              <h1 className={styles.headline}>
                Quality care, made simple to <span className={styles.accentWord}>book</span>.
              </h1>
              <p className={styles.description}>
                Quick Check Clinic lets you find a doctor, book an appointment, and
                manage your visits — all in one calm, simple place.
              </p>
              {/* The hero's whole job is to turn a first-time visitor into a
                  booked appointment — instructed live: this is the page's first
                  impression, and it had no call to action at all for a
                  signed-out visitor (only a "Back to Dashboard" link that only
                  ever showed for someone already logged in). Register/Log in
                  used to live in the header nav (see AppHeader.jsx) — moved here
                  instead and removed there, since the hero is now the one place
                  a signed-out visitor sees them, right under the pitch that
                  explains why they'd want to click either. */}
              {!isAuthenticated && (
                <div className={styles.heroCtaRow}>
                  <Link to="/register" className={styles.heroCtaPrimary}>
                    Register
                  </Link>
                  <Link to="/login" className={styles.heroCtaSecondary}>
                    Log in
                  </Link>
                </div>
              )}
              {/* Shown only to a signed-in patient/admin who navigated here from
                  the header's clinic-name link (see AppHeader.jsx) — lets them
                  return to their dashboard without logging in again, rather than
                  this page auto-redirecting them straight back out (the previous
                  behavior), which would have made visiting it from the header
                  pointless. */}
              {isAuthenticated && (
                <Link to={user?.role === "admin" ? "/admin" : "/patient"} className={styles.backToDashboardBtn}>
                  Back to Dashboard
                </Link>
              )}
            </div>

            <div className={styles.heroVisual} aria-hidden="true">
              <span className={styles.heroVisualDots} />
              <div className={styles.heroVisualCircle}>
                {/* Instructed live: no real photograph is available in the
                    project (checked src/assets and public — nothing usable),
                    and fetching/fabricating an external stock-photo URL isn't
                    something to guess at. Rebuilt as a proper shaded/
                    gradient illustration instead of the earlier thin
                    single-stroke icon. Reported live (2nd report): the first
                    version — a large chest piece centered directly under a
                    symmetric V of tubing — read as a medallion on a ribbon,
                    not a stethoscope. Redrawn asymmetrically: the chest
                    piece is smaller and sits off to one side, connected by
                    an explicit rigid stem (not just the flexible tube
                    fading straight into it), the way a real stethoscope
                    actually hangs rather than dangling dead-center. */}
                <svg viewBox="0 0 160 200" width="152" height="190" className={styles.stethoscopeSvg}>
                  <defs>
                    <linearGradient id="stethTubeGrad" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" className={styles.stethTubeStart} />
                      <stop offset="100%" className={styles.stethTubeEnd} />
                    </linearGradient>
                    <radialGradient id="stethChestGrad" cx="34%" cy="30%" r="75%">
                      <stop offset="0%" className={styles.stethChestStart} />
                      <stop offset="55%" className={styles.stethChestMid} />
                      <stop offset="100%" className={styles.stethChestEnd} />
                    </radialGradient>
                  </defs>
                  {/* Soft contact shadow beneath the chest piece — grounds it
                      instead of looking pasted flat onto the circle. */}
                  <ellipse cx="112" cy="176" rx="24" ry="7" className={styles.stethShadow} />
                  {/* Tube — two branches from the ear tips merging into a
                      single stem, then curving down and to the right toward
                      the chest piece (asymmetric, not a mirrored V) so the
                      whole silhouette reads as something draped rather than
                      a hung medal. */}
                  <path
                    d="M42 22 C 26 52 30 84 54 100 C 68 109 74 118 74 132 C 74 148 84 158 100 163"
                    fill="none"
                    className={styles.stethTube}
                  />
                  <path
                    d="M112 20 C 130 46 126 76 100 94 C 84 105 76 116 76 130"
                    fill="none"
                    className={styles.stethTube}
                  />
                  {/* Glossy highlight riding one branch only — real tubing
                      catches light on one side, not symmetrically. */}
                  <path
                    d="M46 28 C 32 54 35 80 54 95"
                    fill="none"
                    className={styles.stethTubeHighlight}
                  />
                  {/* Binaural spring + ear tips */}
                  <rect x="34" y="12" width="7" height="15" rx="3.5" className={styles.stethMetal} transform="rotate(-16 37.5 19.5)" />
                  <rect x="116" y="10" width="7" height="15" rx="3.5" className={styles.stethMetal} transform="rotate(16 119.5 17.5)" />
                  {/* Rigid stem connecting the flexible tube to the chest
                      piece — a straight thick segment, deliberately reading
                      as stiffer than the curved tube above it. */}
                  <path d="M100 163 L109 170" fill="none" className={styles.stethStem} />
                  {/* Chest piece — smaller and off-center, not a centered
                      medallion. */}
                  <circle cx="115" cy="176" r="21" fill="url(#stethChestGrad)" className={styles.stethChestRing} />
                  <circle cx="115" cy="176" r="12" fill="none" className={styles.stethChestInner} />
                  <circle cx="108" cy="169" r="3.5" className={styles.stethChestSheen} />
                </svg>
              </div>
              {HERO_VISUAL_BADGES.map((badge) => (
                <span
                  key={badge.key}
                  className={`${styles.heroVisualBadge} ${styles[`heroVisualBadge_${badge.key}`]} ${badge.fill ? styles.heroVisualBadge_fill : ""}`}
                >
                  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d={badge.icon} />
                  </svg>
                </span>
              ))}
            </div>
          </div>

          <ul className={styles.heroHighlights}>
            {HERO_HIGHLIGHTS.map((item) => (
              <li key={item.label} className={styles.heroHighlight}>
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d={item.icon} />
                </svg>
                {item.label}
              </li>
            ))}
          </ul>
        </div>
      </section>

      <div className={styles.bandPage}>
        <section
          className={styles.section}
          aria-labelledby="departments-heading"
        >
          <span className={`${styles.eyebrow} reveal ${styles.revealBlur}`} ref={register}>What we offer</span>
          <h2 id="departments-heading" className={`${styles.sectionTitle} reveal ${styles.revealBlur}`} ref={register}>
            Departments
          </h2>
          <div className={styles.deptGrid}>
            {DEPARTMENTS.map((dept, i) => (
              // The reveal fade/slide lives on this wrapper rather than .deptCard
              // itself — .deptCard already declares its own `transition` shorthand
              // for the hover lift/shadow, and a second `transition` rule on the
              // same element would replace that list outright instead of merging
              // with it, silently breaking either the hover effect or the reveal.
              <div
                key={dept.name}
                className={`reveal ${styles.revealBlur} ${revealDelayClass(i % 6)}`}
                ref={register}
              >
                {/* Booking requires an account — the useful next step from any department
                    card for a signed-out visitor is creating one. (A signed-in visitor can
                    also reach this page now, via the header's clinic-name link — see the
                    "Back to Dashboard" button above — but still has no account-scoped
                    department browsing here to link to instead, so this stays as-is.) */}
                <Link
                  to="/register"
                  className={styles.deptCard}
                  title={dept.description}
                  aria-label={`See doctors in ${dept.name} — register to book`}
                >
                  <span className={styles.deptIcon} aria-hidden="true">
                    <svg
                      viewBox="0 0 24 24"
                      width="22"
                      height="22"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d={dept.icon} />
                    </svg>
                  </span>
                  <span className={styles.deptName}>{dept.name}</span>
                </Link>
              </div>
            ))}
          </div>
        </section>
      </div>

      {topRatedDoctors && topRatedDoctors.length > 0 && (
        <div className={styles.bandCard}>
          <section className={styles.section} aria-labelledby="top-rated-heading">
            <span className={`${styles.eyebrow} reveal ${styles.revealBlur}`} ref={register}>Trusted by patients</span>
            <h2 id="top-rated-heading" className={`${styles.sectionTitle} reveal ${styles.revealBlur}`} ref={register}>
              Our top rated doctors
            </h2>
            <div className={styles.doctorGrid}>
              {topRatedDoctors.map((doctor, i) => (
                <div
                  key={doctor.doctor_id}
                  className={`reveal ${styles.revealBlur} ${revealDelayClass(i)}`}
                  ref={register}
                >
                  <TopRatedDoctorCard doctor={doctor} rank={i} />
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      <div className={styles.bandCard}>
        <section className={styles.section} aria-labelledby="why-us-heading">
          <span className={`${styles.eyebrow} reveal ${styles.revealBlur}`} ref={register}>Why Quick Check Clinic</span>
          <h2 id="why-us-heading" className={`${styles.sectionTitle} reveal ${styles.revealBlur}`} ref={register}>No lines. No waiting. Just care.</h2>
          <div className={styles.featureGrid}>
            {WHY_CARDS.map((card, i) => (
              // See the Departments wrapper above for why the reveal transition
              // lives on this wrapper rather than .featureCard itself. Alternates
              // left/right slide-in (see .revealLeft/.revealRight) instead of a
              // uniform slide-up, for a little more visual variety here.
              <div
                className={`reveal ${styles.revealBlur} ${i % 2 === 0 ? styles.revealLeft : styles.revealRight} ${revealDelayClass(i)}`}
                ref={register}
                key={card.title}
              >
                <div className={styles.featureCard}>
                  <span className={styles.featureIcon} aria-hidden="true">
                    <svg
                      viewBox="0 0 24 24"
                      width="22"
                      height="22"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d={card.icon} />
                    </svg>
                  </span>
                  <h3 className={styles.featureTitle}>{card.title}</h3>
                  <p className={styles.featureDescription}>{card.description}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className={styles.bandCard}>
        <section className={styles.section} aria-labelledby="how-it-works-heading">
          <span className={`${styles.eyebrow} reveal ${styles.revealBlur}`} ref={register}>How it works</span>
          <h2 id="how-it-works-heading" className={`${styles.sectionTitle} reveal ${styles.revealBlur}`} ref={register}>
            From symptoms to appointment
          </h2>
          <div className={styles.flow}>
            {STEPS.map((step, i) => (
              <div className={styles.flowStep} key={step.title}>
                <div
                  className={`${styles.flowStepInner} reveal ${styles.revealBlur} ${i % 2 === 0 ? styles.revealLeft : styles.revealRight} ${revealDelayClass(i)}`}
                  ref={register}
                >
                  <span className={styles.flowNumber}>{i + 1}</span>
                  <h3 className={styles.flowTitle}>{step.title}</h3>
                  <p className={styles.flowDescription}>{step.description}</p>
                </div>
                {i < STEPS.length - 1 && (
                  <span className={styles.flowConnector} aria-hidden="true">
                    <svg
                      viewBox="0 0 24 24"
                      width="20"
                      height="20"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M5 12h14M13 6l6 6-6 6" />
                    </svg>
                  </span>
                )}
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className={styles.bandPage}>
        <section className={styles.section} aria-labelledby="about-heading">
          <span className={`${styles.eyebrow} reveal ${styles.revealBlur}`} ref={register}>About us</span>
          <h2 id="about-heading" className={`${styles.sectionTitle} reveal ${styles.revealBlur}`} ref={register}>
            About Quick Check Clinic
          </h2>
          {/* Address/phone/hours below are placeholder data — replace with the
              clinic's real details before launch. */}
          <div className={`${styles.about} reveal ${styles.revealBlur}`} ref={register}>
            <p>
              Quick Check Clinic is an online booking platform that connects
              patients with doctors across every department we offer. Our goal
              is simple: make booking and managing care straightforward, calm,
              and free of unnecessary back-and-forth.
            </p>
            <p className={styles.location}>
              Visit us at 123 Main Boulevard, Gulberg III, Lahore, Punjab. Open
              Monday through Sunday, 8:00 AM – 9:00 PM. For anything the app
              doesn't cover, call +92 42 111 234 567.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
