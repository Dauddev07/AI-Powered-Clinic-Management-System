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
      {/* Clips only the corner glow (see .doctorCardGlow) to the card's own
          rounded corners — kept as a separate layer rather than overflow:
          hidden on .doctorCard itself, which used to clip the top of
          .rankBadge below (it deliberately sits half outside the card's top
          edge, see .rankBadge's own top: -13px). */}
      <span className={styles.doctorCardGlow} aria-hidden="true" />
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
        {/* Ambient depth layer — two softly blurred accent-tinted blobs plus a
            faint dot-grid wash across the whole hero, purely decorative
            (aria-hidden) and fixed/static (no motion, no cursor-follow) —
            deliberately restrained after the earlier particle-field/energy
            experiments were rejected as too busy. This is just quiet visual
            depth behind the same content layout, not a new interactive
            concept. */}
        <div className={styles.heroBgPattern} aria-hidden="true" />
        <div className={styles.heroGlow3} aria-hidden="true" />
        <div className={styles.heroGlow1} aria-hidden="true" />
        <div className={styles.heroGlow2} aria-hidden="true" />

        {/* Instructed live to add a genuine background "3D" treatment
            (a first pass using small floating gradient spheres was
            rejected as "3D shapes", not a 3D backdrop) — a perspective
            grid plane pinned to the hero's floor, tilted back via
            rotateX so it reads as a surface receding into depth, masked
            so it fades to nothing before it reaches the content. Built
            from the app's own border/accent tokens so it holds up in
            both themes with no separate light/dark styling needed. */}
        <div className={styles.heroGridFloor} aria-hidden="true" />
        {/* Two-column layout: text left, a graphic "visual" right. No real
            photography is used (nothing to license/fabricate) — see the
            heroVisual block below for the booking-app mockup that replaced
            it, instructed live from a supplied reference. */}
        <div className={styles.heroCard}>
          <div className={styles.heroTop}>
            <div className={styles.heroContent}>
              <span className={styles.heroEyebrow}>
                <span className={styles.heroEyebrowDot} aria-hidden="true" />
                Online booking, made for patients
              </span>
              <h1 className={styles.headline}>
                Quality care, made simple to <span className={styles.accentWord}>book</span>.
              </h1>
              <p className={styles.description}>
                Quick Check Clinic lets you find a doctor, book an appointment, and
                manage your visits — all in one calm, simple place.
              </p>
              {/* Extra persuasive copy, instructed live to be prose (full
                  sentences), not another bullet/points list — the highlights
                  row below already covers the quick facts, this is the case
                  for booking now rather than waiting. Grounded in real
                  mechanics of the booking system (slots are a finite,
                  first-come resource; doctors carry real patient ratings —
                  see the "Our top rated doctors" section) rather than an
                  invented urgency device like a countdown or a fake
                  "X people viewing this" claim. */}
              <p className={styles.urgency}>
                Every slot shown here is real, and <strong>once it's taken it's gone</strong> —
                the doctor and time you want today may not be open tomorrow. Every
                doctor on Quick Check Clinic is <strong>rated by patients who've
                already been seen</strong>, so you can book with confidence instead of
                guessing who to trust.
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
                    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M9 20a4 4 0 0 1 8 0" />
                      <circle cx="13" cy="8" r="3.2" />
                      <path d="M4 9h4M6 7v4" />
                    </svg>
                    Register
                  </Link>
                  <Link to="/login" className={styles.heroCtaSecondary}>
                    <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3" />
                      <path d="M11 8l4 4-4 4M15 12H3" />
                    </svg>
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
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={styles.backToDashboardArrow}>
                    <path d="M19 12H5M11 6l-6 6 6 6" />
                  </svg>
                  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" />
                    <rect x="13.5" y="3.5" width="7" height="7" rx="1.5" />
                    <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" />
                    <rect x="13.5" y="13.5" width="7" height="7" rx="1.5" />
                  </svg>
                  Back to Dashboard
                </Link>
              )}
            </div>

            {/* Instructed live to match a supplied reference: a booking-app
                phone mockup with a soft circular backdrop, two small desk
                props (a calendar, a plant), and floating feature cards —
                replaces the earlier photo-in-a-circle treatment. Nothing
                here is real photography; the "screen" content is a static
                mockup of the app's own booking flow (slot list + confirm),
                not a live component, and the floating cards restate real
                capabilities (booking, live slots, patient ratings) rather
                than being decoration for its own sake. */}
            <div className={styles.heroVisual} aria-hidden="true">
              <span className={styles.heroVisualBackdrop} />
              {/* Instructed live to drop the dashed ring + connecting-dot
                  "thread" lines that used to surround the phone in favor of
                  one clean complete circle — and then, after trying an
                  animated light traveling around it, instructed live again
                  to remove that motion too and leave the ring static. */}
              <span className={styles.heroVisualRingFull} />
              <span className={styles.heroVisualDots} />

              {/* Desk calendar prop — instructed live to simplify: down to
                  three spiral rings, one neutral header band, a plain grid,
                  and a single highlighted date, instead of the busier
                  five-ring / multi-colored-dot version. */}
              <svg className={styles.propCalendar} viewBox="0 0 120 86" fill="none">
                {[24, 60, 96].map((cx) => (
                  <g key={cx}>
                    <line className={styles.propCalendarRingLink} x1={cx} y1="10" x2={cx} y2="18" />
                    <circle className={styles.propCalendarRing} cx={cx} cy="9" r="4.5" />
                  </g>
                ))}
                <path className={styles.propCalendarHead} d="M16 14h88a6 6 0 0 1 6 6v8H10v-8a6 6 0 0 1 6-6Z" />
                <path className={styles.propCalendarBody} d="M10 22h100v52a6 6 0 0 1-6 6H16a6 6 0 0 1-6-6Z" />
                {[43, 67, 91].map((x) => (
                  <line key={x} className={styles.propCalendarGridLine} x1={x} y1="34" x2={x} y2="74" />
                ))}
                {[50, 66] .map((y) => (
                  <line key={y} className={styles.propCalendarGridLine} x1="14" y1={y} x2="106" y2={y} />
                ))}
                <rect className={styles.propCalendarCellActive} x="65" y="41" width="24" height="18" rx="3" />
                <path className={styles.propCalendarCheck} d="M70.5 49.5l4 4 8-9" />
              </svg>

              {/* Potted plant prop — gradient-shaded leaves (per-leaf tone,
                  darkest at the back) and a pot with a rim highlight, redrawn
                  from the earlier flat-fill version for more depth. */}
              <svg className={styles.propPlant} viewBox="0 0 84 112" fill="none">
                <defs>
                  <linearGradient id="heroPlantLeafGradA" x1="0" y1="0" x2="1" y2="1">
                    <stop className={styles.plantStopDark} offset="0%" />
                    <stop className={styles.plantStopMid} offset="100%" />
                  </linearGradient>
                  <linearGradient id="heroPlantLeafGradB" x1="1" y1="0" x2="0" y2="1">
                    <stop className={styles.plantStopMid} offset="0%" />
                    <stop className={styles.plantStopLight} offset="100%" />
                  </linearGradient>
                  <linearGradient id="heroPlantPotGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop className={styles.plantPotStopLight} offset="0%" />
                    <stop className={styles.plantPotStopDark} offset="100%" />
                  </linearGradient>
                </defs>
                <ellipse className={styles.propPlantShadow} cx="42" cy="108" rx="26" ry="4" />
                {/* Five blades fanning from one base point, outer two painted
                    first (read as furthest back), the tall center blade
                    painted last (frontmost) — gives the cluster depth
                    instead of the previous two-leaf silhouette. */}
                <path fill="url(#heroPlantLeafGradA)" d="M42,84C32,72 14,66 6,42C24,46 36,60 40,80Z" />
                <path fill="url(#heroPlantLeafGradA)" d="M42,84C52,72 70,66 78,42C60,46 48,60 44,80Z" />
                <path fill="url(#heroPlantLeafGradB)" d="M42,84C36,62 24,42 18,16C34,24 40,50 40,80Z" />
                <path fill="url(#heroPlantLeafGradB)" d="M42,84C48,62 60,42 66,16C50,24 44,50 44,80Z" />
                <path fill="url(#heroPlantLeafGradB)" d="M42,84C42,58 40,30 42,6C46,30 46,58 42,84Z" />
                <path className={styles.propPlantPotRim} d="M14 82h56l-3 8H17Z" />
                <path fill="url(#heroPlantPotGrad)" d="M17 90h50l-6 30a6 6 0 0 1-6 5H29a6 6 0 0 1-6-5Z" />
                {/* A subtle horizontal groove near the pot's rim — the kind
                    of thrown-clay banding real terracotta pots have, a
                    small extra bit of "designed" detail rather than a flat
                    tapered block. */}
                <path className={styles.propPlantPotGroove} d="M18.5 97.5h47" />
              </svg>

              <div className={styles.phoneMock}>
                <span className={styles.phoneNotch} />
                <div className={styles.phoneScreen}>
                  <div className={styles.phoneScreenHeader}>
                    <span className={styles.phoneScreenIcon}>
                      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M8 3v3M16 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z M9 14l2 2 4-4" />
                      </svg>
                    </span>
                    <span>Book Appointment</span>
                  </div>
                  <div className={styles.phoneSlotList}>
                    <span className={styles.phoneSlot}>10:00 AM</span>
                    <span className={`${styles.phoneSlot} ${styles.phoneSlotSelected}`}>
                      11:30 AM
                      <span className={styles.phoneSlotCheck}>
                        <svg viewBox="0 0 24 24" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M5 13l4 4L19 7" />
                        </svg>
                      </span>
                    </span>
                    <span className={styles.phoneSlot}>2:00 PM</span>
                  </div>
                  <span className={styles.phoneConfirmBtn}>Confirm Booking</span>
                </div>
              </div>

              <div className={`${styles.floatCard} ${styles.floatCard_booking}`}>
                <span className={styles.floatCardIcon}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 3v3M16 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z M9 14l2 2 4-4" />
                  </svg>
                </span>
                <span className={styles.floatCardText}>
                  <span className={styles.floatCardTitle}>Easy Booking</span>
                  <span className={styles.floatCardDesc}>Book appointments in just a few clicks.</span>
                </span>
              </div>

              <div className={`${styles.floatCard} ${styles.floatCard_live}`}>
                <span className={styles.floatCardIcon}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="8.5" />
                    <path d="M12 7.5V12l3 2" />
                  </svg>
                </span>
                <span className={styles.floatCardText}>
                  <span className={styles.floatCardTitle}>Live Slot Updates</span>
                  <span className={styles.floatCardDesc}>See open slots and book instantly.</span>
                </span>
              </div>

              <div className={`${styles.floatCard} ${styles.floatCard_trust}`}>
                <span className={styles.floatCardIcon}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 20.5s-7.2-4.4-9.6-8.6C.8 8.4 3 5 6.4 5c2 0 3.6 1.4 4.1 2.4.5-1 2.1-2.4 4.1-2.4 3.4 0 5.6 3.4 4 6.9C19.2 16.1 12 20.5 12 20.5Z" />
                  </svg>
                </span>
                <span className={styles.floatCardText}>
                  <span className={styles.floatCardTitle}>Patient First</span>
                  <span className={styles.floatCardDesc}>Every doctor is rated by patients.</span>
                </span>
              </div>
            </div>
          </div>
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
            {DEPARTMENTS.map((dept, i) => {
              // Booking requires a patient account — the useful next step from any
              // department card for a signed-out visitor is creating one. A signed-in
              // patient instead jumps straight to Book Appointment pre-filtered to
              // this department (matched by name against the real backend department
              // list — see BookAppointment.jsx's ?department= handling). A signed-in
              // admin has no booking flow of their own, so the card is deliberately
              // not a link at all for them — hover still shows (same .deptCard styles),
              // it just goes nowhere on click, and the "clickable link" arrow hint
              // (.deptArrow) is dropped since it isn't one for that role.
              const isAdmin = isAuthenticated && user?.role === "admin";
              const isPatient = isAuthenticated && user?.role === "patient";
              const cardIcon = (
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
              );
              const cardName = <span className={styles.deptName}>{dept.name}</span>;

              return (
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
                  {isAdmin ? (
                    <div className={styles.deptCard} title={dept.description} aria-label={dept.name}>
                      {cardIcon}
                      {cardName}
                    </div>
                  ) : (
                    <Link
                      to={isPatient ? `/patient/book?department=${encodeURIComponent(dept.name)}` : "/register"}
                      className={styles.deptCard}
                      title={dept.description}
                      aria-label={
                        isPatient
                          ? `See available slots in ${dept.name}`
                          : `See doctors in ${dept.name} — register to book`
                      }
                    >
                      {cardIcon}
                      {cardName}
                      {/* Only appears on hover/focus (see .deptArrow) — signals the
                          card is a clickable link rather than a static tile,
                          without adding a permanently-visible second element to
                          every one of the 12 cards. */}
                      <span className={styles.deptArrow} aria-hidden="true">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M5 12h14M13 6l6 6-6 6" />
                        </svg>
                      </span>
                    </Link>
                  )}
                </div>
              );
            })}
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
