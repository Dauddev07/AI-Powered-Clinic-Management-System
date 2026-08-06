import { Fragment, useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { fetchPublicTopRatedDoctors } from "../api/clinics";
import { useAuth } from "../auth/AuthContext";
import Logo from "../components/Logo";
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

const TAGLINE_PHRASES = [
  "Describe your symptoms.",
  "We match you to the right doctor.",
  "Book in seconds — no calls, no lines.",
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
      <p className={styles.emergencyBanner} role="note">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 3.5 22 20H2L12 3.5Z" />
          <line x1="12" y1="10" x2="12" y2="14.5" />
          <circle cx="12" cy="17.25" r="0.9" fill="currentColor" stroke="none" />
        </svg>
        Routine checkups only — not for medical emergencies.
      </p>

      <section className={styles.hero}>
        <div className={styles.heroBg} aria-hidden="true" />
        <div className={styles.content}>
          <Logo size="lg" />
          <div className={styles.tagline} aria-hidden="true">
            <div className={styles.taglineTrack}>
              {[...TAGLINE_PHRASES, ...TAGLINE_PHRASES].map((phrase, i) => (
                <Fragment key={i}>
                  <span className={styles.taglinePhrase}>{phrase}</span>
                  <span className={styles.taglineDot}>•</span>
                </Fragment>
              ))}
            </div>
          </div>
          <h1 className={styles.headline}>
            Quality care, made simple to <span className={styles.accentWord}>book</span>.
          </h1>
          <p className={styles.description}>
            Quick Check Clinic lets you find a doctor, book an appointment, and
            manage your visits — all in one calm, simple place.
          </p>
          {/* Shown only to a signed-in patient/admin who navigated here from the
              header's clinic-name link (see AppHeader.jsx) — lets them return to
              their dashboard without logging in again, rather than this page
              auto-redirecting them straight back out (the previous behavior),
              which would have made visiting it from the header pointless. */}
          {isAuthenticated && (
            <Link to={user?.role === "admin" ? "/admin" : "/patient"} className={styles.backToDashboardBtn}>
              Back to Dashboard
            </Link>
          )}
        </div>
      </section>

      <div className={styles.bandPage}>
        <section
          className={styles.section}
          aria-labelledby="departments-heading"
        >
          <span className={`${styles.eyebrow} reveal`} ref={register}>What we offer</span>
          <h2 id="departments-heading" className={`${styles.sectionTitle} reveal`} ref={register}>
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
                className={`reveal ${revealDelayClass(i % 6)}`}
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
            <span className={`${styles.eyebrow} reveal`} ref={register}>Trusted by patients</span>
            <h2 id="top-rated-heading" className={`${styles.sectionTitle} reveal`} ref={register}>
              Our top rated doctors
            </h2>
            <div className={styles.doctorGrid}>
              {topRatedDoctors.map((doctor, i) => (
                <div
                  key={doctor.doctor_id}
                  className={`reveal ${revealDelayClass(i)}`}
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
          <span className={`${styles.eyebrow} reveal`} ref={register}>Why Quick Check Clinic</span>
          <h2 id="why-us-heading" className={`${styles.sectionTitle} reveal`} ref={register}>No lines. No waiting. Just care.</h2>
          <div className={styles.featureGrid}>
            {WHY_CARDS.map((card, i) => (
              // See the Departments wrapper above for why the reveal transition
              // lives on this wrapper rather than .featureCard itself.
              <div
                className={`reveal ${revealDelayClass(i)}`}
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
          <span className={`${styles.eyebrow} reveal`} ref={register}>How it works</span>
          <h2 id="how-it-works-heading" className={`${styles.sectionTitle} reveal`} ref={register}>
            From symptoms to appointment
          </h2>
          <div className={styles.flow}>
            {STEPS.map((step, i) => (
              <div className={styles.flowStep} key={step.title}>
                <div
                  className={`${styles.flowStepInner} reveal ${revealDelayClass(i)}`}
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
          <span className={`${styles.eyebrow} reveal`} ref={register}>About us</span>
          <h2 id="about-heading" className={`${styles.sectionTitle} reveal`} ref={register}>
            About Quick Check Clinic
          </h2>
          {/* Address/phone/hours below are placeholder data — replace with the
              clinic's real details before launch. */}
          <div className={`${styles.about} reveal`} ref={register}>
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
