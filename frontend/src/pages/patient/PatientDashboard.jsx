import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchMyProfile } from "../../api/auth";
import { fetchMyAppointmentSummary } from "../../api/patientBooking";
import { ApiError } from "../../api/client";
import EmptyState from "../../components/EmptyState";
import FloatingChatButton from "../../components/FloatingChatButton";
import Skeleton from "../../components/Skeleton";
import StatusBadge from "../../components/StatusBadge";
import WelcomeBanner from "../../components/WelcomeBanner";
import { useCountUp } from "../../hooks/useCountUp";
import { useNow } from "../../hooks/useNow";
import { isAppointmentInProgress } from "../../utils/appointmentStatus";
import { formatClinicDateTime as formatDateTime } from "../../utils/formatDateTime";
import styles from "./PatientDashboard.module.css";

// Small tone-colored icon per count tile — turns the "0 / 0 / 0"-style row into
// something scannable at a glance instead of three bare numbers that only differ
// by their (easy-to-miss) text label underneath.
const COUNT_ICONS = {
  upcoming: { tone: "info", path: "M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" },
  inProgress: { tone: "warning", path: "M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" },
  completed: { tone: "success", path: "M5 13l4 4L19 7" },
};

function CountIcon({ kind }) {
  const meta = COUNT_ICONS[kind];
  return (
    <span className={`${styles.countIcon} ${styles[`countIcon_${meta.tone}`]}`} aria-hidden="true">
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d={meta.path} />
      </svg>
    </span>
  );
}

// Reported live: at 3:49 for a 3:45-4:00 appointment this showed "10 min"
// instead of "11 min" — Math.round flips to the lower minute as soon as
// you're more than 30s into the current clock-minute (3:49:31 -> 10.48 ->
// rounds to 10), so the display disagreed with what the clock still read.
// Math.ceil keeps it at 11 for the whole 3:49:00-3:49:59 window and only
// drops to 10 the instant the clock actually reaches 3:50.
function formatMinutesRemaining(endIso, now) {
  const minutes = Math.max(0, Math.ceil((new Date(endIso).getTime() - now.getTime()) / 60000));
  if (minutes <= 0) return "Ending now";
  if (minutes < 60) return `Ends in ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `Ends in ${hours}h${rest ? ` ${rest}m` : ""}`;
}

// The profile form that used to live here has moved to View Profile, and the
// Book/Upcoming Appointments action cards that used to sit below the welcome
// banner have moved into the fixed account menu (see SettingsMenu.jsx) — this
// page is the welcome landing spot, plus (added here) a next-appointment
// glance and simple upcoming/completed counts, both reusing GET
// /appointments/summary rather than pulling the full appointments list just
// to derive a count client-side.
export default function PatientDashboard() {
  const [fullName, setFullName] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const now = useNow();

  useEffect(() => {
    fetchMyProfile()
      .then((data) => setFullName(data.full_name))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchMyAppointmentSummary()
      .then(setSummary)
      .catch((err) =>
        setError(err instanceof ApiError ? err.detail || err.message : "Could not load your appointments."),
      );
  }, []);

  // Hoisted out of the render tree below — used both for the status badge
  // and (new) the card's own left-edge accent color, so it's computed once
  // rather than twice via a nested IIFE.
  const nextAppointmentInProgress = summary?.next_appointment
    ? isAppointmentInProgress(summary.next_appointment, now)
    : false;
  const nextCardAccentClass = !summary?.next_appointment
    ? ""
    : nextAppointmentInProgress
      ? styles.card_accentWarning
      : styles.card_accentInfo;

  const inProgressCount = nextAppointmentInProgress ? 1 : 0;
  const upcomingAnimated = useCountUp(summary ? summary.upcoming_count - inProgressCount : null);
  const inProgressAnimated = useCountUp(summary ? inProgressCount : null);
  const completedAnimated = useCountUp(summary ? summary.completed_count : null);

  return (
    <>
      <WelcomeBanner
        name={fullName}
        tagline="Use the account menu to book an appointment or manage your visits."
      />

      <div className={styles.sectionLabel}>Appointments</div>
      <div className={styles.grid}>
        {/* Reported live: this card intermittently stayed permanently invisible
            (opacity: 0, still occupying its grid slot) — the shared scroll-reveal
            IntersectionObserver (see useReveal.js) never toggled its .revealed
            class on some loads, with no visible failure mode short of the card
            never appearing at all. Always-visible above-the-fold summary data has
            no business depending on a scroll-triggered animation to begin with —
            dropped the reveal/ref treatment entirely rather than chase the exact
            observer race, same call made for the sibling card just below. */}
        <div className={`${styles.card} ${nextCardAccentClass}`}>
          <div className={styles.cardTitleRow}>
            <span className={`${styles.titleIcon} ${styles.titleIcon_info}`} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" />
              </svg>
            </span>
            <h2 className={styles.cardTitle}>Next appointment</h2>
          </div>

          {summary === null && !error && <Skeleton rows={2} />}
          {error && <p className={styles.errorText}>{error}</p>}

          {summary && summary.next_appointment && (
            <div className={styles.nextAppointment}>
              <div className={styles.nextStatusRow}>
                <StatusBadge
                  tone={nextAppointmentInProgress ? "warning" : "info"}
                  label={nextAppointmentInProgress ? "In progress" : "Upcoming"}
                  pulse={nextAppointmentInProgress}
                />
                {nextAppointmentInProgress && (
                  <span className={styles.nextEndsIn}>
                    {formatMinutesRemaining(summary.next_appointment.end_utc, now)}
                  </span>
                )}
              </div>
              <span className={styles.nextWhen}>
                {formatDateTime(summary.next_appointment.start_utc, summary.clinic_timezone)}
              </span>
              <span className={styles.nextWith}>
                {summary.next_appointment.doctor_name} · {summary.next_appointment.department_name}
              </span>
              {summary.next_appointment.reason && (
                <span className={styles.nextReason}>{summary.next_appointment.reason}</span>
              )}
              <Link to="/patient/appointments" className={styles.viewAllLink}>
                View all
              </Link>
            </div>
          )}

          {summary && !summary.next_appointment && (
            <EmptyState
              icon="calendar"
              message="No upcoming appointments — book one now."
              actionLabel="Book Appointment"
              actionTo="/patient/book"
            />
          )}
        </div>

        <div className={styles.card}>
          <div className={styles.cardTitleRow}>
            <span className={`${styles.titleIcon} ${styles.titleIcon_success}`} aria-hidden="true">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
              </svg>
            </span>
            <h2 className={styles.cardTitle}>Your appointments</h2>
          </div>

          {summary === null && !error && <Skeleton rows={1} />}

          {/* A brand-new account with zero history ever (not just none upcoming)
              would otherwise land on a flat "0 / 0 / 0" grid — the least
              welcoming thing a first-time patient could see here. Point them at
              the assistant instead, the app's fastest path to a first booking. */}
          {summary && summary.upcoming_count === 0 && summary.completed_count === 0 && (
            <EmptyState
              icon="calendar"
              message="You haven't had any appointments yet. Describe how you're feeling and our assistant will point you to the right doctor."
              actionLabel="Chat with the assistant"
              actionTo="/patient/chat"
            />
          )}

          {/* The next appointment (earliest confirmed, by start time) is the only
              one that can ever be "in progress" — auto-complete only fires once
              end_utc passes, so an ongoing appointment is still the soonest
              confirmed one. inProgressCount (hoisted above) is split out of
              upcoming_count so the three tiles stay mutually exclusive instead
              of double-counting it. */}
          {summary && (summary.upcoming_count > 0 || summary.completed_count > 0) && (
            <div className={styles.counts}>
              <div className={styles.countItem}>
                <CountIcon kind="upcoming" />
                <span className={styles.countValue}>{upcomingAnimated}</span>
                <span className={styles.countLabel}>Upcoming</span>
              </div>
              <div className={styles.countItem}>
                <CountIcon kind="inProgress" />
                <span className={styles.countValue}>{inProgressAnimated}</span>
                <span className={styles.countLabel}>In progress</span>
              </div>
              <div className={styles.countItem}>
                <CountIcon kind="completed" />
                <span className={styles.countValue}>{completedAnimated}</span>
                <span className={styles.countLabel}>Completed</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <FloatingChatButton />
    </>
  );
}
