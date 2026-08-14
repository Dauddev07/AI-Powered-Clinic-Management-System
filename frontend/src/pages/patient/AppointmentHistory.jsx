import { useEffect, useState } from "react";
import { fetchMyAppointmentHistory } from "../../api/patientBooking";
import { ApiError } from "../../api/client";
import StatusBadge from "../../components/StatusBadge";
import SuccessCheck from "../../components/SuccessCheck";
import EmptyState from "../../components/EmptyState";
import Skeleton from "../../components/Skeleton";
import { useReveal, revealDelayClass } from "../../hooks/useReveal";
import { formatClinicDateTime as formatDateTime } from "../../utils/formatDateTime";
import styles from "./PatientScreens.module.css";
import historyStyles from "./AppointmentHistory.module.css";

// tone/label drive StatusBadge; nodeTone/cardTone pick the timeline dot + card
// accent color family — cancelled and missed (no_show) share the same "error"
// family (both mean the visit never happened) while expired is a neutral
// non-outcome. "no_show" is the status value stored in the DB (see
// app/services/booking_engine.confirm_visit) but "Missed" is what the patient
// actually sees — it's what they answer in the visit-confirmation prompt.
const STATUS_META = {
  completed: { tone: "success", label: "Completed", accent: "success" },
  cancelled: { tone: "error", label: "Cancelled", accent: "error" },
  no_show: { tone: "error", label: "Missed", accent: "error" },
  expired: { tone: "neutral", label: "Expired", accent: "neutral" },
};

function formatMonthHeading(iso, timeZone) {
  return new Date(iso).toLocaleString(undefined, { month: "long", year: "numeric", timeZone });
}

// "Dr. Muhammad Qureshi" -> "MQ" — strips a leading "Dr."/"Dr" title before
// taking initials, so the avatar isn't just "D" for every single doctor.
function initialsFor(fullName) {
  const cleaned = (fullName || "").replace(/^dr\.?\s+/i, "").trim();
  const parts = cleaned.split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  return (parts[0][0] + (parts[1]?.[0] || "")).toUpperCase();
}

// Groups the (already newest-first) list by clinic-local calendar month
// without re-sorting anything — a Map preserves first-insertion key order,
// which for an already-sorted list is exactly month-descending.
function groupByMonth(appointments, timeZone) {
  const groups = new Map();
  for (const appointment of appointments) {
    const key = formatMonthHeading(appointment.start_utc, timeZone);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(appointment);
  }
  return Array.from(groups.entries());
}

function CalendarIcon() {
  return (
    <svg className={historyStyles.whenIcon} viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 3v3M17 3v3M4 9h16M5 6h14a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1Z" />
    </svg>
  );
}

function HistoryNodeIcon({ status }) {
  if (status === "completed") {
    return (
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M5 13l4 4L19 7" />
      </svg>
    );
  }
  if (status === "cancelled" || status === "no_show") {
    return (
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M18 6 6 18M6 6l12 12" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 8v4l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
    </svg>
  );
}

// Cancelled: cancelled_at is the exact instant the patient (or clinic) cancelled it.
// Completed: the appointment's own scheduled end_utc — the visit's real date/time —
// rather than updated_at, which only reflects whenever the lazy auto-complete job
// happened to run and flip the status, not when the visit itself actually happened.
function resolvedAtFor(appointment) {
  if (appointment.status === "cancelled" && appointment.cancelled_at) return appointment.cancelled_at;
  if (appointment.status === "completed") return appointment.end_utc;
  return appointment.updated_at;
}

// No "Expired" filter: nothing in the system sets that status anymore (see
// app/models/appointment.py) — not worth a filter chip for a status that can no
// longer occur. "Missed" (no_show), however, is now a real, patient-driven outcome
// (see VisitConfirmationGate) and does get its own chip.
const FILTERS = [
  { key: "all", label: "All" },
  { key: "completed", label: "Completed" },
  { key: "cancelled", label: "Cancelled" },
  { key: "no_show", label: "Missed" },
];

export default function AppointmentHistory() {
  const revealRef = useReveal();
  const [appointments, setAppointments] = useState(null);
  const [clinicTimezone, setClinicTimezone] = useState(null);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    fetchMyAppointmentHistory()
      .then((data) => {
        setAppointments(data.appointments);
        setClinicTimezone(data.clinic_timezone);
      })
      .catch((err) =>
        setError(err instanceof ApiError ? err.detail || err.message : "Could not load appointment history."),
      );
  }, []);

  // Counts are always over the full unfiltered list, not the currently-filtered
  // view — otherwise picking a filter would immediately zero out every other
  // chip's own count, making it impossible to switch between them at a glance.
  const statusCounts = { completed: 0, cancelled: 0, no_show: 0, expired: 0 };
  for (const a of appointments || []) {
    if (a.status in statusCounts) statusCounts[a.status] += 1;
  }

  const filteredAppointments =
    appointments && statusFilter !== "all" ? appointments.filter((a) => a.status === statusFilter) : appointments;

  const monthGroups = filteredAppointments ? groupByMonth(filteredAppointments, clinicTimezone) : [];
  const activeFilterLabel = FILTERS.find((f) => f.key === statusFilter)?.label;

  return (
    <div>
      <h1 className={styles.title}>Appointment history</h1>
      <p className={styles.subtitle}>Your past and cancelled appointments, newest first.</p>

      {error && <p className={styles.errorText}>{error}</p>}

      {appointments === null && !error && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <Skeleton rows={4} />
        </div>
      )}

      {appointments && appointments.length === 0 && (
        <div className={`${styles.card} reveal`} ref={revealRef}>
          <EmptyState icon="calendar" message="No appointment history yet." />
        </div>
      )}

      {appointments && appointments.length > 0 && (
        <>
          <div className={`${historyStyles.statsRow} reveal`} ref={revealRef}>
            {[
              { key: "completed", label: "Completed", accent: "success" },
              { key: "cancelled", label: "Cancelled", accent: "error" },
            ]
              .filter((tile) => statusCounts[tile.key] > 0)
              .map((tile) => (
                <div key={tile.key} className={`${historyStyles.statTile} ${historyStyles[tile.accent]}`}>
                  <span className={historyStyles.statValue}>{statusCounts[tile.key]}</span>
                  <span className={historyStyles.statLabel}>{tile.label}</span>
                </div>
              ))}
          </div>

          <div className={historyStyles.filterBar} role="tablist" aria-label="Filter by status">
            {FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                role="tab"
                aria-selected={statusFilter === f.key}
                className={`${historyStyles.filterChip} ${statusFilter === f.key ? historyStyles.filterChipActive : ""}`}
                onClick={() => setStatusFilter(f.key)}
              >
                {f.label}
                {f.key !== "all" && statusCounts[f.key] > 0 && (
                  <span className={historyStyles.filterChipCount}>{statusCounts[f.key]}</span>
                )}
              </button>
            ))}
          </div>

          {filteredAppointments.length === 0 && (
            <div className={`${styles.card} reveal`} ref={revealRef}>
              <EmptyState icon="calendar" message={`No ${activeFilterLabel.toLowerCase()} appointments in your history.`} />
            </div>
          )}

          {monthGroups.map(([month, monthAppointments]) => (
            <div key={month} className={historyStyles.monthGroup}>
              <h2 className={historyStyles.monthHeading}>
                {month}
                <span className={historyStyles.monthCount}>{monthAppointments.length}</span>
              </h2>
              <div className={historyStyles.timeline}>
                {monthAppointments.map((a, i) => {
                  const meta = STATUS_META[a.status] || { tone: "neutral", label: a.status, accent: "neutral" };
                  return (
                    <div
                      key={a.id}
                      className={`${historyStyles.entry} reveal ${revealDelayClass(i % 5)}`}
                      ref={revealRef}
                    >
                      <span className={`${historyStyles.node} ${historyStyles[meta.accent]}`} aria-hidden="true">
                        <HistoryNodeIcon status={a.status} />
                      </span>

                      <div className={`${historyStyles.card} ${historyStyles[meta.accent]}`}>
                        <div className={historyStyles.cardTop}>
                          <div className={historyStyles.doctorRow}>
                            <span className={historyStyles.avatar} aria-hidden="true">
                              {initialsFor(a.doctor_name)}
                            </span>
                            <span className={historyStyles.doctorText}>
                              <span className={historyStyles.doctorName}>{a.doctor_name}</span>
                              <span className={historyStyles.departmentName}>{a.department_name}</span>
                            </span>
                          </div>
                          <StatusBadge tone={meta.tone} label={meta.label} />
                        </div>

                        <div className={historyStyles.whenRow}>
                          <CalendarIcon />
                          <span>{formatDateTime(a.start_utc, clinicTimezone)}</span>
                          <span className={historyStyles.relativeTime}>
                            {meta.label} {formatDateTime(resolvedAtFor(a), clinicTimezone)}
                          </span>
                        </div>

                        {a.reason && <div className={historyStyles.reason}>{a.reason}</div>}

                        {a.status === "completed" && (
                          <div className={historyStyles.completedNote}>
                            <SuccessCheck />
                            Visit completed — thanks for visiting!
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
