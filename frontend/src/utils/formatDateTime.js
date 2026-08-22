// Shared date/time formatters, consolidated out of near-identical copies that used to
// live independently in AppointmentHistory.jsx, UpcomingAppointments.jsx,
// PatientDashboard.jsx, Feedback.jsx, KnowledgeBase.jsx, and IngestionLogScreen.jsx —
// same output as before, just one place to read/change it.

// Always rendered against the clinic's own timezone, never the viewer's browser-local
// one, so a slot that reads "2pm" on Book Appointment reads the same "2pm" everywhere
// else it's shown (appointment history, upcoming list, dashboard next-appointment card).
export function formatClinicDateTime(iso, timeZone) {
  return new Date(iso).toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone,
  });
}

// The clinic-local calendar date (YYYY-MM-DD) an ISO instant falls on — used to
// scope a reschedule slot query to the SAME day as the appointment being moved
// (the backend only ever allows rescheduling within one calendar day; see
// booking_engine.py's own same-day-only rule), never the viewer's browser-local
// date, which could disagree with the clinic's date near a day boundary.
export function formatClinicDateISO(iso, timeZone) {
  return new Date(iso).toLocaleDateString("en-CA", { timeZone });
}

// Human-readable clinic-local date only, no time — e.g. "Sat, Aug 24". Used
// wherever only the DAY (not a specific time) needs to be shown, such as the
// same-day-only reschedule notice above formatClinicDateTime's own slot list.
export function formatClinicDateLabel(iso, timeZone) {
  return new Date(iso).toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    timeZone,
  });
}

// Browser-local date+time, used for admin-facing timestamps (feedback submitted at,
// KB document uploaded at, ingestion log entries) where there's no single clinic
// timezone convention to match against.
export function formatDateTimeLocal(iso) {
  return new Date(iso).toLocaleString();
}

// Browser-local date only, null-safe (an admin account may have no last-login etc.).
export function formatDateOnly(iso) {
  return iso ? new Date(iso).toLocaleDateString() : "—";
}
