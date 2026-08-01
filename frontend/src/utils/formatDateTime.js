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
