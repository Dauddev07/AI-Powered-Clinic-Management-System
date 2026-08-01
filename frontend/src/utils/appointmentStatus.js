// "In progress" is a display-only concept — the stored status stays
// "confirmed" until the backend's auto-complete job flips it to "completed"
// once end_utc passes (see app.services.appointments.auto_complete_past_appointments).
// This just derives whether `now` currently falls inside the slot's window,
// without touching the stored status.
export function isAppointmentInProgress(appointment, now = new Date()) {
  if (!appointment || appointment.status !== "confirmed") return false;
  const start = new Date(appointment.start_utc);
  const end = new Date(appointment.end_utc);
  return now >= start && now < end;
}
