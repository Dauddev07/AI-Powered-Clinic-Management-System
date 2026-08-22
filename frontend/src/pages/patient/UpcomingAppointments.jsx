import { Fragment, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  cancelAppointment,
  fetchDepartmentsWithSlots,
  fetchMyAppointments,
  fetchSlots,
  rescheduleAppointment,
} from "../../api/patientBooking";
import { ApiError } from "../../api/client";
import StatusBadge from "../../components/StatusBadge";
import EmptyState from "../../components/EmptyState";
import Modal from "../../components/Modal";
import Skeleton from "../../components/Skeleton";
import { useToast } from "../../components/ToastContext";
import { useReveal } from "../../hooks/useReveal";
import { useNow } from "../../hooks/useNow";
import { isAppointmentInProgress } from "../../utils/appointmentStatus";
import {
  formatClinicDateISO,
  formatClinicDateLabel,
  formatClinicDateTime as formatDateTime,
} from "../../utils/formatDateTime";
import styles from "./PatientScreens.module.css";

const STATUS_TONE = {
  confirmed: "info",
  in_progress: "warning",
  completed: "success",
  cancelled: "error",
  no_show: "error",
  expired: "neutral",
};

const STATUS_LABEL = {
  confirmed: "Confirmed",
  in_progress: "In progress",
  completed: "Completed",
  cancelled: "Cancelled",
  no_show: "Missed",
  expired: "Expired",
};

export default function UpcomingAppointments() {
  const location = useLocation();
  const navigate = useNavigate();
  const revealRef = useReveal();
  const now = useNow();
  const [appointments, setAppointments] = useState(null);
  // The clinic's timezone — every displayed time is rendered against this, never the
  // browser's local timezone, so a slot booked as "2pm" on the Book Appointment page
  // (which is clinic-local too) never shows up here shifted by the viewer's own offset.
  const [clinicTimezone, setClinicTimezone] = useState(null);
  const [error, setError] = useState(null);
  const showToast = useToast();
  // Guards against React StrictMode's dev-only double-invocation of this effect
  // (mount -> cleanup -> mount again, both runs seeing the same pre-navigation
  // location.state) — without this, a fresh manual booking showed the same
  // "Appointment booked successfully" toast twice, stacked at the top.
  const bookingMessageShownRef = useRef(false);

  // Clear the navigation state once consumed, so refreshing or navigating back
  // to this page doesn't re-show the same booking confirmation.
  useEffect(() => {
    if (location.state?.bookingMessage && !bookingMessageShownRef.current) {
      bookingMessageShownRef.current = true;
      showToast(location.state.bookingMessage);
      navigate(location.pathname, { replace: true, state: {} });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [busyId, setBusyId] = useState(null);
  // Holds the full appointment object (not just an id) while the confirmation
  // modal is open, so the modal's own copy can name the doctor/time being
  // cancelled rather than a generic "are you sure?" with no specifics.
  const [pendingCancel, setPendingCancel] = useState(null);
  const [cancelError, setCancelError] = useState(null);
  // Holds { appointment, slot } while the reschedule confirmation modal is open —
  // mirrors pendingCancel above, added because picking a slot used to reschedule
  // immediately with no confirmation step at all (unlike cancel and fresh booking,
  // which both already ask first).
  const [pendingReschedule, setPendingReschedule] = useState(null);
  // Kept separate from rescheduleError below (that one is the slot-list panel's
  // own fetch-failure state) so a failed confirm doesn't also render inside the
  // panel behind the modal.
  const [rescheduleConfirmError, setRescheduleConfirmError] = useState(null);
  const [rescheduleTargetId, setRescheduleTargetId] = useState(null);
  const [rescheduleOptions, setRescheduleOptions] = useState([]);
  // True when rescheduleOptions had to fall back to other doctors' soonest slots
  // because the appointment's own doctor had nothing open — lets the panel show
  // each option's doctor name and a message explaining why, instead of silently
  // implying every listed time is still with the original doctor.
  const [rescheduleFallback, setRescheduleFallback] = useState(false);
  const [rescheduleLoading, setRescheduleLoading] = useState(false);
  const [rescheduleError, setRescheduleError] = useState(null);

  const load = async () => {
    try {
      const data = await fetchMyAppointments();
      setAppointments(data.appointments);
      setClinicTimezone(data.clinic_timezone);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Could not load appointments.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const closeReschedule = () => {
    setRescheduleTargetId(null);
    setRescheduleOptions([]);
    setRescheduleFallback(false);
    setRescheduleError(null);
  };

  const confirmCancel = async () => {
    if (!pendingCancel) return;
    const id = pendingCancel.id;
    setBusyId(id);
    setCancelError(null);
    try {
      await cancelAppointment(id);
      showToast("Appointment cancelled.");
      closeReschedule();
      setPendingCancel(null);
      await load();
    } catch (err) {
      // Kept inside the modal (not the page behind it) — this is where the
      // no-same-day-cancellation refusal shows up, and the patient shouldn't
      // lose the confirmation's context (which appointment, that they were
      // mid-cancel) while reading why it failed.
      setCancelError(err instanceof ApiError ? err.detail || err.message : "Cancel failed.");
    } finally {
      setBusyId(null);
    }
  };

  const openReschedule = async (appointment) => {
    setRescheduleTargetId(appointment.id);
    setRescheduleError(null);
    setRescheduleLoading(true);
    setRescheduleOptions([]);
    setRescheduleFallback(false);
    try {
      // Reschedule only ever moves an appointment within the SAME calendar day —
      // the backend refuses a different-day reschedule outright (see
      // booking_engine.py's own same-day-only rule) — so the slot query is scoped
      // to that one clinic-local date from the start, rather than showing every
      // upcoming day's slots and letting the patient pick one that would just get
      // rejected on confirm.
      const sameDay = formatClinicDateISO(appointment.start_utc, clinicTimezone);

      // Queried server-side by doctor_id, not filtered client-side out of a
      // generic top-20-slots-clinic-wide batch — that earlier approach could miss
      // real availability days out whenever other departments' sooner slots
      // crowded the doctor's own (later) slots out of the capped batch.
      const doctorData = await fetchSlots(undefined, {
        doctorId: appointment.doctor_id,
        dateFrom: sameDay,
        dateTo: sameDay,
      });
      const sameDoctor = doctorData.slots.filter((s) => s.is_bookable);
      if (sameDoctor.length > 0) {
        setRescheduleOptions(sameDoctor);
        return;
      }

      // Same reasoning for the department fallback: resolve the real
      // department_id and query it directly, rather than hoping the department's
      // soonest slots happened to land in an unfiltered top-20 batch.
      const departments = await fetchDepartmentsWithSlots();
      const department = departments.find((d) => d.name === appointment.department_name);
      if (department) {
        const deptData = await fetchSlots(department.id, { dateFrom: sameDay, dateTo: sameDay });
        const sameDepartment = deptData.slots.filter((s) => s.is_bookable);
        setRescheduleOptions(sameDepartment);
        setRescheduleFallback(sameDepartment.length > 0);
      }
    } catch (err) {
      setRescheduleError(err instanceof ApiError ? err.detail || err.message : "Could not load slots.");
    } finally {
      setRescheduleLoading(false);
    }
  };

  const handleReschedule = async (appointmentId, newSlotId) => {
    setBusyId(appointmentId);
    setRescheduleConfirmError(null);
    try {
      await rescheduleAppointment(appointmentId, newSlotId);
      showToast("Appointment rescheduled.");
      setPendingReschedule(null);
      closeReschedule();
      await load();
    } catch (err) {
      // Surfaced verbatim — covers the no-same-day-reschedule refusal and the
      // daily per-department booking cap, same pattern as the cancel rejection
      // above. Kept inside the confirmation modal (not the panel behind it),
      // same reasoning as cancelError above.
      setRescheduleConfirmError(err instanceof ApiError ? err.detail || err.message : "Reschedule failed.");
    } finally {
      setBusyId(null);
    }
  };

  const confirmReschedule = () => {
    if (!pendingReschedule) return;
    handleReschedule(pendingReschedule.appointment.id, pendingReschedule.slot.id);
  };

  return (
    <div>
      <h1 className={styles.title}>Upcoming appointments</h1>
      <p className={styles.subtitle}>Your confirmed appointments, soonest first.</p>
      <p className={styles.noticeBox}>
        You can cancel or reschedule your appointments from here as well — scroll the table sideways to reach those
        buttons if they're off-screen.
      </p>

      <div className={`${styles.card} reveal`} ref={revealRef}>
        {error && <p className={styles.errorText}>{error}</p>}

        {appointments === null && <Skeleton rows={3} />}

        {appointments && appointments.length === 0 && (
          <EmptyState
            icon="calendar"
            message="No upcoming appointments yet."
            actionLabel="Book an appointment"
            actionTo="/patient/book"
          />
        )}

        {appointments && appointments.length > 0 && (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Doctor</th>
                  <th>Department</th>
                  <th>Reason</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {appointments.map((a) => {
                  const effectiveStatus = isAppointmentInProgress(a, now) ? "in_progress" : a.status;
                  return (
                  <Fragment key={a.id}>
                    <tr>
                      <td data-label="When">{formatDateTime(a.start_utc, clinicTimezone)}</td>
                      <td data-label="Doctor">{a.doctor_name}</td>
                      <td data-label="Department">{a.department_name}</td>
                      <td data-label="Reason">{a.reason || "—"}</td>
                      <td data-label="Status">
                        <StatusBadge
                          tone={STATUS_TONE[effectiveStatus] || "neutral"}
                          label={STATUS_LABEL[effectiveStatus] || effectiveStatus}
                        />
                      </td>
                      <td data-label="">
                        {effectiveStatus !== "in_progress" && (
                          <div className={styles.actionsCell}>
                            <button
                              type="button"
                              className={styles.dangerBtn}
                              onClick={() => {
                                setCancelError(null);
                                setPendingCancel(a);
                              }}
                              disabled={busyId === a.id}
                            >
                              {busyId === a.id ? "Working…" : "Cancel"}
                            </button>
                            <button
                              type="button"
                              className={styles.secondaryBtn}
                              onClick={() => (rescheduleTargetId === a.id ? closeReschedule() : openReschedule(a))}
                              disabled={busyId === a.id}
                            >
                              {rescheduleTargetId === a.id ? "Close" : "Reschedule"}
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                    {rescheduleTargetId === a.id && (
                      <tr>
                        <td colSpan={6}>
                          <div className={styles.reschedulePanel}>
                            <h3>Reschedule with {a.doctor_name}</h3>
                            <p className={styles.subtitle}>
                              Appointments can only be rescheduled to a different time on the same day — showing
                              open slots for {formatClinicDateLabel(a.start_utc, clinicTimezone)} only.
                            </p>
                            {rescheduleLoading && <p className={styles.subtitle}>Loading available slots…</p>}
                            {rescheduleError && <p className={styles.errorText}>{rescheduleError}</p>}
                            {!rescheduleLoading && rescheduleOptions.length === 0 && !rescheduleError && (
                              <p className={styles.subtitle}>No open slots in {a.department_name} on this day.</p>
                            )}
                            {!rescheduleLoading && rescheduleFallback && (
                              <p className={styles.subtitle}>
                                {a.doctor_name} has nothing open this day — here are other {a.department_name}{" "}
                                doctors' open times on this same day instead:
                              </p>
                            )}
                            <div className={styles.rescheduleList}>
                              {rescheduleOptions.map((slot) => (
                                <div key={slot.id} className={styles.rescheduleOption}>
                                  <span>
                                    {formatDateTime(slot.start_utc, clinicTimezone)}
                                    {rescheduleFallback && ` · ${slot.doctor_name}`}
                                  </span>
                                  <button
                                    type="button"
                                    className={styles.primaryBtn}
                                    onClick={() => {
                                      setRescheduleConfirmError(null);
                                      setPendingReschedule({ appointment: a, slot });
                                    }}
                                    disabled={busyId === a.id}
                                  >
                                    Choose
                                  </button>
                                </div>
                              ))}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Modal open={!!pendingCancel} onClose={() => setPendingCancel(null)} title="Cancel appointment">
        <p className={styles.modalIntro}>
          Cancel your appointment with <strong>{pendingCancel?.doctor_name}</strong> on{" "}
          <strong>{pendingCancel ? formatDateTime(pendingCancel.start_utc, clinicTimezone) : ""}</strong>? This
          cannot be undone.
        </p>
        {cancelError && <p className={styles.errorText}>{cancelError}</p>}
        <div className={styles.modalActions}>
          <button
            type="button"
            className={styles.modalCancelBtn}
            onClick={() => setPendingCancel(null)}
            disabled={busyId === pendingCancel?.id}
          >
            Keep appointment
          </button>
          <button
            type="button"
            className={styles.modalDangerBtn}
            onClick={confirmCancel}
            disabled={busyId === pendingCancel?.id}
          >
            {busyId === pendingCancel?.id ? "Cancelling…" : "Cancel appointment"}
          </button>
        </div>
      </Modal>

      <Modal
        open={!!pendingReschedule}
        onClose={() => setPendingReschedule(null)}
        title="Reschedule appointment"
      >
        <p className={styles.modalIntro}>
          Reschedule your appointment with <strong>{pendingReschedule?.appointment?.doctor_name}</strong> from{" "}
          <strong>
            {pendingReschedule ? formatDateTime(pendingReschedule.appointment.start_utc, clinicTimezone) : ""}
          </strong>{" "}
          to{" "}
          <strong>
            {pendingReschedule ? formatDateTime(pendingReschedule.slot.start_utc, clinicTimezone) : ""}
          </strong>
          ?
        </p>
        {rescheduleConfirmError && <p className={styles.errorText}>{rescheduleConfirmError}</p>}
        <div className={styles.modalActions}>
          <button
            type="button"
            className={styles.modalCancelBtn}
            onClick={() => setPendingReschedule(null)}
            disabled={busyId === pendingReschedule?.appointment?.id}
          >
            Go back
          </button>
          <button
            type="button"
            className={styles.primaryBtn}
            onClick={confirmReschedule}
            disabled={busyId === pendingReschedule?.appointment?.id}
          >
            {busyId === pendingReschedule?.appointment?.id ? "Rescheduling…" : "Confirm reschedule"}
          </button>
        </div>
      </Modal>
    </div>
  );
}
