import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { confirmVisit, fetchPendingVisitConfirmations } from "../api/patientBooking";
import { ApiError } from "../api/client";
import { formatClinicDateTime as formatDateTime } from "../utils/formatDateTime";
import styles from "./VisitConfirmationGate.module.css";

// Deliberately NOT built on top of the shared Modal component: Modal always lets the
// patient dismiss it (overlay click, Escape) — this one must not, since the whole
// point is that a patient can't use the rest of the site again until they've said
// whether a past appointment actually happened (see backend's
// app.services.booking_engine.confirm_visit / GET /appointments/pending-confirmations).
//
// Mounted once in PatientLayout.jsx, above <Outlet/>, so it's checked on every entry
// into the patient area regardless of which screen they land on first.
export default function VisitConfirmationGate() {
  const [queue, setQueue] = useState(null); // null = not loaded yet, [] = nothing pending
  const [clinicTimezone, setClinicTimezone] = useState(null);
  const [step, setStep] = useState("ask"); // "ask" | "verify"
  const [pendingAnswer, setPendingAnswer] = useState(null); // true/false, set once "ask" is answered
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const contentRef = useRef(null);

  useEffect(() => {
    fetchPendingVisitConfirmations()
      .then((data) => {
        setQueue(data.appointments);
        setClinicTimezone(data.clinic_timezone);
      })
      .catch(() => setQueue([])); // fail open — never block the whole app behind a network hiccup
  }, []);

  const current = queue && queue.length > 0 ? queue[0] : null;

  useEffect(() => {
    if (!current) return undefined;
    contentRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [current]);

  if (!current) return null;

  const choose = (completed) => {
    setPendingAnswer(completed);
    setStep("verify");
    setError(null);
  };

  const goBack = () => {
    setStep("ask");
    setPendingAnswer(null);
    setError(null);
  };

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await confirmVisit(current.id, pendingAnswer);
      setQueue((q) => q.slice(1));
      setStep("ask");
      setPendingAnswer(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail || err.message : "Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return createPortal(
    <div className={styles.overlay}>
      <div
        className={styles.content}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="visitGateTitle"
        ref={contentRef}
        tabIndex={-1}
      >
        <div className={styles.icon}>
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M8 2v3M16 2v3M3.5 9h17M5 5h14a1.5 1.5 0 0 1 1.5 1.5V19a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 19V6.5A1.5 1.5 0 0 1 5 5Z" />
            <path d="m8.5 14.5 2.2 2.2 4.8-4.8" />
          </svg>
        </div>

        {step === "ask" && (
          <>
            <h2 id="visitGateTitle" className={styles.title}>
              Have you completed your previous visit?
            </h2>
            <p className={styles.detail}>
              Your appointment with <strong>{current.doctor_name}</strong> ({current.department_name}) on{" "}
              <strong>{formatDateTime(current.start_utc, clinicTimezone)}</strong> has ended. Please let us
              know what happened before continuing.
            </p>
            {queue.length > 1 && (
              <p className={styles.queueNote}>
                {queue.length - 1} more appointment{queue.length - 1 === 1 ? "" : "s"} will need confirming after this one.
              </p>
            )}
            <div className={styles.actions}>
              <button type="button" className={styles.missedBtn} onClick={() => choose(false)}>
                No, I missed it
              </button>
              <button type="button" className={styles.confirmBtn} onClick={() => choose(true)}>
                Yes, I confirm
              </button>
            </div>
          </>
        )}

        {step === "verify" && (
          <>
            <h2 id="visitGateTitle" className={styles.title}>
              Are you sure?
            </h2>
            <p className={styles.detail}>
              {pendingAnswer
                ? "You're confirming that this visit took place. This can't be changed afterward."
                : "You're marking this appointment as missed. This can't be changed afterward."}
            </p>
            {error && (
              <div className={styles.error} role="alert">
                {error}
              </div>
            )}
            <div className={styles.actions}>
              <button type="button" className={styles.backBtn} onClick={goBack} disabled={submitting}>
                Go back
              </button>
              <button
                type="button"
                className={pendingAnswer ? styles.confirmBtn : styles.missedBtn}
                onClick={submit}
                disabled={submitting}
              >
                {submitting ? "Submitting…" : "Yes, I'm sure"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  );
}
