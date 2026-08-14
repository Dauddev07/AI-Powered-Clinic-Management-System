"""Self-healing background jobs, both on one shared BackgroundScheduler:

1. Slot regeneration — rolls the slot horizon forward on its own, so the rolling
   `SLOT_GENERATION_HORIZON_DAYS` window doesn't shrink by a day every day that passes
   without an admin action (CSV re-upload, block-date) to re-trigger it.
2. Appointment reminders — sends the 60m/30m/5m/starting-now reminder for whichever
   appointments just came due.

Appointments whose slot has ended are NOT auto-completed by a timer here — that status
change now only happens via the patient's own explicit confirm-visit action (see
app/services/booking_engine.confirm_visit), surfaced as a blocking prompt on their next
visit to the site (app/api/appointments.py's /pending-confirmations).

Both jobs run on their own short, frequent interval rather than a single fixed daily
tick — each is idempotent and cheap (a no-op tick just finds nothing to do), so there's
no real cost to checking often, and it means neither job can fall behind by more than
one interval no matter what: a missed exact moment (process down, laptop asleep through
it, clock jump on wake) gets caught by the very next tick instead of waiting for the
next fixed trigger time or an admin noticing and re-triggering it by hand.

Each job calls the exact same service-layer function used by its admin/API-triggered
path (`regenerate_slots_for_clinic`) — a job here is only a scheduling trigger, never a
second implementation of that logic.
"""
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal
from app.models.audit_log import AuditLog
from app.models.clinic import Clinic
from app.services.appointment_reminders import send_due_reminders_for_clinic
from app.services.slots import regenerate_slots_for_clinic

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def run_slot_regeneration_tick() -> None:
    """Loops every active clinic and regenerates its slots. Each clinic's own local
    "today" is computed inside regenerate_slots_for_clinic via its timezone, so a
    single fixed-interval tick is enough to keep every clinic's horizon current —
    it doesn't need a separate per-clinic-timezone schedule.
    """
    db = SessionLocal()
    try:
        clinics = db.execute(select(Clinic).where(Clinic.is_active.is_(True))).scalars().all()
        for clinic in clinics:
            try:
                stats = regenerate_slots_for_clinic(db, clinic.id)
                changed = stats.inserted or stats.blocked or stats.unblocked or stats.flagged_for_review
                # Only audit-logged when something actually changed — at a 30-minute
                # tick interval, logging every no-op check as well would fill this
                # table with ~48 identical all-zero rows per clinic per day.
                if changed:
                    db.add(
                        AuditLog(
                            clinic_id=clinic.id,
                            actor_user_id=None,
                            action="scheduled_slot_regeneration",
                            entity_type="clinic",
                            entity_id=clinic.id,
                            metadata_json={
                                "doctors_processed": stats.doctors_processed,
                                "inserted": stats.inserted,
                                "blocked": stats.blocked,
                                "unblocked": stats.unblocked,
                                "flagged_for_review": stats.flagged_for_review,
                            },
                        )
                    )
                db.commit()
                if changed:
                    logger.info(
                        "scheduled_slot_regeneration clinic=%s inserted=%d blocked=%d unblocked=%d flagged_for_review=%d",
                        clinic.id,
                        stats.inserted,
                        stats.blocked,
                        stats.unblocked,
                        stats.flagged_for_review,
                    )
            except Exception:
                db.rollback()
                logger.exception("scheduled_slot_regeneration failed for clinic=%s", clinic.id)
    finally:
        db.close()


def run_appointment_reminder_tick() -> None:
    """Loops every active clinic and sends any appointment reminder (60m/30m/5m/
    starting-now) that's now due — see app.services.appointment_reminders for the
    exactly-once-per-window logic. Runs on a tight 1-minute interval (unlike the
    other two ticks above) since these reminders are meant to land close to an
    exact offset from the appointment's start time.
    """
    db = SessionLocal()
    try:
        clinics = db.execute(select(Clinic).where(Clinic.is_active.is_(True))).scalars().all()
        for clinic in clinics:
            try:
                created = send_due_reminders_for_clinic(db, clinic.id)
                db.commit()
                if created:
                    logger.info(
                        "scheduled_appointment_reminders clinic=%s sent=%d",
                        clinic.id,
                        len(created),
                    )
            except Exception:
                db.rollback()
                logger.exception("scheduled_appointment_reminders failed for clinic=%s", clinic.id)
    finally:
        db.close()


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_slot_regeneration_tick,
        trigger=IntervalTrigger(minutes=settings.SLOT_REGENERATION_INTERVAL_MINUTES),
        id="slot_regeneration_tick",
        replace_existing=True,
        # A tick delayed past its scheduled time (e.g. the process was asleep, or a
        # previous tick ran long) still runs once rather than being skipped — a
        # skipped tick is exactly the silent-falling-behind failure mode this interval
        # approach exists to avoid.
        misfire_grace_time=None,
    )
    scheduler.add_job(
        run_appointment_reminder_tick,
        trigger=IntervalTrigger(minutes=settings.APPOINTMENT_REMINDER_INTERVAL_MINUTES),
        id="appointment_reminder_tick",
        replace_existing=True,
        # Same reasoning as the other two jobs: a delayed tick still runs once rather
        # than being skipped, so a sleeping/restarted process sends any reminder that
        # came due while it was down instead of silently losing it forever (each
        # reminder type is idempotent — see send_due_reminders_for_clinic — so a late
        # catch-up run is safe, never a duplicate).
        misfire_grace_time=None,
    )
    scheduler.start()
    _scheduler = scheduler

    # The interval trigger above only ticks for a process that's actually running —
    # any stretch the server was down (or asleep through every tick in the interval),
    # the horizon silently falls behind and never catches up on its own until the
    # next tick after it wakes/restarts. Running once here on every startup means the
    # rolling horizon is always brought back out to a full SLOT_GENERATION_HORIZON_DAYS
    # from "now" immediately, rather than waiting up to one interval for the first tick.
    #
    # Best-effort: a DB hiccup at boot (e.g. the database isn't accepting
    # connections yet) must not prevent the app itself from starting — the next
    # tick (or the next restart) will simply try the catch-up again.
    try:
        run_slot_regeneration_tick()
    except Exception:
        logger.exception("startup slot regeneration catch-up failed")

    return scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
