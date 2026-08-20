"""Weekly plain-English digest of the admin dashboard's own numbers — one short LLM
call over facts already computed for the Overview/Insights cards
(app.api.admin_dashboard), never a second source of truth for any of them.

Same fail-safe, cache-and-refresh shape as app.services.memory_summary: a failed
generation leaves whatever digest already exists untouched (or returns None if there
was never one), rather than blocking the dashboard or showing an error where a
paragraph of prose would go. Cached in admin_insight_digests and only regenerated once
REFRESH_INTERVAL has passed, so a dashboard visit doesn't cost an LLM call every time.
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.admin_dashboard import get_admin_dashboard_stats, get_appointments_trend, get_top_rated_doctors
from app.core.api_keys import api_key_manager
from app.core.config import settings
from app.models.admin_insight_digest import AdminInsightDigest
from app.models.user import User

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = timedelta(days=7)

# A generation run outside this length band is treated as a degenerate/failed
# response (the model answered something else, or produced far too much prose for a
# dashboard card) and discarded in favor of leaving the existing digest untouched —
# same convention as memory_summary._MAX_SUMMARY_LENGTH.
_MAX_DIGEST_LENGTH = 800

_SYSTEM_PROMPT = """You write a short weekly summary for a clinic administrator's \
dashboard, based ONLY on the numbers given to you below.

Rules:
- Use ONLY the facts provided. Never invent a number, doctor name, date, or trend \
that isn't explicitly present in them.
- This is a WEEKLY summary — describe the week as a whole (the 7-day total/trend), \
never a single day's figures. The facts below deliberately exclude anything scoped to \
"today" for exactly this reason.
- Write 2-4 plain sentences of continuous prose. No headings, no markdown, no bullet \
points, no greeting or sign-off.
- Point out whatever is actually notable in the numbers (a doctor standing out, a busy \
or quiet week, a highly or poorly rated doctor) rather than just restating every figure \
in order. If nothing stands out, a brief, plain factual summary is fine.
- Never give medical advice or comment on any patient's health — this is operational \
clinic data only.

Output ONLY the summary text itself — no preamble, no labels, no quotes."""


def _facts_text(db: Session, current_user: User) -> str:
    """Builds the digest prompt from the exact same queries that power the
    dashboard's own cards — calling the FastAPI route functions directly (their
    `Depends(...)` defaults are only used when FastAPI itself invokes them; a plain
    call with real args bypasses that entirely) so this can never drift from what an
    admin actually sees on screen.

    Deliberately excludes anything scoped to "today" (slot_utilization_today,
    busiest_doctors_today) — this is a WEEKLY digest, and feeding the model a
    same-day figure alongside "weekly" framing in the prompt produced summaries that
    led with today's single-day numbers instead of the week. Only genuinely
    week-or-wider facts go in: the 7-day trend (and its own total), active doctor
    headcount, and all-time top ratings.
    """
    stats = get_admin_dashboard_stats(current_user=current_user, db=db)
    trend = get_appointments_trend(current_user=current_user, db=db)
    top_rated = get_top_rated_doctors(current_user=current_user, db=db)

    trend_line = ", ".join(f"{day.date} booked={day.count}" for day in trend)
    week_total = sum(day.count for day in trend)
    top_rated_line = (
        ", ".join(f"{d.doctor_name} ({d.average_rating}/5 from {d.rating_count} ratings)" for d in top_rated)
        or "no doctor ratings yet"
    )

    return (
        f"Active doctors: {stats.active_doctors_count} of {stats.total_doctors_count} total.\n"
        f"Appointments booked per day, last 7 days, oldest to newest: {trend_line} "
        f"(week total: {week_total}).\n"
        f"Top rated doctors (clinic-wide, all time): {top_rated_line}."
    )


def _generate(facts: str) -> str | None:
    """Returns the generated digest, or None if the call failed/produced a
    degenerate result — callers should keep whatever digest already exists in that
    case, exactly like memory_summary._summarize."""
    if not settings.LLM_API_KEY:
        return None

    try:
        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=api_key_manager.next_key(), temperature=0.2)
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=facts)]
        digest = llm.invoke(messages).content.strip()
    except Exception:
        logger.exception("Admin weekly digest generation failed")
        return None

    if not digest or len(digest) > _MAX_DIGEST_LENGTH:
        return None
    return digest


def get_weekly_digest(
    db: Session, clinic_id: uuid.UUID, current_user: User
) -> tuple[str | None, datetime | None]:
    """Returns (digest_text, generated_at). digest_text is None only when nothing has
    ever been successfully generated for this clinic (no LLM key configured, or every
    attempt so far has failed) — the caller/frontend should treat that as "no digest
    available" rather than an error. A digest already cached within REFRESH_INTERVAL
    is returned as-is with no LLM call; a stale one triggers a regeneration attempt,
    falling back to the stale text (with its original generated_at) if that attempt
    fails.
    """
    row = db.execute(
        select(AdminInsightDigest).where(AdminInsightDigest.clinic_id == clinic_id)
    ).scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if row is not None and now - row.generated_at < REFRESH_INTERVAL:
        return row.digest_text, row.generated_at

    facts = _facts_text(db, current_user)
    digest = _generate(facts)
    if digest is None:
        return (row.digest_text, row.generated_at) if row is not None else (None, None)

    if row is None:
        row = AdminInsightDigest(clinic_id=clinic_id, digest_text=digest, generated_at=now)
        db.add(row)
    else:
        row.digest_text = digest
        row.generated_at = now
    db.flush()
    return digest, row.generated_at
