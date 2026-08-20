"""Synthesizes recurring themes out of this clinic's low-rating (1-2 star) feedback
reasons — one short LLM call over real AppointmentFeedback rows (app.api.admin_feedback
already lists these individually; this never becomes a second source of truth for the
ratings themselves), so an admin gets "these are the 2-3 things patients keep
complaining about, and who they're about" instead of reading every reason one at a time.

Same fail-safe, cache-and-refresh shape as app.services.admin_insights: a failed
generation leaves whatever digest already exists untouched (or returns None if there
was never one), and a clinic with no low-rating feedback yet gets no digest at all
rather than the model being asked to invent a summary out of nothing.
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.api_keys import api_key_manager
from app.core.config import settings
from app.models.appointment_feedback import AppointmentFeedback
from app.models.doctor import Doctor
from app.models.feedback_insight_digest import FeedbackInsightDigest

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = timedelta(days=7)

# Low-rating feedback is rare (see AppointmentFeedback's own docstring: `reason` is
# only ever populated for a 1-2 star rating) — a single calendar week's worth is often
# zero or one row, nowhere near enough to synthesize a real "recurring theme" from.
# The digest still refreshes weekly (REFRESH_INTERVAL above), but the facts it's
# built from look back much further so there's actually a pattern to find.
LOOKBACK_WINDOW = timedelta(days=90)

# Bounds prompt size against a clinic with an unusually large backlog of low-rating
# feedback — most recent first, so a long-resolved complaint pattern doesn't crowd out
# what's actually recent.
_MAX_FEEDBACK_ROWS = 150

# Same convention as admin_insights._MAX_DIGEST_LENGTH: a generation run outside this
# length band is treated as a degenerate/failed response and discarded.
_MAX_DIGEST_LENGTH = 800

_SYSTEM_PROMPT = """You write a short internal summary for a clinic administrator, \
based ONLY on the low-rating (1-2 star) patient feedback reasons given to you below, \
each one labeled with the doctor it was about.

Rules:
- Use ONLY the feedback text provided. Never invent a complaint, doctor name, or \
detail that isn't actually present in it.
- Identify RECURRING themes — something multiple patients independently raised, or a \
pattern tied to one specific doctor — rather than just restating every complaint in \
order. A single one-off complaint is only worth mentioning if nothing else recurs at \
all, and must be described as ONE report, never phrased as "patients" (plural) or \
"a pattern" when only one entry actually supports it.
- Name the doctor a recurring theme is about when the feedback makes that clear; \
never name or describe an individual patient.
- Write 2-4 plain sentences of continuous prose. No headings, no markdown, no bullet \
points, no greeting or sign-off.
- Stay factual and neutral — this is operational feedback for the clinic's own use, \
not a public review.

Output ONLY the summary text itself — no preamble, no labels, no quotes."""


def _facts_text(db: Session, clinic_id: uuid.UUID) -> str | None:
    """Returns the prompt facts, or None if there's no low-rating feedback with a
    reason at all in the lookback window — callers should skip the LLM call entirely
    in that case rather than asking the model to summarize nothing."""
    cutoff = datetime.now(timezone.utc) - LOOKBACK_WINDOW
    rows = db.execute(
        select(AppointmentFeedback.reason, Doctor.full_name)
        .join(Doctor, Doctor.id == AppointmentFeedback.doctor_id)
        .where(
            AppointmentFeedback.clinic_id == clinic_id,
            AppointmentFeedback.rating <= 2,
            AppointmentFeedback.reason.isnot(None),
            AppointmentFeedback.created_at >= cutoff,
        )
        .order_by(AppointmentFeedback.created_at.desc())
        .limit(_MAX_FEEDBACK_ROWS)
    ).all()

    if not rows:
        return None

    lines = "\n".join(f"- [{doctor_name}] {reason}" for reason, doctor_name in rows)
    return f"Low-rating feedback from the last {LOOKBACK_WINDOW.days} days, most recent first:\n{lines}"


def _generate(facts: str) -> str | None:
    """Returns the generated digest, or None if the call failed/produced a
    degenerate result — callers should keep whatever digest already exists in that
    case, exactly like admin_insights._generate."""
    if not settings.LLM_API_KEY:
        return None

    try:
        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=api_key_manager.next_key(), temperature=0.2)
        messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=facts)]
        digest = llm.invoke(messages).content.strip()
    except Exception:
        logger.exception("Feedback insight digest generation failed")
        return None

    if not digest or len(digest) > _MAX_DIGEST_LENGTH:
        return None
    return digest


def get_feedback_digest(db: Session, clinic_id: uuid.UUID) -> tuple[str | None, datetime | None]:
    """Returns (digest_text, generated_at). digest_text is None either when this
    clinic has no low-rating feedback with a reason yet, or when nothing has ever
    been successfully generated (no LLM key configured, or every attempt so far has
    failed) — the caller/frontend should treat both the same way: no digest
    available. A digest already cached within REFRESH_INTERVAL is returned as-is with
    no LLM call; a stale one triggers a regeneration attempt, falling back to the
    stale text (with its original generated_at) if that attempt fails.
    """
    row = db.execute(
        select(FeedbackInsightDigest).where(FeedbackInsightDigest.clinic_id == clinic_id)
    ).scalar_one_or_none()

    now = datetime.now(timezone.utc)
    if row is not None and now - row.generated_at < REFRESH_INTERVAL:
        return row.digest_text, row.generated_at

    facts = _facts_text(db, clinic_id)
    if facts is None:
        return (row.digest_text, row.generated_at) if row is not None else (None, None)

    digest = _generate(facts)
    if digest is None:
        return (row.digest_text, row.generated_at) if row is not None else (None, None)

    if row is None:
        row = FeedbackInsightDigest(clinic_id=clinic_id, digest_text=digest, generated_at=now)
        db.add(row)
    else:
        row.digest_text = digest
        row.generated_at = now
    db.flush()
    return digest, row.generated_at
