"""Maintains each patient's cross-session memory digest (see
app.models.patient_memory_profile) — a short summary of symptoms they've previously
described, general personal info they've volunteered, and the general subject of
anything else they've asked about, carried into a brand new chat session in place of
the full transcript (see app.services.chat: within a session, the real
conversation_memory transcript is replayed in full; ACROSS sessions, only this digest
is).

One small, focused LLM call, same fail-safe shape as app.services.query_rewrite:
any failure (no LLM configured, API error, degenerate output) leaves the existing
summary untouched rather than blocking the chat turn or losing prior memory.
"""
import logging
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.api_keys import api_key_manager
from app.core.config import settings
from app.models.conversation_memory import ConversationMemory
from app.models.patient_memory_profile import PatientMemoryProfile

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You maintain a short running memory profile for one patient, \
carried forward between separate chat sessions with a clinic assistant chatbot.

From the "New messages" below, extract ONLY three kinds of information:
1. Symptoms or health complaints the patient described (e.g. "reported recurring \
migraines", "mentioned a peanut allergy", "described knee pain from an old injury").
2. General personal info the patient volunteered about themselves that isn't specific \
to one single appointment (e.g. their name, occupation, a chronic condition, a \
preference like "prefers morning appointments").
3. The general SUBJECT of anything else the patient asked about or discussed, so a \
later "what did I tell you about" question has something real to answer beyond just \
symptoms/personal info (e.g. "asked about Cardiology availability", "asked whether \
Dr. Ahmed Farooq sees patients on Fridays", "asked about the clinic's opening \
hours"). Capture only the topic/subject, never the specific answer given (a \
department/doctor NAME as the subject of a question is fine to keep — e.g. "asked \
about Dr. X" — since this is only ever used to recall what the conversation was \
about, never as a source for an actual booking action).

Do NOT include: the RESULT of any appointment booking/reschedule/cancellation \
(whether it succeeded, what time was booked, etc.) or any specific date/time/slot \
detail — appointment state can change outside this chat entirely and must always be \
verified fresh via a real lookup, never recalled from this summary as fact. Also \
exclude small talk, and anything you're not confident is actually true of the \
patient (never invent details).

Merge this with the "Existing summary" (which may be empty) into ONE updated summary: \
combine related facts, drop anything clearly superseded or no longer relevant, and \
keep the whole thing SHORT — a few plain sentences, not a list, under 120 words. If \
there is genuinely nothing worth remembering (only a booking's result, small talk, or \
nothing new), return the existing summary unchanged, or an empty string if it was \
already empty.

Output ONLY the updated summary text itself — no preamble, no labels, no quotes.
"""

# A summarization run outside this length band is treated as a degenerate/failed
# response (the model answered something else, or produced far too much) and
# discarded in favor of leaving the existing summary untouched.
_MAX_SUMMARY_LENGTH = 1200


def _messages_text(rows: list[ConversationMemory]) -> str:
    return "\n".join(f"{row.role}: {row.content}" for row in rows)


def _summarize(existing_summary: str, new_rows: list[ConversationMemory]) -> str | None:
    """Returns the updated summary, or None if the call failed/produced a degenerate
    result — callers should leave the existing summary untouched in that case."""
    if not settings.LLM_API_KEY:
        return None

    try:
        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=api_key_manager.next_key(), temperature=0.0)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Existing summary: {existing_summary or '(empty)'}\n\n"
                    f"New messages:\n{_messages_text(new_rows)}"
                )
            ),
        ]
        updated = llm.invoke(messages).content.strip()
    except Exception:
        logger.exception("Patient memory summarization failed, leaving existing summary untouched")
        return None

    if len(updated) > _MAX_SUMMARY_LENGTH:
        return None
    return updated


def get_patient_summary(db: Session, clinic_id: uuid.UUID, user_id: uuid.UUID) -> str:
    """Cheap read-only lookup of the patient's current memory digest — no LLM call,
    no watermark advance. Used for every turn of a CONTINUING session (see
    app.services.chat), so a fact from an earlier session (e.g. the patient's name)
    stays available to the whole session, not just its first turn — only the
    expensive re-summarization in refresh_patient_summary_for_new_session below is
    limited to once per new session."""
    profile = db.execute(
        select(PatientMemoryProfile).where(
            PatientMemoryProfile.clinic_id == clinic_id, PatientMemoryProfile.user_id == user_id
        )
    ).scalar_one_or_none()
    return profile.summary_text if profile else ""


def reset_patient_summary(db: Session, clinic_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Wipes the patient's memory digest outright — called by app.services.chat's
    delete_session() so a deleted chat is genuinely forgotten everywhere, not just in
    the raw transcript. The digest is an LLM-generated blend of facts pulled from
    across every session (see refresh_patient_summary_for_new_session's docstring) —
    there's no reliable way to surgically remove just the deleted session's
    contribution from that free-text blend, so this clears it entirely and resets the
    watermark to 0. Nothing is lost for good: the next new session's refresh call
    regenerates the digest from whatever ConversationMemory rows still exist (i.e.
    every session that WASN'T deleted), just without the deleted session's content."""
    profile = db.execute(
        select(PatientMemoryProfile).where(
            PatientMemoryProfile.clinic_id == clinic_id, PatientMemoryProfile.user_id == user_id
        )
    ).scalar_one_or_none()
    if profile is None:
        return
    profile.summary_text = ""
    profile.last_summarized_seq = 0
    db.flush()


def refresh_patient_summary_for_new_session(db: Session, clinic_id: uuid.UUID, user_id: uuid.UUID) -> str:
    """Called once, at the start of a brand new chat session (see app.services.chat)
    — folds in any messages from earlier sessions that haven't been summarized yet,
    and returns the resulting summary text (never None; "" if there's nothing to
    remember). A continuing session never calls this — it replays its own full
    transcript instead, so there's nothing to add to the digest until a later session
    starts.
    """
    profile = db.execute(
        select(PatientMemoryProfile).where(
            PatientMemoryProfile.clinic_id == clinic_id, PatientMemoryProfile.user_id == user_id
        )
    ).scalar_one_or_none()

    if profile is None:
        profile = PatientMemoryProfile(clinic_id=clinic_id, user_id=user_id, summary_text="", last_summarized_seq=0)
        db.add(profile)
        db.flush()

    new_rows = list(
        db.execute(
            select(ConversationMemory)
            .where(
                ConversationMemory.clinic_id == clinic_id,
                ConversationMemory.user_id == user_id,
                ConversationMemory.seq > profile.last_summarized_seq,
            )
            .order_by(ConversationMemory.seq.asc())
        ).scalars()
    )

    if not new_rows:
        return profile.summary_text

    updated = _summarize(profile.summary_text, new_rows)
    if updated is None:
        # Summarization failed — still advance the watermark past these rows so a
        # persistently failing call (e.g. no LLM key) doesn't force this clinic to
        # re-attempt summarizing an ever-growing backlog on every new session.
        profile.last_summarized_seq = new_rows[-1].seq
        db.flush()
        return profile.summary_text

    profile.summary_text = updated
    profile.last_summarized_seq = new_rows[-1].seq
    db.flush()
    return profile.summary_text
