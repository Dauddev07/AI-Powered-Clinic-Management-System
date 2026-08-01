import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.api.chat import get_chat_history
from app.core.tenancy import ClinicContext
from app.models.clinic import Clinic
from app.models.conversation_memory import ConversationMemory
from app.models.user import User


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def ctx(db, clinic):
    patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(patient)
    db.flush()
    return ClinicContext(clinic_id=clinic.id, user_id=patient.id, role="patient")


def _save(db, ctx, session_id, role, content, created_at=None):
    """Mirrors app.services.chat._save_message, but lets a test pin an explicit
    created_at to reproduce a same-timestamp tie deterministically. Each call is
    flushed on its own so the auto-incrementing `seq` column reflects true insertion
    order even when created_at is identical across rows."""
    row = ConversationMemory(
        clinic_id=ctx.clinic_id, session_id=session_id, user_id=ctx.user_id, role=role, content=content,
    )
    if created_at is not None:
        row.created_at = created_at
    db.add(row)
    db.flush()
    return row


def test_history_endpoint_returns_messages_in_strict_chronological_order(db, ctx):
    session_id = uuid.uuid4()
    _save(db, ctx, session_id, "user", "How many departments does this clinic have?")
    _save(db, ctx, session_id, "assistant", "This clinic has 12 departments.")
    _save(db, ctx, session_id, "user", "Can you name them?")
    _save(db, ctx, session_id, "assistant", "Cardiology, Neurology, Pediatrics, ...")

    result = get_chat_history(session_id=session_id, _current_user=None, ctx=ctx, db=db)

    contents = [m.content for m in result.messages]
    assert contents == [
        "How many departments does this clinic have?",
        "This clinic has 12 departments.",
        "Can you name them?",
        "Cardiology, Neurology, Pediatrics, ...",
    ]


def test_history_endpoint_breaks_same_timestamp_ties_by_insertion_order(db, ctx):
    """Reproduces the reported bug directly: a single chat turn saves its user and
    assistant rows in the same DB transaction, and Postgres now() returns the same
    value for every statement within one transaction — so these two rows can have an
    IDENTICAL created_at. ORDER BY created_at alone has no defined tiebreaker in that
    case; the `seq` column must resolve it deterministically in true insertion
    order."""
    session_id = uuid.uuid4()
    tied_timestamp = datetime.now(timezone.utc)

    first = _save(db, ctx, session_id, "user", "My name is Daud, remember this", created_at=tied_timestamp)
    second = _save(db, ctx, session_id, "assistant", "Got it, Daud!", created_at=tied_timestamp)
    third = _save(db, ctx, session_id, "user", "What is my name?", created_at=tied_timestamp)
    fourth = _save(db, ctx, session_id, "assistant", "Your name is Daud.", created_at=tied_timestamp)

    assert first.created_at == second.created_at == third.created_at == fourth.created_at, (
        "test is only meaningful if these rows genuinely tie on created_at"
    )
    # seq must still be strictly increasing in insertion order despite the tie.
    assert first.seq < second.seq < third.seq < fourth.seq

    result = get_chat_history(session_id=session_id, _current_user=None, ctx=ctx, db=db)

    contents = [m.content for m in result.messages]
    assert contents == [
        "My name is Daud, remember this",
        "Got it, Daud!",
        "What is my name?",
        "Your name is Daud.",
    ]


@pytest.mark.parametrize("session_index", [0, 1, 2])
def test_history_ordering_holds_across_multiple_independent_sessions(db, ctx, session_index):
    # Same check repeated over several distinct conversations, so the ordering fix
    # isn't verified as a one-off against a single lucky session.
    session_id = uuid.uuid4()
    base = datetime.now(timezone.utc) + timedelta(seconds=session_index)

    expected = [
        ("user", f"Session {session_index}: question one"),
        ("assistant", f"Session {session_index}: answer one"),
        ("user", f"Session {session_index}: question two"),
        ("assistant", f"Session {session_index}: answer two"),
    ]
    for role, content in expected:
        _save(db, ctx, session_id, role, content, created_at=base)

    result = get_chat_history(session_id=session_id, _current_user=None, ctx=ctx, db=db)

    assert [m.content for m in result.messages] == [content for _, content in expected]


def test_history_endpoint_with_no_session_id_resolves_latest_session_correctly_ordered(db, ctx):
    older_session = uuid.uuid4()
    newer_session = uuid.uuid4()
    now = datetime.now(timezone.utc)

    _save(db, ctx, older_session, "user", "older session message", created_at=now - timedelta(minutes=5))
    _save(db, ctx, newer_session, "user", "newer session question", created_at=now)
    _save(db, ctx, newer_session, "assistant", "newer session answer", created_at=now)

    result = get_chat_history(session_id=None, _current_user=None, ctx=ctx, db=db)

    assert result.session_id == newer_session
    assert [m.content for m in result.messages] == ["newer session question", "newer session answer"]
