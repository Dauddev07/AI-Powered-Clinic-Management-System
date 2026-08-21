import uuid

import pytest

from app.core.tenancy import ClinicContext
from app.models.clinic import Clinic
from app.models.user import User
from app.rag.retrieval import FALLBACK_MESSAGE, FALLBACK_MESSAGE_UR
from app.services.chat import (
    APPOINTMENT_AGENT_HISTORY_LIMIT,
    GENERAL_INFO_AGENT_HISTORY_LIMIT,
    SYMPTOM_AGENT_HISTORY_LIMIT,
    _save_message,
    delete_session,
    handle_chat_message,
    list_sessions,
)
from app.services.orchestrator.router import APPOINTMENT, GENERAL_INFO, SYMPTOM_GENERAL


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}")
    db.add(c)
    db.flush()
    return c


def _patient(db, clinic):
    p = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(p)
    db.flush()
    return p


@pytest.fixture
def ctx(db, clinic):
    patient = _patient(db, clinic)
    return ClinicContext(clinic_id=clinic.id, user_id=patient.id, role="patient")


def _patch_intent(monkeypatch, intent):
    """Forces app.services.chat's routing decision deterministically, so tests about
    session/history/memory/red-flag/tenancy behavior don't depend on the router's
    own heuristic classifying a given test message a particular way — that
    classification logic has its own dedicated coverage in test_orchestrator.py."""
    monkeypatch.setattr("app.services.chat.classify_agent_intent", lambda message, history=None: intent)


def _patch_general_info_reply(monkeypatch, reply):
    monkeypatch.setattr(
        "app.services.chat.run_general_info_agent",
        lambda db, ctx, message, language, history: reply,
    )


def _patch_symptom_reply(monkeypatch, reply):
    monkeypatch.setattr(
        "app.services.chat.run_symptom_agent",
        lambda db, ctx, message, language, history: reply,
    )


# --- routing: the orchestrator's intent layer decides which specialist runs --------


def test_symptom_general_intent_routes_to_symptom_agent(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, SYMPTOM_GENERAL)

    for name in ("run_appointment_agent", "run_general_info_agent"):
        monkeypatch.setattr(
            f"app.services.chat.{name}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} must not be called")),
        )
    _patch_symptom_reply(monkeypatch, "Let's figure out which department fits.")

    result = handle_chat_message(db, ctx, "I have a fever and body aches", None)

    assert result.reply == "Let's figure out which department fits."


def test_symptom_agents_own_path1_emergency_reply_is_persisted_as_red_flag(db, ctx, monkeypatch):
    # Requested: only the red_flag.py pre-guard's own canned message was ever
    # persisted with red_flag=True — symptom_agent's own PATH 1 determination
    # (a patient's stated severity, e.g. "very severe") was never marked at
    # all, so nothing downstream (including the emergency-downgrade safety
    # note) could reliably tell "this session already had an emergency" from
    # the DB column alone.
    _patch_intent(monkeypatch, SYMPTOM_GENERAL)
    _patch_symptom_reply(
        monkeypatch,
        "This sounds like an emergency. Call 1122 right away or go to the nearest ER.",
    )

    result = handle_chat_message(db, ctx, "very severe headache", None)

    assert result.red_flag is True


def test_symptom_agents_ordinary_reply_is_not_persisted_as_red_flag(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, SYMPTOM_GENERAL)
    _patch_symptom_reply(monkeypatch, "Let's figure out which department fits.")

    result = handle_chat_message(db, ctx, "I have a mild headache", None)

    assert result.red_flag is False


def test_appointment_intent_routes_to_appointment_agent(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, APPOINTMENT)

    for name in ("run_symptom_agent", "run_general_info_agent"):
        monkeypatch.setattr(
            f"app.services.chat.{name}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} must not be called")),
        )
    monkeypatch.setattr(
        "app.services.chat.run_appointment_agent",
        lambda db, ctx, message, language, history: "Booked.",
    )

    result = handle_chat_message(db, ctx, "book me with Dr. Ahmed at 4pm", None)

    assert result.reply == "Booked."


def test_general_info_intent_routes_to_general_info_agent(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)

    for name in ("run_symptom_agent", "run_appointment_agent"):
        monkeypatch.setattr(
            f"app.services.chat.{name}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} must not be called")),
        )
    _patch_general_info_reply(monkeypatch, "We're open 9am-5pm.")

    result = handle_chat_message(db, ctx, "what are your clinic hours?", None)

    assert result.reply == "We're open 9am-5pm."


# --- compound intent: two distinct things in one message ---------------------------
# Requested: a message stating a real symptom alongside an unrelated cancel/reschedule
# request, or a symptom alongside a clinic-logistics question, used to have one of the
# two silently dropped entirely — the router picks exactly one bucket, and whichever
# specialist agent it hands off to has no idea the other thing was even said. Now asks
# which one to help with first instead of guessing/dropping either.


def test_compound_symptom_and_cancel_message_asks_which_one_first(db, ctx, monkeypatch):
    for name in ("run_symptom_agent", "run_appointment_agent", "run_general_info_agent"):
        monkeypatch.setattr(
            f"app.services.chat.{name}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} must not be called before a choice is made")),
        )

    result = handle_chat_message(
        db, ctx, "my chest really hurts, please cancel my appointment tomorrow", None
    )

    assert "which would you like me to help with first" in result.reply.lower()
    assert "symptom" in result.reply.lower()
    assert "cancel" in result.reply.lower() or "reschedul" in result.reply.lower()


def test_compound_symptom_and_logistics_message_asks_which_one_first(db, ctx, monkeypatch):
    for name in ("run_symptom_agent", "run_appointment_agent", "run_general_info_agent"):
        monkeypatch.setattr(
            f"app.services.chat.{name}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} must not be called before a choice is made")),
        )

    result = handle_chat_message(db, ctx, "i have a bad headache, also what are your clinic hours", None)

    assert "which would you like me to help with first" in result.reply.lower()


def test_compound_message_answered_with_symptom_choice_reruns_original_message(db, ctx, monkeypatch):
    # Once the patient picks "symptom", the ORIGINAL compound message (not the
    # bare "symptom first" answer, which has no real content of its own) must
    # be what actually reaches the symptom agent.
    seen = {}

    def fake_symptom_agent(db, ctx, message, language, history):
        seen["message"] = message
        return "Let's figure out which department fits."

    monkeypatch.setattr("app.services.chat.run_symptom_agent", fake_symptom_agent)
    monkeypatch.setattr(
        "app.services.chat.run_appointment_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_appointment_agent must not be called")),
    )

    original = "my chest really hurts, please cancel my appointment tomorrow"
    first = handle_chat_message(db, ctx, original, None)
    result = handle_chat_message(db, ctx, "symptom first please", first.session_id)

    assert result.reply == "Let's figure out which department fits."
    assert seen["message"] == original


def test_compound_message_answered_with_cancel_choice_reruns_original_message(db, ctx, monkeypatch):
    seen = {}

    def fake_appointment_agent(db, ctx, message, language, history):
        seen["message"] = message
        return "Sure, let's cancel that."

    monkeypatch.setattr("app.services.chat.run_appointment_agent", fake_appointment_agent)
    monkeypatch.setattr(
        "app.services.chat.run_symptom_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_symptom_agent must not be called")),
    )

    original = "my chest really hurts, please cancel my appointment tomorrow"
    first = handle_chat_message(db, ctx, original, None)
    result = handle_chat_message(db, ctx, "the cancellation please", first.session_id)

    assert result.reply == "Sure, let's cancel that."
    assert seen["message"] == original


def test_compound_message_answered_with_an_unrelated_reply_falls_through_to_normal_routing(
    db, ctx, monkeypatch
):
    # Requested: if the patient's next message DOESN'T answer the clarifying
    # question at all (a genuinely unrelated message), it must be handled
    # exactly like any other message — normal classification of THAT message,
    # never forced into either track from the stale compound question.
    _patch_general_info_reply(monkeypatch, "We're open 9am-5pm.")
    monkeypatch.setattr(
        "app.services.chat.run_symptom_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_symptom_agent must not be called")),
    )
    monkeypatch.setattr(
        "app.services.chat.run_appointment_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_appointment_agent must not be called")),
    )
    _patch_intent(monkeypatch, GENERAL_INFO)

    first = handle_chat_message(
        db, ctx, "my chest really hurts, please cancel my appointment tomorrow", None
    )
    result = handle_chat_message(db, ctx, "where is this clinic located", first.session_id)

    assert result.reply == "We're open 9am-5pm."


def test_compound_correction_after_a_wrong_choice_is_not_reasked_as_compound_again(db, ctx, monkeypatch):
    # Reported live, full transcript: "i am having pain in my back, i want to
    # cancel my appointment" -> asked which one first -> "cancelling" ->
    # appointment_agent said "you don't have an upcoming appointment to
    # cancel" -> "no no not cancel, symptoms" (a correction) WRONGLY got the
    # exact same compound question again, instead of being read as "just the
    # symptom, please" — router._message_states_a_cancel_or_reschedule_action
    # is a bare keyword check with no negation awareness, so the literal word
    # "cancel" inside "not cancel" still counted as a second thing.
    monkeypatch.setattr(
        "app.services.chat.run_appointment_agent",
        lambda db, ctx, message, language, history: "You don't have an upcoming appointment to cancel.",
    )
    seen = {}

    def fake_symptom_agent(db, ctx, message, language, history):
        seen["message"] = message
        return "Let's figure out which department fits."

    monkeypatch.setattr("app.services.chat.run_symptom_agent", fake_symptom_agent)

    original = "i am having pain in my back, i want to cancel my appointment"
    first = handle_chat_message(db, ctx, original, None)
    assert "which would you like me to help with first" in first.reply.lower()

    second = handle_chat_message(db, ctx, "cancelling", first.session_id)
    assert second.reply == "You don't have an upcoming appointment to cancel."

    third = handle_chat_message(db, ctx, "no no not cancel,symptoms", second.session_id)

    assert "which would you like me to help with first" not in third.reply.lower()
    assert third.reply == "Let's figure out which department fits."
    assert seen["message"] == "no no not cancel,symptoms"


def test_a_genuinely_coherent_single_ask_is_never_treated_as_compound(db, ctx, monkeypatch):
    # "I have a headache, can you book me an appointment" is ONE coherent ask
    # (a symptom leading to a booking), not two separate things — must go
    # straight to the normally-routed agent, never the clarifying question.
    _patch_intent(monkeypatch, SYMPTOM_GENERAL)
    _patch_symptom_reply(monkeypatch, "Let's figure out which department fits.")
    monkeypatch.setattr(
        "app.services.chat.run_appointment_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_appointment_agent must not be called")),
    )

    result = handle_chat_message(db, ctx, "I have a headache, can you book me an appointment", None)

    assert result.reply == "Let's figure out which department fits."


def test_a_real_emergency_bypasses_the_compound_intent_question_entirely(db, ctx, monkeypatch):
    # A genuine emergency signal must short-circuit BEFORE the compound-intent
    # check ever runs — never offered as "would you like help with the
    # emergency or the cancellation first?".
    for name in ("run_symptom_agent", "run_appointment_agent", "run_general_info_agent"):
        monkeypatch.setattr(
            f"app.services.chat.{name}",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"{name} must not be called for a real emergency")),
        )

    result = handle_chat_message(
        db, ctx, "i am having severe chest pain and shortness of breath, please cancel my appointment", None
    )

    assert result.red_flag is True
    assert "which would you like me to help with first" not in result.reply.lower()


# --- cross-session memory -------------------------------------------------------------


def test_new_session_does_not_replay_a_prior_sessions_full_transcript(db, ctx, monkeypatch):
    # session_id IS a real memory boundary for the verbatim transcript (see module
    # docstring) — a genuinely new session must never see another session's
    # messages in `history`, only its own (empty, since it's brand new).
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "ack")

    first = handle_chat_message(db, ctx, "My name is Ali and I have a headache.", None)

    seen = {}

    def capture_reply(db, ctx, message, language, history):
        seen["history"] = history
        return "second reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_reply)

    second = handle_chat_message(db, ctx, "What did I just tell you my name was?", None)

    assert second.session_id != first.session_id, "test is only meaningful across two distinct sessions"
    assert seen["history"] == []


def test_continuing_session_replays_its_own_full_transcript(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "ack")

    first = handle_chat_message(db, ctx, "My name is Ali and I have a headache.", None)

    seen = {}

    def capture_reply(db, ctx, message, language, history):
        seen["history"] = history
        return "second reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_reply)

    handle_chat_message(db, ctx, "What did I just tell you my name was?", first.session_id)

    history_contents = [row.content for row in seen["history"]]
    assert "My name is Ali and I have a headache." in history_contents
    assert "ack" in history_contents


def test_new_chat_never_carries_a_cross_session_memory_digest(db, ctx, monkeypatch):
    # Reported live: memory must be scoped to the current chat/session only — a
    # brand new chat starts completely fresh, with no digest of anything said in a
    # PREVIOUS session (also keeps prompts smaller/cheaper: no digest text, and no
    # per-new-session summarization LLM call at all anymore). The patient_memory
    # concept and prompt section were later removed entirely (not just left
    # empty) — see this module's docstring — since leaving the LLM prompt's
    # memory instructions in place with an always-empty value caused the model to
    # misfire on unrelated messages.
    _patch_intent(monkeypatch, GENERAL_INFO)

    seen = {"call_count": 0}

    def capture_reply(db, ctx, message, language, history):
        seen["call_count"] += 1
        return "reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_reply)

    first = handle_chat_message(db, ctx, "My name is Ali and I have a headache.", None)
    handle_chat_message(db, ctx, "hi", None)
    handle_chat_message(db, ctx, "still there?", first.session_id)

    # No TypeError from an unexpected patient_memory argument across a fresh
    # session, a second new session, and a continuing session — the call is
    # made with exactly (db, ctx, message, language, history), nothing more.
    assert seen["call_count"] == 3


# --- Part C: per-agent history limits -------------------------------------------------


def _seed_history(db, ctx, count):
    """Writes `count` alternating user/assistant rows directly (no LLM calls), all in
    one session, so a test can cheaply exceed every agent's history limit."""
    session_id = uuid.uuid4()
    for i in range(count):
        _save_message(db, ctx, session_id, "user" if i % 2 == 0 else "assistant", f"message {i}")
    db.commit()
    return session_id


def test_router_sees_the_full_history_window_but_symptom_agent_gets_a_trimmed_slice(db, ctx, monkeypatch):
    total = SYMPTOM_AGENT_HISTORY_LIMIT + 5
    session_id = _seed_history(db, ctx, total)

    seen = {}

    def capture_intent(message, history=None):
        seen["router_history_len"] = len(history)
        return SYMPTOM_GENERAL

    monkeypatch.setattr("app.services.chat.classify_agent_intent", capture_intent)
    _patch_symptom_reply_capturing(monkeypatch, seen)

    handle_chat_message(db, ctx, "still there?", session_id)

    # The router decides intent from the full (un-trimmed) window...
    assert seen["router_history_len"] == total
    # ...but the agent that actually runs only gets its own, smaller slice.
    assert seen["agent_history_len"] == SYMPTOM_AGENT_HISTORY_LIMIT
    assert [row.content for row in seen["agent_history"]] == [f"message {i}" for i in range(5, total)]


def _patch_symptom_reply_capturing(monkeypatch, seen):
    def capture_reply(db, ctx, message, language, history):
        seen["agent_history_len"] = len(history)
        seen["agent_history"] = history
        return "reply"

    monkeypatch.setattr("app.services.chat.run_symptom_agent", capture_reply)


def test_appointment_agent_gets_its_own_smaller_history_slice(db, ctx, monkeypatch):
    total = APPOINTMENT_AGENT_HISTORY_LIMIT + 5
    session_id = _seed_history(db, ctx, total)
    _patch_intent(monkeypatch, APPOINTMENT)

    seen = {}

    def capture_reply(db, ctx, message, language, history):
        seen["agent_history_len"] = len(history)
        return "reply"

    monkeypatch.setattr("app.services.chat.run_appointment_agent", capture_reply)

    handle_chat_message(db, ctx, "book me a slot", session_id)

    assert seen["agent_history_len"] == APPOINTMENT_AGENT_HISTORY_LIMIT


def test_general_info_agent_gets_its_own_smaller_history_slice(db, ctx, monkeypatch):
    total = GENERAL_INFO_AGENT_HISTORY_LIMIT + 5
    session_id = _seed_history(db, ctx, total)
    _patch_intent(monkeypatch, GENERAL_INFO)

    seen = {}

    def capture_reply(db, ctx, message, language, history):
        seen["agent_history_len"] = len(history)
        return "reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_reply)

    handle_chat_message(db, ctx, "what are your hours", session_id)

    assert seen["agent_history_len"] == GENERAL_INFO_AGENT_HISTORY_LIMIT


def test_history_shorter_than_the_agent_limit_is_passed_through_unchanged(db, ctx, monkeypatch):
    session_id = _seed_history(db, ctx, 3)
    _patch_intent(monkeypatch, GENERAL_INFO)

    seen = {}

    def capture_reply(db, ctx, message, language, history):
        seen["agent_history_len"] = len(history)
        return "reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_reply)

    handle_chat_message(db, ctx, "hi", session_id)

    assert seen["agent_history_len"] == 3


def test_load_recent_history_breaks_same_timestamp_ties_by_insertion_order(db, ctx):
    """Mirrors the get_chat_history tiebreaker test in tests/test_chat_history_api.py,
    applied to _load_recent_history() instead: a single chat turn saves its user and
    assistant rows in the same DB transaction, and Postgres now() returns the same
    value for every statement within one transaction, so these rows can tie exactly
    on created_at. _load_recent_history() feeds the LLM's own conversation context —
    an out-of-order result here could confuse the model about who said what, not just
    look wrong on screen — so the `seq` tiebreaker must hold here too."""
    from datetime import datetime, timezone

    from app.services.chat import _load_recent_history

    session_id = uuid.uuid4()
    tied_timestamp = datetime.now(timezone.utc)

    rows_in_order = []
    for role, content in [
        ("user", "My name is Daud, remember this"),
        ("assistant", "Got it, Daud!"),
        ("user", "What is my name?"),
        ("assistant", "Your name is Daud."),
    ]:
        from app.models.conversation_memory import ConversationMemory

        row = ConversationMemory(
            clinic_id=ctx.clinic_id, session_id=session_id, user_id=ctx.user_id, role=role, content=content,
        )
        row.created_at = tied_timestamp
        db.add(row)
        db.flush()
        rows_in_order.append(row)

    assert len({row.created_at for row in rows_in_order}) == 1, (
        "test is only meaningful if these rows genuinely tie on created_at"
    )
    assert rows_in_order[0].seq < rows_in_order[1].seq < rows_in_order[2].seq < rows_in_order[3].seq

    history = _load_recent_history(db, ctx, session_id)

    assert [row.content for row in history] == [row.content for row in rows_in_order]


# --- language detection ---------------------------------------------------------------


def test_urdu_message_takes_urdu_path(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)

    seen = {}

    def fake_reply(db, ctx, message, language, history):
        seen["language"] = language
        return FALLBACK_MESSAGE_UR

    monkeypatch.setattr("app.services.chat.run_general_info_agent", fake_reply)

    result = handle_chat_message(db, ctx, "کلینک کب کھلتا ہے؟", None)

    assert seen["language"] == "ur"
    assert result.reply == FALLBACK_MESSAGE_UR


def test_english_message_takes_english_path(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, FALLBACK_MESSAGE)

    result = handle_chat_message(db, ctx, "When does the clinic open?", None)

    assert result.reply == FALLBACK_MESSAGE


def test_language_passed_through_to_the_agent(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)

    seen = {}

    def fake_reply(db, ctx, message, language, history):
        seen["language"] = language
        return "جواب"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", fake_reply)

    handle_chat_message(db, ctx, "کتنے شعبے ہیں؟", None)

    assert seen["language"] == "ur"


# --- list_sessions / delete_session ----------------------------------------------------


def test_list_sessions_returns_only_callers_own_sessions_correctly_titled(db, ctx, clinic, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "reply")

    other_patient = _patient(db, clinic)
    other_ctx = ClinicContext(clinic_id=clinic.id, user_id=other_patient.id, role="patient")

    handle_chat_message(db, ctx, "My first question for my own session", None)
    handle_chat_message(db, other_ctx, "A different patient's question", None)

    sessions = list_sessions(db, ctx)

    assert len(sessions) == 1
    assert sessions[0].title == "My first question for my own session"


def test_delete_session_returns_false_for_session_not_belonging_to_caller(db, ctx, clinic, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "reply")

    other_patient = _patient(db, clinic)
    other_ctx = ClinicContext(clinic_id=clinic.id, user_id=other_patient.id, role="patient")

    other_result = handle_chat_message(db, other_ctx, "Not the caller's session", None)

    deleted = delete_session(db, ctx, other_result.session_id)

    assert deleted is False
    # And it's untouched — the other patient can still see it.
    assert len(list_sessions(db, other_ctx)) == 1


def test_delete_session_returns_true_and_removes_own_session(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "reply")

    result = handle_chat_message(db, ctx, "Delete me later", None)

    deleted = delete_session(db, ctx, result.session_id)

    assert deleted is True
    assert list_sessions(db, ctx) == []


def test_delete_session_hard_deletes_conversation_memory_rows_not_just_hides_them(db, ctx, clinic, monkeypatch):
    """delete_session()'s own docstring promises a genuinely forgotten chat, not one
    merely hidden from the sidebar — this asserts that directly against the
    ConversationMemory table (a real SQL DELETE, no soft-delete flag), and that the
    delete is scoped so it can never remove another patient's or another clinic's
    rows, even when those rows share the same session_id by coincidence."""
    from sqlalchemy import select

    from app.models.conversation_memory import ConversationMemory

    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "reply")

    other_clinic = Clinic(name="Other Clinic", slug=f"other-{uuid.uuid4().hex[:8]}")
    db.add(other_clinic)
    db.flush()
    other_clinic_patient = _patient(db, other_clinic)
    other_clinic_ctx = ClinicContext(clinic_id=other_clinic.id, user_id=other_clinic_patient.id, role="patient")

    same_clinic_other_patient = _patient(db, clinic)
    same_clinic_other_ctx = ClinicContext(clinic_id=clinic.id, user_id=same_clinic_other_patient.id, role="patient")

    result = handle_chat_message(db, ctx, "first message", None)
    handle_chat_message(db, ctx, "second message", result.session_id)
    handle_chat_message(db, ctx, "third message", result.session_id)

    other_clinic_result = handle_chat_message(db, other_clinic_ctx, "other clinic message", None)
    same_clinic_other_result = handle_chat_message(db, same_clinic_other_ctx, "other patient message", None)

    own_rows_before = db.scalars(
        select(ConversationMemory).where(ConversationMemory.session_id == result.session_id)
    ).all()
    assert len(own_rows_before) >= 4  # 2 user turns + 2 assistant replies for the target session

    deleted = delete_session(db, ctx, result.session_id)
    assert deleted is True

    own_rows_after = db.scalars(
        select(ConversationMemory).where(ConversationMemory.session_id == result.session_id)
    ).all()
    assert own_rows_after == []

    # Untouched: another patient in the same clinic, and a patient in another clinic.
    other_clinic_rows = db.scalars(
        select(ConversationMemory).where(ConversationMemory.session_id == other_clinic_result.session_id)
    ).all()
    assert len(other_clinic_rows) >= 2

    same_clinic_other_rows = db.scalars(
        select(ConversationMemory).where(ConversationMemory.session_id == same_clinic_other_result.session_id)
    ).all()
    assert len(same_clinic_other_rows) >= 2


# --- tenancy scoping ---------------------------------------------------------------------


def test_cross_clinic_and_cross_patient_history_does_not_leak(db, monkeypatch):
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "reply")

    clinic_a = Clinic(name="Clinic A", slug=f"a-{uuid.uuid4().hex[:8]}")
    clinic_b = Clinic(name="Clinic B", slug=f"b-{uuid.uuid4().hex[:8]}")
    db.add_all([clinic_a, clinic_b])
    db.flush()

    patient_a = _patient(db, clinic_a)
    patient_b = _patient(db, clinic_b)
    ctx_a = ClinicContext(clinic_id=clinic_a.id, user_id=patient_a.id, role="patient")
    ctx_b = ClinicContext(clinic_id=clinic_b.id, user_id=patient_b.id, role="patient")

    handle_chat_message(db, ctx_a, "Clinic A's confidential patient question", None)

    seen_history = {}

    def capture_reply(db, ctx, message, language, history):
        seen_history["history"] = history
        return "reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_reply)

    handle_chat_message(db, ctx_b, "Unrelated clinic B question", None)

    history_contents = [row.content for row in seen_history["history"]]
    assert "Clinic A's confidential patient question" not in history_contents
    assert list_sessions(db, ctx_b)[0].title == "Unrelated clinic B question"
    assert len(list_sessions(db, ctx_a)) == 1


# --- delete-session memory verification (Task 3) --------------------------------------


def test_deleted_session_content_is_excluded_from_a_later_new_sessions_memory_digest(db, ctx, monkeypatch):
    """delete_session() must remove a session's rows so thoroughly that a later new
    session's memory-digest refresh (see app.services.memory_summary) has nothing of
    the deleted session's messages left to fold in."""
    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "ack")

    secret_turn = handle_chat_message(
        db, ctx, "My secret condition is a rare allergy to shellfish.", None
    )

    deleted = delete_session(db, ctx, secret_turn.session_id)
    assert deleted is True

    seen = {}

    def capture_kwargs(db, ctx, message, language, history):
        seen["history"] = history
        return "later reply"

    monkeypatch.setattr("app.services.chat.run_general_info_agent", capture_kwargs)

    handle_chat_message(db, ctx, "What allergy did I mention earlier?", None)

    assert seen["history"] == []


def test_delete_session_wipes_the_already_computed_memory_digest_not_just_the_transcript(db, ctx):
    """Reported bug: the raw transcript rows are deleted, but the memory digest
    (app.models.patient_memory_profile.PatientMemoryProfile) is a separately-stored
    LLM-generated summary — once a fact had already been folded into summary_text by
    an earlier real summarization run, deleting the session's transcript rows alone
    left that fact sitting in summary_text untouched, so it kept surfacing in new
    sessions after the chat that mentioned it was "deleted". delete_session() must
    wipe the digest too, not just the transcript it was built from."""
    from app.models.patient_memory_profile import PatientMemoryProfile
    from app.services.chat import _save_message

    session_id = uuid.uuid4()
    _save_message(db, ctx, session_id, "user", "Hi, my name is Daud.")
    db.commit()

    # Simulate an already-summarized digest, as a real summarization run would leave
    # behind — this is the state the reported bug actually depends on.
    profile = PatientMemoryProfile(
        clinic_id=ctx.clinic_id, user_id=ctx.user_id,
        summary_text="The patient's name is Daud.", last_summarized_seq=999999,
    )
    db.add(profile)
    db.commit()

    deleted = delete_session(db, ctx, session_id)
    assert deleted is True

    db.refresh(profile)
    assert profile.summary_text == ""
    assert profile.last_summarized_seq == 0


# --- red_flag persistence (survives a history reload, not just the live response) --


def test_red_flag_reply_is_persisted_with_red_flag_true(db, ctx):
    from sqlalchemy import select

    from app.models.conversation_memory import ConversationMemory

    # "chest pain" alone no longer auto-fires (see red_flag.py) — an explicit
    # "heart attack" still does. This never reaches the intent layer or any agent
    # at all — the red-flag gate short-circuits before either.
    result = handle_chat_message(db, ctx, "I think I'm having a heart attack", None)
    assert result.red_flag is True

    rows = db.execute(
        select(ConversationMemory)
        .where(ConversationMemory.session_id == result.session_id)
        .order_by(ConversationMemory.seq.asc())
    ).scalars().all()
    assert [(r.role, r.red_flag) for r in rows] == [("user", False), ("assistant", True)]


def test_ordinary_reply_is_persisted_with_red_flag_false(db, ctx, monkeypatch):
    from sqlalchemy import select

    from app.models.conversation_memory import ConversationMemory

    _patch_intent(monkeypatch, GENERAL_INFO)
    _patch_general_info_reply(monkeypatch, "The clinic is open 9am-5pm.")

    result = handle_chat_message(db, ctx, "what are your clinic hours?", None)
    assert result.red_flag is False

    rows = db.execute(
        select(ConversationMemory).where(ConversationMemory.session_id == result.session_id)
    ).scalars().all()
    assert all(r.red_flag is False for r in rows)


# --- diagnosis guard still runs on non-marker agent replies -------------------------


def test_diagnosis_guard_runs_on_plain_agent_text(db, ctx, monkeypatch):
    _patch_intent(monkeypatch, SYMPTOM_GENERAL)
    _patch_symptom_reply(monkeypatch, "You might have the flu.")

    result = handle_chat_message(db, ctx, "I have a fever and body aches", None)

    # enforce_no_diagnosis must have swapped this out — never the raw diagnostic text.
    assert "flu" not in result.reply.lower()


def test_diagnosis_guard_skips_marker_prefixed_replies(db, ctx, monkeypatch):
    from app.services.chat_markers import DOCTOR_OPTIONS_MARKER

    card = DOCTOR_OPTIONS_MARKER + '{"department_name": "Cardiology", "doctors": []}'
    _patch_intent(monkeypatch, SYMPTOM_GENERAL)
    _patch_symptom_reply(monkeypatch, card)

    result = handle_chat_message(db, ctx, "I have a fever and body aches", None)

    assert result.reply == card
