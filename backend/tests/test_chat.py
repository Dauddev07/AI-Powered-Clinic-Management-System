import uuid

import pytest

from app.core.tenancy import ClinicContext
from app.models.clinic import Clinic
from app.models.user import User
from app.rag.retrieval import FALLBACK_MESSAGE, FALLBACK_MESSAGE_UR, RetrievalResult
from app.services.chat import delete_session, handle_chat_message, list_sessions


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


def _matched_result(chunks=("The clinic has 5 departments.",), score=0.9):
    return RetrievalResult(matched=True, best_score=score, chunks=list(chunks), fallback_message=None)


def _unmatched_result(score=0.1):
    return RetrievalResult(matched=False, best_score=score, chunks=[], fallback_message=FALLBACK_MESSAGE)


def _patch_retrieval_path_defaults(monkeypatch):
    """Most tests aren't about query rewriting or message classification themselves —
    keep the query unchanged and force the retrieve()-gated path deterministically, so
    assertions about what reached `retrieve`/`run_chat_agent` stay simple and no test
    depends on a real LLM classification call for a message that wasn't crafted with
    the classifier's heuristic in mind."""
    monkeypatch.setattr("app.services.chat.rewrite_query", lambda message, history: message)
    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "knowledge_seeking")


def _patch_agent_reply(monkeypatch, reply):
    """Non-conversational/personal-recall messages go through the tool-calling agent
    (app.services.chat.run_chat_agent), not a plain grounded completion — see
    chat.py's module-level routing. Stubbing it out here avoids every knowledge-
    seeking-path test making a real, slow, network-dependent LLM call just to reach
    an assertion that doesn't care about the model's actual behavior."""
    monkeypatch.setattr("app.services.chat.run_chat_agent", lambda **kwargs: reply)


# --- grounded reply vs. fallback -----------------------------------------------------


def test_above_threshold_reaches_llm_and_returns_grounded_reply(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())

    calls = []

    def fake_run_chat_agent(**kwargs):
        calls.append(kwargs)
        return "This clinic has 5 departments."

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    result = handle_chat_message(db, ctx, "How many departments does this clinic have?", None)

    assert result.reply == "This clinic has 5 departments."
    assert len(calls) == 1
    assert calls[0]["context_chunks"] == ["The clinic has 5 departments."]


def test_below_threshold_still_calls_agent_with_empty_context(db, ctx, monkeypatch):
    # The agent is always invoked for a knowledge-seeking message, even below the KB
    # similarity threshold — this is what lets a booking/triage message (which never
    # lives in the KB) through the same path. Grounding safety now lives in the
    # agent's own system prompt (it's told to decline a genuine out-of-scope
    # knowledge question when context is "(none)"), not in a hard Python-level
    # bypass — so this test verifies the agent IS reached, with empty context, and
    # simulates a well-behaved model correctly declining.
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _unmatched_result())

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["context_chunks"] = kwargs["context_chunks"]
        return FALLBACK_MESSAGE

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    result = handle_chat_message(db, ctx, "What's the capital of France?", None)

    assert seen["context_chunks"] == []
    assert result.reply == FALLBACK_MESSAGE


# --- cross-session memory -------------------------------------------------------------


def test_new_session_does_not_replay_a_prior_sessions_full_transcript(db, ctx, monkeypatch):
    # session_id IS now a real memory boundary for the verbatim transcript (see
    # module docstring) — a genuinely new session must never see another session's
    # messages in `history`, only its own (empty, since it's brand new).
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    monkeypatch.setattr("app.services.chat.refresh_patient_summary_for_new_session", lambda db, cid, uid: "")
    _patch_agent_reply(monkeypatch, "ack")

    first = handle_chat_message(db, ctx, "My name is Ali and I have a headache.", None)

    seen = {}

    def capture_reply(**kwargs):
        seen["history"] = kwargs["history"]
        return "second reply"

    monkeypatch.setattr("app.services.chat.run_chat_agent", capture_reply)

    second = handle_chat_message(db, ctx, "What did I just tell you my name was?", None)

    assert second.session_id != first.session_id, "test is only meaningful across two distinct sessions"
    assert seen["history"] == []


def test_continuing_session_replays_its_own_full_transcript(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    monkeypatch.setattr("app.services.chat.refresh_patient_summary_for_new_session", lambda db, cid, uid: "")
    _patch_agent_reply(monkeypatch, "ack")

    first = handle_chat_message(db, ctx, "My name is Ali and I have a headache.", None)

    seen = {}

    def capture_reply(**kwargs):
        seen["history"] = kwargs["history"]
        return "second reply"

    monkeypatch.setattr("app.services.chat.run_chat_agent", capture_reply)

    handle_chat_message(db, ctx, "What did I just tell you my name was?", first.session_id)

    history_contents = [row.content for row in seen["history"]]
    assert "My name is Ali and I have a headache." in history_contents
    assert "ack" in history_contents


def test_new_session_loads_patient_memory_digest_and_continuing_session_does_not(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())

    refresh_calls = []

    def fake_refresh(db, clinic_id, user_id):
        refresh_calls.append((clinic_id, user_id))
        return "Patient previously mentioned recurring headaches."

    monkeypatch.setattr("app.services.chat.refresh_patient_summary_for_new_session", fake_refresh)

    seen = {}

    def capture_reply(**kwargs):
        seen["patient_memory"] = kwargs["patient_memory"]
        return "reply"

    monkeypatch.setattr("app.services.chat.run_chat_agent", capture_reply)

    first = handle_chat_message(db, ctx, "hi", None)
    assert len(refresh_calls) == 1
    assert refresh_calls[0] == (ctx.clinic_id, ctx.user_id)
    assert seen["patient_memory"] == "Patient previously mentioned recurring headaches."

    # A second message in the SAME (now continuing) session must not call refresh
    # again, and gets no digest — its own real transcript already has the context.
    handle_chat_message(db, ctx, "still there?", first.session_id)
    assert len(refresh_calls) == 1
    assert seen["patient_memory"] == ""


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
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _unmatched_result())

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["language"] = kwargs["language"]
        return FALLBACK_MESSAGE_UR

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    result = handle_chat_message(db, ctx, "کلینک کب کھلتا ہے؟", None)

    assert seen["language"] == "ur"
    assert result.reply == FALLBACK_MESSAGE_UR


def test_english_message_takes_english_path(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _unmatched_result())
    _patch_agent_reply(monkeypatch, FALLBACK_MESSAGE)

    result = handle_chat_message(db, ctx, "When does the clinic open?", None)

    assert result.reply == FALLBACK_MESSAGE


def test_language_passed_through_to_llm_for_grounded_reply(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["language"] = kwargs["language"]
        return "جواب"

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    handle_chat_message(db, ctx, "کتنے شعبے ہیں؟", None)

    assert seen["language"] == "ur"


# --- list_sessions / delete_session ----------------------------------------------------


def test_list_sessions_returns_only_callers_own_sessions_correctly_titled(db, ctx, clinic, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "reply")

    other_patient = _patient(db, clinic)
    other_ctx = ClinicContext(clinic_id=clinic.id, user_id=other_patient.id, role="patient")

    handle_chat_message(db, ctx, "My first question for my own session", None)
    handle_chat_message(db, other_ctx, "A different patient's question", None)

    sessions = list_sessions(db, ctx)

    assert len(sessions) == 1
    assert sessions[0].title == "My first question for my own session"


def test_delete_session_returns_false_for_session_not_belonging_to_caller(db, ctx, clinic, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "reply")

    other_patient = _patient(db, clinic)
    other_ctx = ClinicContext(clinic_id=clinic.id, user_id=other_patient.id, role="patient")

    other_result = handle_chat_message(db, other_ctx, "Not the caller's session", None)

    deleted = delete_session(db, ctx, other_result.session_id)

    assert deleted is False
    # And it's untouched — the other patient can still see it.
    assert len(list_sessions(db, other_ctx)) == 1


def test_delete_session_returns_true_and_removes_own_session(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "reply")

    result = handle_chat_message(db, ctx, "Delete me later", None)

    deleted = delete_session(db, ctx, result.session_id)

    assert deleted is True
    assert list_sessions(db, ctx) == []


# --- tenancy scoping ---------------------------------------------------------------------


def test_cross_clinic_and_cross_patient_history_does_not_leak(db, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "reply")

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

    def capture_reply(**kwargs):
        seen_history["history"] = kwargs["history"]
        return "reply"

    monkeypatch.setattr("app.services.chat.run_chat_agent", capture_reply)

    handle_chat_message(db, ctx_b, "Unrelated clinic B question", None)

    history_contents = [row.content for row in seen_history["history"]]
    assert "Clinic A's confidential patient question" not in history_contents
    assert list_sessions(db, ctx_b)[0].title == "Unrelated clinic B question"
    assert len(list_sessions(db, ctx_a)) == 1


# --- query rewriting changes the retrieval outcome (Task 1) --------------------------


def test_query_rewriting_flips_a_conversationally_phrased_query_from_fallback_to_grounded(db, ctx, monkeypatch):
    """The raw, conversationally-diluted message would score below threshold on its
    own; rewrite_query cleans it into a standalone question that clears it. Proves
    handle_chat_message actually uses the rewritten query for retrieval, not the raw
    message, per the module's contract."""
    raw_message = "not on the info but overall in total how many departments does this clinic have"
    rewritten = "How many departments does this clinic have?"

    monkeypatch.setattr("app.services.chat.rewrite_query", lambda message, history: rewritten)

    def fake_retrieve(db, clinic_id, query):
        if query == rewritten:
            return _matched_result()
        return _unmatched_result()

    monkeypatch.setattr("app.services.chat.retrieve", fake_retrieve)
    _patch_agent_reply(monkeypatch, "5 departments.")

    result = handle_chat_message(db, ctx, raw_message, None)

    assert result.reply == "5 departments."

    # The raw message is still what's persisted/shown, untouched by rewriting (the
    # sidebar title truncates long messages, so compare the stored row directly).
    from app.models.conversation_memory import ConversationMemory
    from sqlalchemy import select

    stored = db.execute(
        select(ConversationMemory.content).where(
            ConversationMemory.session_id == result.session_id,
            ConversationMemory.role == "user",
        )
    ).scalar()
    assert stored == raw_message


def test_query_rewrite_failure_falls_back_to_raw_message_without_blocking_the_turn(db, ctx, monkeypatch):
    # rewrite_query itself is responsible for catching an internal LLM failure and
    # returning the raw message unchanged — but this test guards handle_chat_message
    # against ever letting that failure propagate and block the turn, in case that
    # contract regresses. Raw must fail to clear the threshold on its own so the
    # rewrite-rescue path (where the guarded failure lives) actually runs.
    from app.services import query_rewrite as query_rewrite_module

    monkeypatch.setattr(query_rewrite_module, "ChatGroq", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _unmatched_result())
    _patch_agent_reply(monkeypatch, FALLBACK_MESSAGE)
    # Message ends in "?" so the classifier's heuristic deterministically calls it
    # knowledge_seeking without needing an LLM call — keeps this test exercising the
    # real retrieve()-gated path the rewrite-failure guard is meant to protect.

    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "fake-key-for-this-test")

    result = handle_chat_message(db, ctx, "What are the clinic hours?", None)

    # rewrite_query silently falls back to the raw message on its own internal
    # failure, so the (still-unmatched) raw result stands — the turn completes with
    # the fixed fallback rather than raising or hanging.
    assert result.reply == FALLBACK_MESSAGE


def test_clean_standalone_question_is_never_rewritten_even_with_unrelated_prior_history(db, ctx, monkeypatch):
    """Reproduces the reported bug: history is loaded cross-session (see module
    docstring), so a brand-new, unambiguous question ("just tell me the names of
    doctors in cardiology") used to get silently rewritten using leftover context
    from an unrelated prior conversation (a chest-tightness/cardiology-vs-pulmonology
    exchange) even though the raw question already clears the threshold on its own.
    rewrite_query must not run at all when the raw message already matches."""
    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "knowledge_seeking")

    calls = {"rewrite": 0}

    def counting_rewrite(message, history):
        calls["rewrite"] += 1
        return "a rewritten query that must never be used"

    monkeypatch.setattr("app.services.chat.rewrite_query", counting_rewrite)

    clean_question = "just tell me the names of doctors in cardiology"

    def fake_retrieve(db, clinic_id, query):
        assert query == clean_question, "retrieve() must receive the raw message untouched"
        return _matched_result(chunks=["Cardiology: Dr. Ahmed Farooq, Dr. Farhan Malik."])

    monkeypatch.setattr("app.services.chat.retrieve", fake_retrieve)
    _patch_agent_reply(monkeypatch, "Dr. Ahmed Farooq and Dr. Farhan Malik.")

    # Unrelated prior conversation already sitting in cross-session history, from a
    # DIFFERENT session_id — this is what a "new chat" doesn't actually leave behind.
    from app.models.conversation_memory import ConversationMemory

    db.add(ConversationMemory(
        clinic_id=ctx.clinic_id, session_id=uuid.uuid4(), user_id=ctx.user_id, role="user",
        content="I have chest tightness and a cough, is that cardiology or pulmonology?",
    ))
    db.add(ConversationMemory(
        clinic_id=ctx.clinic_id, session_id=uuid.uuid4(), user_id=ctx.user_id, role="assistant",
        content="That could be either — can you tell me more about your breathing?",
    ))
    db.flush()

    result = handle_chat_message(db, ctx, clean_question, None)

    assert calls["rewrite"] == 0, "rewrite_query must not run when the raw message already clears the threshold"
    assert result.reply == "Dr. Ahmed Farooq and Dr. Farhan Malik."


def test_rewrite_rescue_path_still_runs_when_raw_message_fails_on_its_own(db, ctx, monkeypatch):
    # The other half of the contract: rewriting must still kick in as a rescue when
    # the raw message genuinely doesn't clear the threshold by itself.
    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "knowledge_seeking")

    raw_message = "not on the info but overall in total how many departments does this clinic have"
    rewritten = "How many departments does this clinic have?"

    monkeypatch.setattr("app.services.chat.rewrite_query", lambda message, history: rewritten)

    def fake_retrieve(db, clinic_id, query):
        if query == rewritten:
            return _matched_result()
        return _unmatched_result()

    monkeypatch.setattr("app.services.chat.retrieve", fake_retrieve)
    _patch_agent_reply(monkeypatch, "5 departments.")

    result = handle_chat_message(db, ctx, raw_message, None)

    assert result.reply == "5 departments."


# --- delete-session memory verification (Task 3) --------------------------------------


def test_deleted_session_content_is_excluded_from_a_later_new_sessions_memory_digest(db, ctx, monkeypatch):
    """delete_session() must remove a session's rows so thoroughly that a later new
    session's memory-digest refresh (see app.services.memory_summary) has nothing of
    the deleted session's messages left to fold in."""
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "ack")

    secret_turn = handle_chat_message(
        db, ctx, "My secret condition is a rare allergy to shellfish.", None
    )

    deleted = delete_session(db, ctx, secret_turn.session_id)
    assert deleted is True

    seen = {}

    def capture_kwargs(**kwargs):
        seen["history"] = kwargs["history"]
        seen["patient_memory"] = kwargs["patient_memory"]
        return "later reply"

    monkeypatch.setattr("app.services.chat.run_chat_agent", capture_kwargs)

    handle_chat_message(db, ctx, "What allergy did I mention earlier?", None)

    assert seen["history"] == []
    history_contents_never_seen = "My secret condition is a rare allergy to shellfish."
    assert history_contents_never_seen not in (seen["patient_memory"] or "")


# --- conversational passthrough (skips the retrieval gate) ---------------------------


def test_conversational_message_skips_retrieval_gate_and_gets_direct_reply(db, ctx, monkeypatch):
    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "conversational")

    def fail_if_called(db, clinic_id, query):
        raise AssertionError("retrieve() must not be called for conversational input")

    monkeypatch.setattr("app.services.chat.retrieve", fail_if_called)

    seen = {}

    def fake_get_chat_reply(**kwargs):
        seen["context_chunks"] = kwargs["context_chunks"]
        return "Got it — your favorite color is green!"

    monkeypatch.setattr("app.services.chat.get_chat_reply", fake_get_chat_reply)

    result = handle_chat_message(db, ctx, "My favorite color is green, please remember that", None)

    assert result.reply == "Got it — your favorite color is green!"
    assert result.reply != FALLBACK_MESSAGE
    # No KB grounding to hand it — the conversational path has nothing to retrieve.
    assert seen["context_chunks"] == []


@pytest.mark.parametrize("message", ["Hi", "Thanks"])
def test_greeting_and_thanks_classified_conversational_and_bypass_retrieval(db, ctx, monkeypatch, message):
    calls = {"retrieve": 0}

    def counting_retrieve(db, clinic_id, query):
        calls["retrieve"] += 1
        return _unmatched_result()

    monkeypatch.setattr("app.services.chat.retrieve", counting_retrieve)
    monkeypatch.setattr("app.services.chat.get_chat_reply", lambda **kwargs: "Hello! How can I help?")

    result = handle_chat_message(db, ctx, message, None)

    assert calls["retrieve"] == 0, "retrieve() should never run for plain small talk"
    assert result.reply == "Hello! How can I help?"
    assert result.reply != FALLBACK_MESSAGE


def test_ambiguous_off_topic_statement_uses_llm_fallback_classifier_not_heuristic_default(db, ctx, monkeypatch):
    # "My favorite color is green, please remember that" has no "?", no keyword, and
    # is too long for the heuristic's short-message shortcut — it must reach the LLM
    # fallback classifier rather than the heuristic silently guessing either way.
    from app.services import message_classifier

    seen = {}

    def fake_llm_classify(message):
        seen["called_with"] = message
        return message_classifier.CONVERSATIONAL

    monkeypatch.setattr("app.services.chat.rewrite_query", lambda message, history: message)
    monkeypatch.setattr(message_classifier, "_llm_classify", fake_llm_classify)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: (_ for _ in ()).throw(
        AssertionError("retrieve() must not run once the LLM classifier says conversational")
    ))
    monkeypatch.setattr("app.services.chat.get_chat_reply", lambda **kwargs: "noted")

    message = "My favorite color is green, please remember that"
    result = handle_chat_message(db, ctx, message, None)

    assert seen["called_with"] == message
    assert result.reply == "noted"


# --- genuine KB / out-of-scope behavior is unchanged by the new classifier -----------


def test_genuine_out_of_scope_question_still_returns_fixed_fallback(db, ctx, monkeypatch):
    # A real factual question ("what's the weather today") is knowledge_seeking (has
    # a leading question word / "?"), so it must still go through retrieve() and
    # reach the agent with empty context when nothing clears the threshold — the
    # agent's own system prompt (not a hard Python bypass) is what's now responsible
    # for declining it, so this simulates a well-behaved model doing exactly that.
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _unmatched_result())
    _patch_agent_reply(monkeypatch, FALLBACK_MESSAGE)

    result = handle_chat_message(db, ctx, "What's the weather today?", None)

    assert result.reply == FALLBACK_MESSAGE


def test_real_kb_question_still_grounded_exactly_as_before(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "This clinic has 5 departments.")

    result = handle_chat_message(db, ctx, "How many departments does this clinic have?", None)

    assert result.reply == "This clinic has 5 departments."


def test_symptom_like_statement_is_never_classified_conversational_by_heuristic():
    # Guards the "err toward knowledge_seeking" constraint directly against the
    # heuristic: a loosely clinical statement must never fall through to the
    # conversational shortcut, even without a question mark.
    from app.services.message_classifier import KNOWLEDGE_SEEKING, _heuristic_classify

    assert _heuristic_classify("I have a fever and body aches") == KNOWLEDGE_SEEKING
    assert _heuristic_classify("my stomach hurts") == KNOWLEDGE_SEEKING


# --- item 2: symptom messages never touch KB retrieval, get real department context -


def test_symptom_message_never_calls_retrieve_and_gets_real_department_names(db, ctx, clinic, monkeypatch):
    from app.models.department import Department

    db.add(Department(clinic_id=clinic.id, name="Cardiology"))
    db.add(Department(clinic_id=clinic.id, name="Dermatology"))
    db.flush()

    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "knowledge_seeking")

    def fail_if_called(db, clinic_id, query):
        raise AssertionError("retrieve() (hospital_info) must not be called for a symptom message")

    monkeypatch.setattr("app.services.chat.retrieve", fail_if_called)

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["context_chunks"] = kwargs["context_chunks"]
        return "Could you tell me how long you've had this?"

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    result = handle_chat_message(db, ctx, "I have a fever and a bad cough", None)

    assert len(seen["context_chunks"]) == 1
    assert "Cardiology" in seen["context_chunks"][0]
    assert "Dermatology" in seen["context_chunks"][0]
    assert result.reply == "Could you tell me how long you've had this?"


def test_non_symptom_knowledge_message_still_uses_ordinary_retrieval(db, ctx, monkeypatch):
    # Unaffected-by-item-2 control: a plain logistics question still goes through
    # retrieve() exactly as before — only symptom messages skip it.
    _patch_retrieval_path_defaults(monkeypatch)
    calls = {"retrieve": 0}

    def counting_retrieve(db, clinic_id, query):
        calls["retrieve"] += 1
        return _matched_result()

    monkeypatch.setattr("app.services.chat.retrieve", counting_retrieve)
    _patch_agent_reply(monkeypatch, "We're open 9am-5pm.")

    result = handle_chat_message(db, ctx, "What are your clinic hours?", None)

    assert calls["retrieve"] == 1
    assert result.reply == "We're open 9am-5pm."


# --- personal recall passthrough (skips the retrieval gate, same as conversational) --


def test_personal_recall_question_skips_retrieval_gate_and_answers_from_history(db, ctx, monkeypatch):
    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "personal_recall")

    def fail_if_called(db, clinic_id, query):
        raise AssertionError("retrieve() must not be called for a personal-recall question")

    monkeypatch.setattr("app.services.chat.retrieve", fail_if_called)

    seen = {}

    def fake_get_chat_reply(**kwargs):
        seen["context_chunks"] = kwargs["context_chunks"]
        seen["history"] = kwargs["history"]
        return "Your name is Daud."

    monkeypatch.setattr("app.services.chat.get_chat_reply", fake_get_chat_reply)

    first = handle_chat_message(db, ctx, "My name is Daud, remember this", None)
    result = handle_chat_message(db, ctx, "What is my name?", first.session_id)

    assert result.reply == "Your name is Daud."
    assert result.reply != FALLBACK_MESSAGE
    # No KB grounding needed — the answer comes from THIS session's own conversation
    # history, which is passed through untouched regardless of the retrieval bypass.
    assert seen["context_chunks"] == []
    history_contents = [row.content for row in seen["history"]]
    assert "My name is Daud, remember this" in history_contents


def test_personal_recall_in_a_new_session_uses_the_memory_digest_not_the_fallback(db, ctx, monkeypatch):
    # session_id is now a real memory boundary (see module docstring) — "what is my
    # name?" in a brand-new thread has no history of its own, so it must instead be
    # answered from the patient's cross-session memory digest (refreshed at the start
    # of every new session), not silently fall back to the fixed "I don't know" reply.
    monkeypatch.setattr("app.services.chat.classify_message_intent", lambda message, history=None: "personal_recall")
    monkeypatch.setattr(
        "app.services.chat.retrieve",
        lambda db, clinic_id, query: (_ for _ in ()).throw(
            AssertionError("retrieve() must not run for personal_recall")
        ),
    )
    monkeypatch.setattr(
        "app.services.chat.refresh_patient_summary_for_new_session",
        lambda db, clinic_id, user_id: "The patient's name is Daud.",
    )

    seen = {}

    def fake_get_chat_reply(**kwargs):
        seen["patient_memory"] = kwargs["patient_memory"]
        seen["history"] = kwargs["history"]
        return "Your name is Daud."

    monkeypatch.setattr("app.services.chat.get_chat_reply", fake_get_chat_reply)

    result = handle_chat_message(db, ctx, "What is my name?", None)

    assert result.reply == "Your name is Daud."
    assert result.reply != FALLBACK_MESSAGE
    assert seen["history"] == []
    assert seen["patient_memory"] == "The patient's name is Daud."


def test_heuristic_classifies_self_referential_recall_question_as_personal_recall():
    from app.services.message_classifier import PERSONAL_RECALL, _heuristic_classify

    assert _heuristic_classify("What is my name?") == PERSONAL_RECALL
    assert _heuristic_classify("What did I just tell you my name was?") == PERSONAL_RECALL


def test_real_clinic_question_and_out_of_scope_trivia_never_classified_personal_recall(db, ctx, monkeypatch):
    # Guards the "must never become a loophole" constraint: a genuine clinic question
    # still goes through retrieval and grounds normally, and genuine out-of-scope
    # trivia still hits the exact fixed fallback — neither is reclassified as
    # personal_recall just because it's phrased as a question.
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "The clinic is open 9am-5pm.")

    result = handle_chat_message(db, ctx, "what are your clinic hours?", None)
    assert result.reply == "The clinic is open 9am-5pm."

    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _unmatched_result())
    _patch_agent_reply(monkeypatch, FALLBACK_MESSAGE)

    result = handle_chat_message(db, ctx, "what's the capital of France?", None)
    assert result.reply == FALLBACK_MESSAGE


# --- red_flag persistence (survives a history reload, not just the live response) --


def test_red_flag_reply_is_persisted_with_red_flag_true(db, ctx):
    from sqlalchemy import select

    from app.models.conversation_memory import ConversationMemory

    # "chest pain" alone no longer auto-fires (see red_flag.py) — an explicit
    # "heart attack" still does.
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

    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())
    _patch_agent_reply(monkeypatch, "The clinic is open 9am-5pm.")

    result = handle_chat_message(db, ctx, "what are your clinic hours?", None)
    assert result.red_flag is False

    rows = db.execute(
        select(ConversationMemory).where(ConversationMemory.session_id == result.session_id)
    ).scalars().all()
    assert all(r.red_flag is False for r in rows)


# --- token-reduction: include_triage passed to run_chat_agent (item 1 of the
# token-usage work) — the large symptom-triage prompt block should only be sent on
# turns that are actually symptom-related, current or earlier in the conversation. ---


def test_symptom_message_reaches_agent_with_include_triage_true(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["include_triage"] = kwargs["include_triage"]
        return "Let's figure out which department fits."

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    handle_chat_message(db, ctx, "I have a fever and body aches", None)

    assert seen["include_triage"] is True


def test_non_symptom_message_reaches_agent_with_include_triage_false(db, ctx, monkeypatch):
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["include_triage"] = kwargs["include_triage"]
        return "We're open 9am-5pm."

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    handle_chat_message(db, ctx, "what are your clinic hours?", None)

    assert seen["include_triage"] is False


def test_non_symptom_followup_still_gets_triage_when_conversation_started_with_a_symptom(db, ctx, monkeypatch):
    # Reproduces the mid-triage scenario the flag has to handle correctly: the
    # patient's first message names a symptom (include_triage=True), but their next
    # reply answering a screening question ("severe, started yesterday") contains no
    # symptom keyword on its own — it must still get the triage rules, since the
    # conversation is still mid-flow.
    _patch_retrieval_path_defaults(monkeypatch)
    monkeypatch.setattr("app.services.chat.retrieve", lambda db, clinic_id, query: _matched_result())

    monkeypatch.setattr("app.services.chat.run_chat_agent", lambda **kwargs: "Is it severe or mild?")
    first = handle_chat_message(db, ctx, "I have a headache", None)

    seen = {}

    def fake_run_chat_agent(**kwargs):
        seen["include_triage"] = kwargs["include_triage"]
        return "Got it, checking availability."

    monkeypatch.setattr("app.services.chat.run_chat_agent", fake_run_chat_agent)

    handle_chat_message(db, ctx, "severe, started yesterday", first.session_id)

    assert seen["include_triage"] is True
