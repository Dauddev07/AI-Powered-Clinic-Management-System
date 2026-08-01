from types import SimpleNamespace

import pytest

from app.services.chat_markers import DOCTOR_OPTIONS_MARKER
from app.services.message_classifier import (
    CONVERSATIONAL,
    KNOWLEDGE_SEEKING,
    PERSONAL_RECALL,
    _heuristic_classify,
    classify_message_intent,
    is_symptom_message,
)


def _history_row(role, content):
    return SimpleNamespace(role=role, content=content)


# --- clear-cut conversational cases (heuristic, no LLM needed) -----------------------


@pytest.mark.parametrize(
    "message",
    ["Hi", "hello", "Hey!", "Thanks", "thank you.", "ok", "Okay!", "Great", "Bye", "lol"],
)
def test_short_greetings_and_acknowledgments_classified_conversational(message):
    assert _heuristic_classify(message) == CONVERSATIONAL


# --- clear-cut knowledge-seeking cases (heuristic, no LLM needed) --------------------


@pytest.mark.parametrize(
    "message",
    [
        "What are the clinic hours?",
        "How many departments does this clinic have?",
        "Can I book an appointment?",
        "I have a fever and body aches",
        "my stomach hurts",
        "mera sar dard kar raha hai",  # Roman Urdu symptom keyword
        "کلینک کب کھلتا ہے؟",  # Urdu-script question mark
    ],
)
def test_questions_and_symptom_statements_classified_knowledge_seeking(message):
    assert _heuristic_classify(message) == KNOWLEDGE_SEEKING


# --- the "err toward knowledge_seeking" constraint --------------------------------


def test_loosely_clinical_statement_never_falls_through_to_conversational():
    # No question mark, but "fever"/"hurts" are strong enough signals that this must
    # never be treated as small talk, per the task's explicit err-toward-grounding
    # requirement.
    assert _heuristic_classify("I have a fever") == KNOWLEDGE_SEEKING
    assert _heuristic_classify("my knee hurts") == KNOWLEDGE_SEEKING


def test_empty_message_is_conversational_not_a_crash():
    assert _heuristic_classify("") == CONVERSATIONAL
    assert _heuristic_classify("   ") == CONVERSATIONAL


# --- ambiguous cases fall through to the LLM classifier, and fail safe --------------


def test_ambiguous_statement_is_not_resolved_by_heuristic_alone():
    # No question mark, no keyword, too long for the short-message shortcut — must
    # be left for the LLM fallback rather than guessed by the heuristic.
    assert _heuristic_classify("My favorite color is green, please remember that") is None


def test_classify_message_intent_defaults_to_knowledge_seeking_when_llm_unavailable(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "")

    # Ambiguous message with no LLM key configured must fail safe toward grounding,
    # never silently skip it.
    result = classify_message_intent("My favorite color is green, please remember that")

    assert result == KNOWLEDGE_SEEKING


def test_classify_message_intent_defaults_to_knowledge_seeking_when_llm_call_raises(monkeypatch):
    from app.core.config import settings
    from app.services import message_classifier

    monkeypatch.setattr(settings, "LLM_API_KEY", "fake-key-for-this-test")
    monkeypatch.setattr(
        message_classifier, "ChatGroq", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    result = classify_message_intent("My favorite color is green, please remember that")

    assert result == KNOWLEDGE_SEEKING


def test_ambiguous_message_uses_llm_result_when_available(monkeypatch):
    from app.services import message_classifier

    class _FakeResponse:
        content = "CONVERSATIONAL"

    class _FakeLLM:
        def invoke(self, messages):
            return _FakeResponse()

    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "fake-key-for-this-test")
    monkeypatch.setattr(message_classifier, "ChatGroq", lambda **kwargs: _FakeLLM())

    result = classify_message_intent("My favorite color is green, please remember that")

    assert result == CONVERSATIONAL


# --- personal recall (heuristic, no LLM needed) --------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What is my name?",
        "what's my name?",
        "What did I just tell you my name was?",
        "What did I tell you earlier?",
        "Do you remember what I said?",
        "What do you know about me?",
        "Remind me what I told you.",
    ],
)
def test_self_referential_recall_questions_classified_personal_recall(message):
    assert _heuristic_classify(message) == PERSONAL_RECALL


# --- regression: apostrophe-optional contractions (item 3) --------------------------


@pytest.mark.parametrize(
    "message",
    ["whats my name", "whats my name?", "what did i say", "What did I say?", "Whats My Name?"],
)
def test_apostrophe_optional_personal_recall_phrasing_still_recognized(message):
    # _PERSONAL_RECALL_RE previously required a literal apostrophe ("what's my
    # name") and failed on how most people actually type on mobile ("whats my
    # name") — every contraction in the pattern must be apostrophe-optional.
    assert _heuristic_classify(message) == PERSONAL_RECALL


def test_personal_recall_checked_before_the_question_mark_shortcut():
    # "What is my name?" has a "?" — must not fall into the generic
    # knowledge_seeking bucket just because of that; the personal-recall pattern
    # takes priority.
    assert _heuristic_classify("What is my name?") == PERSONAL_RECALL


# --- personal recall must never become a loophole for real questions ----------------


@pytest.mark.parametrize(
    "message",
    [
        "what are your clinic hours?",
        "what's the capital of France?",
        "What time does the clinic open?",
        "What is a fever?",
        "What do doctors recommend for a headache?",
    ],
)
def test_real_clinical_or_trivia_questions_never_classified_personal_recall(message):
    assert _heuristic_classify(message) != PERSONAL_RECALL


def test_statement_sharing_personal_info_reaches_llm_fallback_not_forced_either_way():
    # "My name is Daud, remember this" is a statement, not a recall question — the
    # heuristic correctly leaves it ambiguous (either CONVERSATIONAL or
    # PERSONAL_RECALL is fine downstream, both skip retrieval) rather than guessing.
    assert _heuristic_classify("My name is Daud, remember this") is None


def test_ambiguous_message_can_resolve_to_personal_recall_via_llm(monkeypatch):
    from app.services import message_classifier

    class _FakeResponse:
        content = "PERSONAL_RECALL"

    class _FakeLLM:
        def invoke(self, messages):
            return _FakeResponse()

    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "fake-key-for-this-test")
    monkeypatch.setattr(message_classifier, "ChatGroq", lambda **kwargs: _FakeLLM())

    result = classify_message_intent("My name is Daud, remember this")

    assert result == PERSONAL_RECALL


def test_llm_fallback_still_defaults_to_knowledge_seeking_over_personal_recall_when_ambiguous_response(monkeypatch):
    from app.services import message_classifier

    class _FakeResponse:
        content = "something unparseable"

    class _FakeLLM:
        def invoke(self, messages):
            return _FakeResponse()

    from app.core.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "fake-key-for-this-test")
    monkeypatch.setattr(message_classifier, "ChatGroq", lambda **kwargs: _FakeLLM())

    result = classify_message_intent("My favorite color is green, please remember that")

    assert result == KNOWLEDGE_SEEKING


# --- regression: a bare reply mid-triage must reach the agent, not small talk -------


@pytest.mark.parametrize("reply", ["yes", "no", "ok", "sure", "Yes.", "OK!"])
def test_bare_reply_after_a_clarifying_question_reaches_knowledge_seeking(reply):
    # Exactly what a patient sends answering a mid-triage clarifying question — must
    # reach the tool-calling agent (and diagnosis_guard.enforce_no_diagnosis, which
    # chat.py only ever applies to the agent's output), not a generic CONVERSATIONAL
    # reply that skips both.
    history = [
        _history_row("user", "I have chest tightness"),
        _history_row("assistant", "How long have you had this discomfort?"),
    ]
    assert _heuristic_classify(reply, history) == KNOWLEDGE_SEEKING
    assert classify_message_intent(reply, history) == KNOWLEDGE_SEEKING


@pytest.mark.parametrize("reply", ["yes", "ok"])
def test_bare_reply_after_a_doctor_options_card_reaches_knowledge_seeking(reply):
    history = [
        _history_row("user", "cardiology please"),
        _history_row("assistant", DOCTOR_OPTIONS_MARKER + '{"department_name": "Cardiology", "doctors": []}'),
    ]
    assert _heuristic_classify(reply, history) == KNOWLEDGE_SEEKING


def test_standalone_ok_with_no_preceding_question_still_behaves_as_small_talk():
    # No history at all — unchanged from today's behavior.
    assert _heuristic_classify("ok") == CONVERSATIONAL
    assert classify_message_intent("ok") == CONVERSATIONAL


def test_standalone_ok_after_a_non_question_assistant_turn_still_behaves_as_small_talk():
    history = [
        _history_row("user", "thanks for your help"),
        _history_row("assistant", "You're welcome!"),
    ]
    assert _heuristic_classify("ok", history) == CONVERSATIONAL


@pytest.mark.parametrize("reply", ["yes Dr.Babar Ali", "yes Dr Babar Ali", "yes, Dr. Babar Ali is right"])
def test_confirming_a_doctor_name_after_a_did_you_mean_question_reaches_knowledge_seeking(reply):
    # Reproduces a reported bug: the bare-reply override above was capped at 3 words,
    # but confirming a fuzzy-matched doctor's name routinely runs longer than that
    # once the name itself is included ("yes Dr.Babar Ali" is already 4 words:
    # yes/dr/babar/ali). That 3-word cap silently missed this case, sending it to the
    # LLM classifier fallback, which misjudged it as small talk — routing a patient
    # who'd just confirmed a doctor's name to the tool-less conversational reply path
    # instead of the tool-calling agent that could actually check availability.
    history = [
        _history_row("user", "is there any slot available for dr ali baber"),
        _history_row("assistant", "Did you mean Dr. Babar Ali?"),
    ]
    assert _heuristic_classify(reply, history) == KNOWLEDGE_SEEKING
    assert classify_message_intent(reply, history) == KNOWLEDGE_SEEKING


def test_bare_reply_after_a_user_turn_not_assistant_still_behaves_as_small_talk():
    # The override only applies when the LAST turn was the assistant asking
    # something — not when the last stored row happens to be from the user.
    history = [_history_row("user", "ok")]
    assert _heuristic_classify("ok", history) == CONVERSATIONAL


# --- is_symptom_message (item 2: routes chat.py away from medical_kb, which no
# longer exists, toward real department-list context) ------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "I have a fever and body aches",
        "my stomach hurts",
        "I've had a headache for two days",
        "persistent cough and chills",
        "mera sar dard kar raha hai",
        "I feel dizzy and nauseous",
        "there's a rash on my arm",
    ],
)
def test_symptom_messages_detected(message):
    assert is_symptom_message(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "What are your clinic hours?",
        "How do I book an appointment?",
        "Where is the clinic located?",
        "What's the parking situation like?",
        "thanks a lot",
        "who's available in cardiology",
    ],
)
def test_non_symptom_messages_not_detected(message):
    assert is_symptom_message(message) is False
