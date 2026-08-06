from types import SimpleNamespace

import pytest

from app.services.chat_markers import BOOKING_MARKER, DEPARTMENT_LIST_MARKER, DOCTOR_OPTIONS_MARKER
from app.services.message_classifier import (
    CONVERSATIONAL,
    KNOWLEDGE_SEEKING,
    PERSONAL_RECALL,
    _heuristic_classify,
    classify_message_intent,
    is_department_list_request,
    is_department_scope_question,
    is_symptom_message,
    needs_booking_action_tools,
    needs_path2_screening,
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


@pytest.mark.parametrize(
    "message",
    [
        "what are the things i told u",
        "What are the things I told you?",
        "what all did i tell you",
        "what did we discuss before",
        "what have we talked about",
        "what did i share with you earlier",
    ],
)
def test_broader_recall_phrasings_classified_personal_recall(message):
    # Reported gap: "what are the things I told you" (a natural way to ask for a
    # broader recap, not just "what's my name") wasn't recognized at all, so it
    # fell through to a generic knowledge-seeking question instead.
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


@pytest.mark.parametrize(
    "message",
    [
        "what's my name",
        "what are the things i told you",
        "What did we discuss before?",
        "what have i described to u?",
        "what did i describe to you",
        "what all did i describe",
        "what i have discussed with u uptil now",
        "what have i discussed with you",
        "what info did i tell u",
        "what info did i told u",
        "what all have we discussed up till now",
    ],
)
def test_personal_recall_phrasing_classified_as_personal_recall(message):
    # Reported live: "what have i described to u?" fell through the recall
    # regex entirely (only told/said/mention/share were covered, not
    # "describe(d)") and was answered as if memory were empty, even though the
    # patient had genuinely described real symptoms earlier. Same gap recurred
    # for "what i have discussed" (subject-before-auxiliary word order) and
    # "what info did i tell u" (a filler word breaking the old fixed "what did i"
    # phrase) — the regex was rebuilt as a flexible pattern instead of adding
    # another one-off phrase each time a new variant gets reported.
    assert _heuristic_classify(message) == PERSONAL_RECALL


@pytest.mark.parametrize(
    "message",
    [
        "what should i tell my doctor about this pain",
        "what medicine did i take yesterday",
    ],
)
def test_prospective_or_unrelated_did_i_phrasing_not_classified_as_personal_recall(message):
    # The flexible "what ... did/have ... i/we ... <recall verb>" pattern must not
    # over-match: "what should I tell" has no did/have auxiliary at all (safe), and
    # "what did I take" uses "did I" but "take" isn't a recall verb, so it must stay
    # a real clinical question, not personal recall.
    assert _heuristic_classify(message) != PERSONAL_RECALL


@pytest.mark.parametrize("message", ["what are your clinic hours?", "I have a headache", "hi"])
def test_non_recall_messages_not_classified_as_personal_recall(message):
    assert _heuristic_classify(message) != PERSONAL_RECALL


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
        "I feel dizzy and nauseous",
        # Reported live: this common typo (missing the second "z") matched no
        # keyword at all and fell through to GENERAL_INFO instead of the symptom
        # triage flow.
        "I am feeling diziness",
        "there's a rash on my arm",
        "i have brain tumor",
        "i think i have cancer",
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


# --- needs_booking_action_tools (conditional binding of book/reschedule/cancel) -----


@pytest.mark.parametrize(
    "message",
    [
        "I want to book an appointment",
        "can I reschedule my appointment",
        "please cancel my appointment",
        "I need to cancel my booking",
        "can we postpone my visit",
        "I can't make it anymore",
        "I won't be able to make it tomorrow",
        "never mind, I don't need it",
        "yes please confirm that slot",
        "is there any cardiologist available on fri",
        "what's the availability like this week",
    ],
)
def test_needs_booking_action_tools_true_for_explicit_booking_language(message):
    assert needs_booking_action_tools(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I have a headache and mild fever",
        "what are your clinic hours?",
        "where is the clinic located?",
        "who is Dr. Iqra Raza",
    ],
)
def test_needs_booking_action_tools_false_for_plain_symptom_or_info_messages_with_no_prior_context(message):
    assert needs_booking_action_tools(message) is False


def test_needs_booking_action_tools_true_when_replying_to_a_clarifying_question():
    # Could be a slot pick or a booking confirmation — never confidently ruled out.
    history = [_history_row("assistant", "Is the pain severe, bearable, or mild?")]
    assert needs_booking_action_tools("mild", history) is True


def test_needs_booking_action_tools_true_when_availability_was_already_shown():
    history = [
        _history_row("user", "who's free in cardiology"),
        _history_row("assistant", DEPARTMENT_LIST_MARKER + '{"departments": []}'),
    ]
    assert needs_booking_action_tools("the 6pm one please", history) is True


def test_needs_booking_action_tools_true_when_a_booking_was_already_confirmed_this_session():
    history = [_history_row("assistant", BOOKING_MARKER + '{"doctor": "Dr. X"}')]
    assert needs_booking_action_tools("actually can we change the time", history) is True


def test_needs_booking_action_tools_false_for_fresh_symptom_turn_with_no_booking_context_yet():
    history = [_history_row("user", "hi")]
    assert needs_booking_action_tools("my hand got broken", history) is False


# --- needs_path2_screening (token-reduction: conditionally droppable PATH 2 body) ---


@pytest.mark.parametrize(
    "message",
    [
        "I have chest pain",
        "chest tightness for an hour",
        "bad chest pressure since this morning",
        "severe head pain",
        "I've had a severe headache since yesterday",
        "I think it's a broken bone in my finger",
        "suspected fracture in my wrist",
        "abdominal pain since last night",
        "my stomach pain won't go away",
        "high fever for two days",
        "persistent fever, not going down",
        "I feel dizzy and lightheaded",
        "I almost fainted earlier",
        "severe back pain",
        "persistent vomiting since this morning",
        "I have diarrhea that won't stop",
        "a deep cut on my arm",
        "sudden vision changes",
        "I got a moderate burn on my hand",
        "a dog bite on my leg",
        "racing heartbeat and palpitations",
        "irregular heart, feels off",
        "I'm pregnant and having some pain",
    ],
)
def test_needs_path2_screening_true_for_path2_named_symptoms(message):
    assert needs_path2_screening(message) is True


@pytest.mark.parametrize(
    "message",
    ["I have a cough", "mild sore throat", "I feel a bit tired", "there's a rash on my arm"],
)
def test_needs_path2_screening_false_for_clean_routine_symptoms_with_no_history(message):
    assert needs_path2_screening(message) is False


def test_needs_path2_screening_true_when_replying_to_a_screening_question():
    # "very severe" alone looks like a routine reply with no PATH-2 keyword at all —
    # the continuity check must still catch it, since it's answering a screening
    # question already in progress.
    history = [_history_row("assistant", "Is the pain severe, bearable, or mild?")]
    assert needs_path2_screening("very severe", history) is True


def test_needs_path2_screening_true_for_major_weight_bearing_bone_fracture():
    # This is the message that also needs PATH 1 — but the EXCEPTION text that
    # tells the model to route straight to PATH 1 for this case physically lives
    # inside PATH 2's own body, so PATH 2 must stay included here too.
    assert needs_path2_screening("my leg is broken") is True
    assert needs_path2_screening("I broke my hip") is True
    assert needs_path2_screening("my spine might be fractured") is True


@pytest.mark.parametrize(
    "message",
    [
        # Reported live: "i am having pain in stomach" routed straight to General
        # Medicine with slots shown, skipping PATH 2 screening entirely —
        # _PATH2_SYMPTOM_PHRASES only matched the fixed order "stomach pain".
        "i am having pain in stomach",
        "i have pain in my abdomen",
        "there is pain in my chest",
        "pain in the back since yesterday",
        "pain in my ear",
    ],
)
def test_needs_path2_screening_true_for_pain_and_body_part_in_any_word_order(message):
    assert needs_path2_screening(message) is True


def test_needs_path2_screening_false_for_a_minor_non_major_bone_mention():
    # A finger/wrist/toe fracture is still on PATH 2's own named list (covered by
    # the "fracture"/"fractured" keyword), so this should stay True, not False —
    # only a genuinely unrelated routine symptom drops PATH 2.
    assert needs_path2_screening("I think I fractured my finger") is True


def test_needs_path2_screening_defaults_true_for_genuine_ambiguity():
    # An unfamiliar/ambiguous symptom phrasing that isn't confidently routine and
    # isn't confidently on PATH 2's list either — the safe default keeps PATH 2.
    # (Not directly testable as "ambiguous" in isolation since the function is
    # binary, but this documents the intended default: anything not clearly
    # matching the routine-drop criteria stays True via the OR-based implementation.)
    history = [_history_row("assistant", "Can you tell me more about when this started?")]
    assert needs_path2_screening("it's been going on for a while", history) is True


# --- is_department_list_request -----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "what are the available depts",
        "what are the available departments",
        "show me available depts",
        "show departments",
        "what departments do you have",
        "which departments are there",
        "list of departments",
        "list departments",
        "all departments",
        "every department",  # singular after all/every is still an unambiguous full-list request
        "available departments",
    ],
)
def test_is_department_list_request_true_for_list_phrasings(message):
    # Reported live: "what are the available depts" wasn't recognized at all — it
    # contains "available", a booking-action keyword, so it used to be misrouted to
    # appointment_agent, which has no tool that can list every department.
    assert is_department_list_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what department is Dr. Smith in",
        "is there a cardiology department",
        "what's available for Dr. Ahmed on Friday",
        "I have a headache",
    ],
)
def test_is_department_list_request_false_for_singular_or_unrelated_messages(message):
    # Singular "department" (asking about ONE specific doctor/department) must not
    # be treated as a request for the full list — plural is required by design.
    assert is_department_list_request(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "so what symptoms does dermatologist treats?",
        "what does cardiology treat?",
        "what symptoms does a neurologist handle?",
        "what conditions does the ENT department deal with?",
        "what does a dermatologist look at?",
        "what does psychiatry specialize in?",
    ],
)
def test_is_department_scope_question_true_for_scope_phrasings(message):
    # Reported live: this kind of general "what does this specialty treat" question
    # landed on symptom_agent (via screening continuity) and got its generic
    # "I'm not able to diagnose... tell me more about your symptom" dead end instead
    # of a real answer — it's an informational question, not a symptom description.
    assert is_department_scope_question(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I have a headache",
        "book me with Dr. Ahmed",
        "what are your clinic hours?",
        "i think dermatologist can be a best fit for it",
    ],
)
def test_is_department_scope_question_false_for_unrelated_messages(message):
    assert is_department_scope_question(message) is False
