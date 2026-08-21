import json
import uuid
from types import SimpleNamespace

import pytest

from app.core.tenancy import ClinicContext
from app.models.clinic import Clinic
from app.models.department import Department
from app.models.doctor import Doctor
from app.models.user import User
from app.services import llm
from app.services.chat_markers import (
    BOOKING_MARKER,
    DEPARTMENT_LIST_MARKER,
    DOCTOR_DISAMBIGUATION_MARKER,
    DOCTOR_OPTIONS_MARKER,
)
from app.services.orchestrator.agents import appointment_agent, general_info_agent, symptom_agent
from app.services.orchestrator.router import (
    APPOINTMENT,
    GENERAL_INFO,
    SYMPTOM_GENERAL,
    _heuristic_classify,
    _symptom_context_more_recent_than_booking_context,
    classify_agent_intent,
)


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def department(db, clinic):
    dept = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(dept)
    db.flush()
    return dept


@pytest.fixture
def doctor(db, clinic, department):
    doc = Doctor(
        clinic_id=clinic.id,
        department_id=department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ahmed Khan",
        is_active=True,
    )
    db.add(doc)
    db.flush()
    return doc


@pytest.fixture
def other_department(db, clinic):
    dept = Department(clinic_id=clinic.id, name="Neurology")
    db.add(dept)
    db.flush()
    return dept


@pytest.fixture
def other_doctor(db, clinic, other_department):
    doc = Doctor(
        clinic_id=clinic.id,
        department_id=other_department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ahmed Raza",
        is_active=True,
    )
    db.add(doc)
    db.flush()
    return doc


@pytest.fixture
def patient(db, clinic):
    p = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(p)
    db.flush()
    return p


@pytest.fixture
def ctx(clinic, patient):
    return ClinicContext(clinic_id=clinic.id, user_id=patient.id, role="patient")


def _row(role, content):
    return SimpleNamespace(role=role, content=content)


class _FakeToolCallResponse:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = tool_calls


class _FakeFinalResponse:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class _FakeLLM:
    def __init__(self, response):
        self._response = response

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._response


# =====================================================================================
# router.classify_agent_intent — heuristic fast-path (mirrors classify_message_intent)
# =====================================================================================


@pytest.mark.parametrize(
    "message", ["I have chest pain", "my leg is broken", "I have a headache and mild fever"]
)
def test_router_classifies_symptom_messages(message):
    assert _heuristic_classify(message) == SYMPTOM_GENERAL


def test_router_classifies_diziness_typo_as_symptom():
    # Reported live: "I am feeling diziness" (missing the second "z") matched no
    # symptom keyword at all, so it fell through to GENERAL_INFO's plain freehand
    # LLM reply — which has no PATH 1/2/3 triage logic — instead of symptom_agent's
    # flow, which always asks 1-2 clarifying questions before naming a department.
    assert _heuristic_classify("I am feeling diziness") == SYMPTOM_GENERAL


def test_router_classifies_difficulty_walking_as_symptom():
    # Reported live: "i am having difficulty in walking properly" matched no
    # symptom keyword at all — mobility/gait complaints had zero coverage
    # anywhere, so this fell through every symptom/booking rule to a generic
    # signal-free-short-statement default (GENERAL_INFO), getting a free-text
    # reply with no screening question and no real department routing at all.
    assert _heuristic_classify("i am having difficulty in walking properly") == SYMPTOM_GENERAL


def test_router_classifies_breething_typo_as_symptom():
    # Reported live: "i am having a bit difficulty in breething" (a common
    # typo for "breathing") matched no symptom keyword at all — same shape as
    # the "diziness" typo already covered — so it fell through to
    # GENERAL_INFO, and the follow-up booking request then reached
    # appointment_agent (no symptom awareness) with no recorded symptom to
    # work from, eventually tripping the no-diagnosis guard's generic
    # redirect instead of ever actually triaging the complaint.
    assert _heuristic_classify("i am having a bit difficulty in breething") == SYMPTOM_GENERAL


@pytest.mark.parametrize(
    "message", ["I want to book an appointment", "please cancel my appointment", "can I reschedule my visit"]
)
def test_router_classifies_booking_action_messages(message):
    assert _heuristic_classify(message) == APPOINTMENT


@pytest.mark.parametrize("message", ["what are your clinic hours?", "hi", "thanks a lot"])
def test_router_classifies_plain_info_and_smalltalk_messages(message):
    assert _heuristic_classify(message) == GENERAL_INFO


@pytest.mark.parametrize(
    "message",
    [
        "isn't neurologist a better idea?",
        "what do you think based upon my symptoms i have told you",
        "which department should i go with",
        # Reported live: "i think i can goto neorologist,what do u suggest?" — the
        # informal "u" instead of "you" fell through the regex entirely (only "you"
        # was covered), so this stayed in APPOINTMENT via marker continuity and
        # showed Neurology's availability directly, with no symptom-grounded
        # reasoning at all.
        "what do u suggest?",
        "what do u think?",
        # Reported live: "i think cardiologist can be a best fit for it?" used
        # "best fit" rather than "better", which wasn't covered at all.
        "i think cardiologist can be a best fit for it?",
        "would this be a good fit?",
        # Reported live: "show me available dept according to my describes
        # symptoms" used "according to" rather than "based on", which wasn't
        # covered at all — fell through to a plain screening question with no
        # grounded symptom-to-department reasoning behind it.
        "show me available dept according to my describes symptoms",
    ],
)
def test_router_rule0_5_recommendation_request_overrides_marker_continuity(message):
    # Reported live: after a DOCTOR_OPTIONS_MARKER card was shown, a recommendation
    # question fell to Rule 1 (marker continuity) and went to appointment_agent —
    # which has no symptom awareness at all — instead of symptom_agent's real
    # symptom-to-department reasoning. Rule 0.5 must win even with a card active.
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + '{"doctors": []}')]
    assert _heuristic_classify(message, history) == SYMPTOM_GENERAL


def test_router_rule1_unambiguous_reply_to_slot_pick_card_routes_to_appointment():
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + '{"doctors": []}')]
    assert _heuristic_classify("the 6pm one please", history) == APPOINTMENT


def test_router_rule1_unambiguous_reply_to_disambiguation_card_routes_to_appointment():
    history = [_row("assistant", DOCTOR_DISAMBIGUATION_MARKER + '{"candidates": []}')]
    assert _heuristic_classify("the Cardiology one", history) == APPOINTMENT


@pytest.mark.parametrize(
    "message",
    ["what are the available depts", "what are the available departments", "show me available depts"],
)
def test_router_rule1_5_department_list_request_routes_to_general_info_despite_available_keyword(message):
    # Reported live: "what are the available depts" wasn't recognized at all —
    # "available" is a booking-action keyword, so without this rule it would reach
    # rule 3 (needs_booking_action_tools) and be misrouted to appointment_agent,
    # which has no tool that can list every department.
    assert _heuristic_classify(message) == GENERAL_INFO


def test_router_rule1_5_department_list_request_overrides_marker_continuity():
    # Reported live: "how many total depts are there in this clinic" asked right
    # after a DOCTOR_OPTIONS_MARKER card was shown still matched rule 1 (marker
    # continuity — not a symptom message) and got routed to appointment_agent,
    # which has no department-LIST capability at all and just re-showed the
    # same stale card instead of answering. Rule 1.5 now runs before rule 1 so
    # an explicit, unambiguous new department-list request wins regardless of
    # what card was shown a moment ago.
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + '{"doctors": []}')]
    assert _heuristic_classify("how many total depts are there in this clinic", history) == GENERAL_INFO


def test_router_rule1_6_doctor_count_question_routes_to_appointment():
    # Reported live: "how many cardiologist are there in this clinic??" has no
    # booking-action keyword and isn't a symptom, so it fell through to
    # general_info_agent, which answered from static KB prose instead of the
    # real, current doctor count. appointment_agent has the real per-department
    # doctor lookup this needs.
    assert _heuristic_classify("how many cardiologist are there in this clinic??") == APPOINTMENT
    assert _heuristic_classify("how many doctors do you have in dermatology") == APPOINTMENT


def test_router_rule0_6_department_scope_question_overrides_screening_continuity():
    # Reported live: after a Cardiology recommendation and a clarifying follow-up
    # question from the assistant, "so what symptoms does dermatologist treats?"
    # fell to rule 2 (screening continuity — preceding turn was a question, symptom
    # context still most recent) and stayed with symptom_agent, which produced its
    # generic "I'm not able to diagnose... tell me more about your symptom" dead
    # end instead of answering the actual question asked. Rule 0.6 must win even
    # when the preceding assistant turn looks like a clarifying question.
    history = [
        _row("user", "i am having pain in my chest"),
        _row("assistant", "Is the chest pain severe, bearable, or mild?"),
        _row("user", "its mild and steady and it is from past 1 week"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + '{"department_name": "Cardiology"}'),
        _row("user", "i think dermatologist can be a best fit for it?isnt it?"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + '{"department_name": "Cardiology"}'),
        _row("user", "so what does dermatologist looks to?"),
        _row(
            "assistant",
            "Based on the chest pain you described, Cardiology might be a better fit than "
            "Dermatology. Would you like me to show Cardiology availability instead, or "
            "would you still like to see Dermatology's availability?",
        ),
    ]
    assert _heuristic_classify("so what symptoms does dermatologist treats?", history) == GENERAL_INFO


def test_router_rule0_7_clinic_logistics_question_overrides_screening_continuity():
    # Reported live: "i have headache" -> "how severe, how long?" -> "mild" ->
    # "are you experiencing nausea, visual changes, fever?" -> "where is this
    # clinic located" still got routed to SYMPTOM_GENERAL and refused as off-topic
    # ("I don't have that information... contact the clinic directly"). Unlike the
    # earlier rule 4 fix (which only helps when a booking-context turn intervenes),
    # this transcript has NO booking activity at all — rule 2's screening
    # continuity fires purely because the preceding turn looks like a question
    # (true for ANY screening question) and nothing booking-related outranks the
    # symptom mention (trivially true with zero booking turns). A logistics
    # question must always win regardless of what clarifying question came right
    # before it.
    history = [
        _row("user", "i have headache"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long "
            "you've had it?",
        ),
        _row("user", "mild"),
        _row(
            "assistant",
            "Are you experiencing any nausea, visual changes, or fever along with the headache?",
        ),
    ]
    assert _heuristic_classify("where is this clinic located", history) == GENERAL_INFO


@pytest.mark.parametrize(
    "message",
    [
        "what are your clinic hours",
        "when does the clinic open",
        "what is the address of this clinic",
        "do you accept insurance",
        "what is your cancellation policy",
        "how can i contact the clinic",
    ],
)
def test_router_rule0_7_covers_other_clinic_logistics_topics_not_just_hours(message):
    # The fix generalizes to any clinic-logistics topic, not just hours/location —
    # same mid-screening history as the reported transcript, different question.
    history = [
        _row("user", "i have headache"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "mild"),
        _row("assistant", "Are you experiencing any nausea, visual changes, or fever?"),
    ]
    assert _heuristic_classify(message, history) == GENERAL_INFO


def test_router_rule0_7_does_not_override_a_genuine_symptom_message():
    # Guard against the new rule being too broad: a message that plainly states a
    # NEW symptom must still route to SYMPTOM_GENERAL even if it happens to also
    # mention a logistics word.
    assert _heuristic_classify("my chest hurts and I want to know your clinic hours too") == SYMPTOM_GENERAL


@pytest.mark.parametrize(
    "message",
    [
        "cancel my appointment",
        "i want to reschedule my appointment",
        "cancel my upcoming appointment",
        "can you reschedule it for me",
    ],
)
def test_router_rule0_8_cancel_reschedule_action_overrides_screening_continuity(message):
    # Reported live: "i have headache" -> "how severe, how long?" -> "mild" ->
    # "are you experiencing nausea, visual changes, fever?" -> "cancel my
    # appointment" (or a reschedule equivalent) still got routed to
    # SYMPTOM_GENERAL instead of appointment_agent — same class of bug as rule 0.7
    # (screening continuity has no way to notice the CURRENT message is an
    # explicit booking action), one category over. Rule 3 already exists to catch
    # this via needs_booking_action_tools, but sits AFTER rule 2 and can't simply
    # be moved ahead of it (see rule 0.8's own docstring for why).
    history = [
        _row("user", "i have headache"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long "
            "you've had it?",
        ),
        _row("user", "mild"),
        _row(
            "assistant",
            "Are you experiencing any nausea, visual changes, or fever along with the headache?",
        ),
    ]
    assert _heuristic_classify(message, history) == APPOINTMENT


def test_router_rule0_8_does_not_swallow_a_plain_screening_answer():
    # Guard against the new rule being too broad: a plain reply to a clarifying
    # screening question (no cancel/reschedule keyword at all) must still
    # continue triage with symptom_agent, not jump to appointment_agent. This is
    # exactly the failure mode rule 0.8 avoids by NOT reusing
    # needs_booking_action_tools' own "preceding turn is a question" fallback.
    history = [
        _row("user", "i have headache"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long "
            "you've had it?",
        ),
    ]
    assert _heuristic_classify("mild", history) == SYMPTOM_GENERAL
    assert _heuristic_classify("severe", history) == SYMPTOM_GENERAL


@pytest.mark.parametrize(
    "message",
    [
        "when was my last cancelled appointment",
        "what was my last completed appointment",
        "when was my most recent missed appointment",
        "what is my last past appointment",
    ],
)
def test_router_rule0_9_appointment_status_history_query_overrides_screening_continuity(message):
    # Follow-up report: "when was my last cancelled appointment" happened to
    # already work after rule 0.8 shipped, purely because "cancelled" is also a
    # literal cancel-action word — but "what was my last completed appointment" /
    # "when was my most recent missed appointment" (same category of question, no
    # cancel/reschedule keyword at all) still fell through to rule 2 and got
    # routed to SYMPTOM_GENERAL instead of appointment_agent's real, DB-grounded
    # status-history answer.
    history = [
        _row("user", "i have headache"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long "
            "you've had it?",
        ),
        _row("user", "mild"),
        _row(
            "assistant",
            "Are you experiencing any nausea, visual changes, or fever along with the headache?",
        ),
    ]
    assert _heuristic_classify(message, history) == APPOINTMENT


def test_router_rule0_9_does_not_swallow_a_plain_screening_answer():
    # Guard against the new rule being too broad: a plain reply to a clarifying
    # screening question must still continue triage.
    history = [
        _row("user", "i have headache"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long "
            "you've had it?",
        ),
    ]
    assert _heuristic_classify("mild", history) == SYMPTOM_GENERAL


def test_router_rule1_8_a_new_symptom_statement_overrides_stale_booking_continuity():
    # Reported live: General Medicine was resolved for dizziness/fainting, then
    # small talk ("okay thank you" / "one more question"), then the assistant's
    # own generic filler reply ("Sure, what would you like to know? ... doctors
    # ...") — its trailing "?" plus mentioning "doctors" tripped rule 2/3's
    # booking-continuity heuristics (the DOCTOR_OPTIONS_MARKER card was more
    # recent than the last symptom-shaped user turn), sending "i am also having
    # pain in my chest" — an unambiguous NEW symptom statement — to
    # appointment_agent, which has no screening logic at all, skipping triage
    # entirely.
    history = [
        _row("user", "i am feeling diziness today"),
        _row("assistant", "How would you describe the dizziness — mild, occasional, or more severe?"),
        _row("user", "its occasional and mild and i am also feeling faint"),
        _row("assistant", "How long have you been feeling faint, and does it happen when you stand up quickly?"),
        _row("user", "yes it happens when i stands up quickly, and been happening from 2 days"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + '{"department_name": "General Medicine"}'),
        _row("user", "okay thank you"),
        _row(
            "assistant",
            "You're welcome! If you'd like to book an appointment with one of the listed doctors, just let "
            "me know which time works best for you. If you have any other questions, feel free to ask.",
        ),
        _row("user", "yes i have one more question"),
        _row(
            "assistant",
            "Sure, what would you like to know? Feel free to ask about appointments, doctors, or anything "
            "else you need help with.",
        ),
    ]
    assert _heuristic_classify("i am also having pain in my chest", history) == SYMPTOM_GENERAL


def test_router_rule1_8_new_symptom_after_a_confirmed_booking_still_overrides_continuity():
    # Reported live: a Cardiology appointment was booked and confirmed
    # (BOOKING_MARKER), then "i am also having skin related issues as well" went
    # straight to a Dermatology card with zero screening. Root cause was two
    # layers deep: bare "skin" wasn't in message_classifier._SYMPTOM_KEYWORDS at
    # all (even though symptom_hints.py already treats it as real symptom
    # vocabulary), so is_symptom_message returned False and Rule 1.8 above never
    # got the chance to fire, falling through to booking-continuity instead.
    history = [
        _row("user", "i am having pain in my chest"),
        _row("assistant", "Is the chest pain severe, bearable, or mild?"),
        _row("user", "its mild and started today morning with no additional symptoms"),
        _row("assistant", "Does the pain stay only in your chest, or does it radiate to your arm, jaw, or back?"),
        _row("user", "it only in chest"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + '{"department_name": "Cardiology"}'),
        _row("user", "Book with Dr. Farhan Malik at Sat, Aug 8 at 11:00 AM"),
        _row("assistant", BOOKING_MARKER + '{"doctor_name": "Dr. Farhan Malik"}'),
    ]
    assert _heuristic_classify("i am also having skin related issues as well", history) == SYMPTOM_GENERAL


def test_router_rule1_new_symptom_right_after_a_doctor_options_card_overrides_marker_continuity():
    # Reported live: a Cardiology DOCTOR_OPTIONS_MARKER card was shown (doctor
    # slots displayed, nothing booked yet), then "i am also having some skin
    # related issues" — an unprompted, genuinely NEW symptom, unrelated to any
    # slot pick — matched Rule 1's old unconditional "reply to a slot-pick card
    # is always APPOINTMENT" and went straight to appointment_agent, which has no
    # symptom-triage logic and just showed Dermatology availability with ZERO
    # clarifying questions. Distinct from the confirmed-booking test above (there
    # the preceding marker is BOOKING_MARKER, which Rule 1 was never scoped to
    # match at all) — here the preceding marker IS exactly what Rule 1 matches,
    # so this is the case that actually needed Rule 1 itself to change, not just
    # the keyword list Rule 1.8 depends on.
    history = [
        _row("user", "i am having pain in my chest"),
        _row(
            "assistant",
            "Is the chest pain severe, bearable, or mild? Also, did it start suddenly or is it coming "
            "and going, and are you feeling any sweating, shortness of breath, or nausea?",
        ),
        _row("user", "its mild and started today morning with no other symptoms"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + '{"department_name": "Cardiology"}'),
    ]
    assert _heuristic_classify("i am also having some skin related issues", history) == SYMPTOM_GENERAL

    # Sanity: a genuine slot-pick/doctor-name reply in the same position must
    # still go to appointment — confirms this isn't a blanket bypass of Rule 1.
    assert _heuristic_classify("the 9am one please", history) == APPOINTMENT
    assert _heuristic_classify("Dr. Farooq please", history) == APPOINTMENT


def test_router_rule2_continuity_reply_to_screening_question_routes_back_to_symptom():
    # "very severe" has no symptom keyword of its own — only correct because the
    # conversation is still symptom-side (item 5 continuity).
    history = [_row("user", "I have chest pain"), _row("assistant", "Is the pain severe, bearable, or mild?")]
    assert _heuristic_classify("very severe", history) == SYMPTOM_GENERAL


def test_router_recency_fix_old_symptom_mention_does_not_outrank_active_booking_flow():
    # The exact gap flagged during design review: a symptom mentioned once several
    # turns ago must not outrank a conversation that has since clearly moved into
    # booking territory — a disambiguation reply here must go to appointment, not
    # back to symptom_agent just because "chest pain" appears earlier in history.
    history = [
        _row("user", "I have chest pain"),
        _row("assistant", "Is the pain severe, bearable, or mild?"),
        _row("user", "mild, it's fine"),
        _row("assistant", DEPARTMENT_LIST_MARKER + '{"departments": []}'),
        _row("user", "book me with Dr Ahmed"),
        _row("assistant", DOCTOR_DISAMBIGUATION_MARKER + '{"candidates": []}'),
    ]
    assert _heuristic_classify("the Cardiology one", history) == APPOINTMENT


def test_symptom_context_more_recent_than_booking_context_true_with_no_booking_signal():
    history = [_row("user", "I have chest pain")]
    assert _symptom_context_more_recent_than_booking_context(history) is True


def test_symptom_context_more_recent_than_booking_context_false_once_booking_marker_appears():
    history = [
        _row("user", "I have chest pain"),
        _row("assistant", DEPARTMENT_LIST_MARKER + '{"departments": []}'),
    ]
    assert _symptom_context_more_recent_than_booking_context(history) is False


def test_router_falls_back_to_llm_only_on_genuine_ambiguity(monkeypatch):
    # A longer message with no recognizable heuristic signal at all and no
    # heuristic verdict from the classic classifier either — confirms the free
    # fast-path genuinely returns None here (forcing the LLM fallback), and that the
    # fallback is wired through classify_agent_intent. Deliberately longer than
    # _MAX_NO_SIGNAL_WORDS (rule 6's own short-statement default) so it isn't
    # resolved for free before ever reaching the LLM fallback.
    called = {}
    message = "I have been meaning to ask you something but I keep forgetting exactly what it was"

    def fake_llm_classify(msg):
        called["message"] = msg
        return APPOINTMENT

    monkeypatch.setattr("app.services.orchestrator.router._llm_classify", fake_llm_classify)
    result = classify_agent_intent(message)

    assert called["message"] == message
    assert result == APPOINTMENT


def test_router_llm_classify_defaults_to_symptom_general_on_failure(monkeypatch):
    from app.services.orchestrator import router

    def raise_error(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(router, "ChatGroq", raise_error)
    assert router._llm_classify("some ambiguous message") == SYMPTOM_GENERAL


# =====================================================================================
# router rule 6 — signal-free short statement defaults to GENERAL_INFO for free,
# instead of paying for (and risking) the LLM fallback's SYMPTOM_GENERAL-biased guess
# =====================================================================================


@pytest.mark.parametrize(
    "message",
    [
        "my name is daud",
        "my name is daud please remember it",
        "I'm daud, nice to meet you",
        "just saying hello to you",
    ],
)
def test_router_resolves_signal_free_short_statements_to_general_info_for_free(monkeypatch, message):
    # Reported live: "my name is daud" (4 words, no symptom/booking keyword) fell
    # through every free rule and reached the LLM fallback, which is deliberately
    # biased toward SYMPTOM_GENERAL "when in doubt" — attaching the entire triage
    # prompt to a message with nothing to do with symptoms. The LLM fallback must
    # never even be called for these.
    def _fail_if_called(message):
        raise AssertionError("the LLM fallback must not be called for a signal-free short statement")

    monkeypatch.setattr("app.services.orchestrator.router._llm_classify", _fail_if_called)

    assert classify_agent_intent(message) == GENERAL_INFO


def test_router_rule6_does_not_swallow_a_real_symptom_message():
    # Guard against the new rule being too broad: a genuine symptom description
    # must still route to SYMPTOM_GENERAL, never fall into the signal-free default.
    assert classify_agent_intent("my chest hurts a lot right now") == SYMPTOM_GENERAL


def test_router_rule6_does_not_swallow_a_real_booking_message():
    assert classify_agent_intent("cancel my appointment") == APPOINTMENT


def test_router_rule6_has_a_word_count_ceiling_not_unlimited():
    # Long enough that it's no longer a "short statement" — must still be capable
    # of reaching the LLM fallback rather than silently defaulting forever.
    from app.services.orchestrator.router import _heuristic_classify

    long_message = "I have been meaning to ask you something but I keep forgetting exactly what it was"
    assert _heuristic_classify(long_message, []) is None


def test_router_rule4_symptom_signal_does_not_outrank_more_recent_booking_context():
    # Reported live, full transcript: "i am having pain in chest" (symptom
    # screening starts) -> "cancel my upcoming appointment" / "reschedule my
    # upcoming appointment" (clearly booking-related, correctly handled by
    # appointment_agent) -> "whats the clinic hours?" — a plain, self-contained,
    # keyword-clean informational question with nothing to do with symptoms or
    # booking. Old rule 4 had no recency bound on its history scan at all: once
    # ANY symptom mention was on record, every later turn not otherwise claimed by
    # an earlier rule fell to SYMPTOM_GENERAL forever — even after the
    # conversation had since clearly moved through several unrelated booking
    # turns — producing symptom_agent's off-topic refusal ("I don't have that
    # information... contact the clinic directly") for a plain clinic-hours
    # question. Must now respect the same recency-vs-booking-context comparison
    # rule 2 already uses for its own (screening-continuity) case.
    history = [
        _row("user", "i am having pain in chest"),
        _row(
            "assistant",
            "Is the chest pain severe, moderate, or mild? Also, does it come on suddenly...",
        ),
        _row("user", "i want to cancel my appointment"),
        _row("assistant", "Could you please provide the appointment ID or the date and time?"),
        _row("user", "cancel my upcoming appointment"),
        _row("assistant", "You don't have an upcoming appointment to cancel."),
        _row("user", "reschedule my upcoming appointment"),
        _row("assistant", "You don't have an upcoming appointment to reschedule."),
    ]
    assert classify_agent_intent("whats the clinic hours?", history) != SYMPTOM_GENERAL


def test_router_rule4_still_recovers_a_stale_symptom_turn_with_no_intervening_booking():
    # Unchanged behavior: with nothing booking-related in between, a symptom
    # mentioned a turn or two ago must still be recoverable by this rule. Tests
    # _rule_4_symptom_signal_anywhere directly (not the full cascade) so this
    # isolates rule 4's own fallback rather than incidentally passing via rule 2's
    # separate screening-continuity check.
    from app.services.orchestrator.router import _rule_4_symptom_signal_anywhere

    history = [
        _row("user", "my chest hurts a lot right now"),
        _row("assistant", "That sounds concerning — let's find you the right doctor."),
    ]
    assert _rule_4_symptom_signal_anywhere("ok, whatever you think is best for this", history, "assistant", history[-1].content) == SYMPTOM_GENERAL


def test_router_rule4_symptom_fallback_requires_recency_over_booking_context_directly():
    # Direct unit check on the rule itself (not the full cascade), isolating
    # exactly the condition that was missing: a symptom turn index that exists but
    # is NOT more recent than a booking-context turn must not fire this rule.
    from app.services.orchestrator.router import _rule_4_symptom_signal_anywhere

    history = [
        _row("user", "my chest hurts a lot right now"),
        _row("assistant", "Is it severe?"),
        _row("user", "cancel my upcoming appointment"),
        _row("assistant", "You don't have an upcoming appointment to cancel."),
    ]
    assert _rule_4_symptom_signal_anywhere("whats the clinic hours?", history, "assistant", history[-1].content) is None


# =====================================================================================
# symptom_agent
# =====================================================================================


def test_symptom_agent_prompt_includes_triage_and_department_context():
    prompt = symptom_agent._build_system_prompt("English", ["Cardiology", "ENT"], include_path2=True)
    assert "SYMPTOM TRIAGE RULE" in prompt
    assert "PATH 1" in prompt
    assert "PATH 2 — AMBIGUOUS" in prompt
    assert "Active departments at this clinic: Cardiology, ENT." in prompt
    assert "Today's date is" in prompt


def test_symptom_agent_prompt_omits_path2_when_told_to():
    prompt = symptom_agent._build_system_prompt("English", ["Cardiology"], include_path2=False)
    assert "PATH 2 — AMBIGUOUS" not in prompt
    assert "PATH 1 — CONFIRMED EMERGENCY" in prompt
    assert "PATH 3 — ROUTINE SYMPTOM" in prompt


def test_symptom_agent_prompt_includes_find_doctors_by_name_rule_only():
    prompt = symptom_agent._build_system_prompt("English", [], include_path2=True)
    assert "NEVER treat a patient-typed doctor name as a confirmed match" in prompt
    # Appointment-action-only rules must NOT leak into symptom_agent's prompt.
    assert "get_my_appointments returns structured data" not in prompt
    assert "Only call book_appointment once the patient has clearly picked" not in prompt


def test_symptom_agent_prompt_omits_patient_memory_section_entirely():
    # PATIENT MEMORY was removed from the prompt entirely (not just left empty) —
    # cross-session memory no longer exists, so the section only confused the model
    # into misfiring on unrelated messages (see app.services.chat's module docstring).
    prompt = symptom_agent._build_system_prompt("English", [], include_path2=True)
    assert "PATIENT MEMORY" not in prompt


def test_symptom_agent_prompt_handles_no_active_departments():
    prompt = symptom_agent._build_system_prompt("English", [], include_path2=True)
    assert "Active departments at this clinic: (none configured)." in prompt


def test_run_symptom_agent_only_binds_its_two_tools(monkeypatch, db, ctx):
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["tools"] = {t.name for t in tools}
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(llm, "run_tool_calling_agent", fake_run_tool_calling_agent)
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    # Duration + severity already given together (PATH 3's own stated exception)
    # so this reaches the LLM normally instead of the first-message backstop —
    # this test is about tool binding, not the backstop itself.
    result = symptom_agent.run_symptom_agent(db, ctx, "I have had a mild cough for 3 days", "en", [])

    assert result == "reply"
    assert captured["tools"] == {"get_department_availability", "find_doctors_by_name"}
    assert "SYMPTOM TRIAGE RULE" in captured["system_prompt"]


def test_run_symptom_agent_fetches_a_second_department_named_in_the_note_but_never_queried(
    monkeypatch, db, ctx, clinic, department, doctor, other_department, other_doctor
):
    # Reported live: "ear pain and chest pain" resolved a note saying "this could be
    # evaluated by ENT for the ear pain and Cardiology for the chest pain" but the
    # model only actually called get_department_availability for the first one
    # (Cardiology fixture here stands in for the reported ENT card; Neurology stands
    # in for the reported Cardiology card) — the second, real, active department
    # named in the model's own note must still get fetched and shown, not silently
    # dropped.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    db.add(Slot(
        clinic_id=clinic.id, doctor_id=other_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    card = json.dumps(
        {
            "note": "This could be evaluated by Cardiology for the chest pain and Neurology for the dizziness.",
            "department_name": "Cardiology",
            "doctors": [{"doctor_id": "d1", "doctor_name": doctor.full_name, "specialization": None, "slots": []}],
        }
    )
    reply_with_marker = DOCTOR_OPTIONS_MARKER + card

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: reply_with_marker)

    result = symptom_agent.run_symptom_agent(db, ctx, "chest pain and dizziness", "en", [])

    assert result.startswith(DEPARTMENT_LIST_MARKER)
    payload = json.loads(result[len(DEPARTMENT_LIST_MARKER):])
    names = {d["department_name"] for d in payload["departments"]}
    assert names == {"Cardiology", "Neurology"}
    neuro_entry = next(d for d in payload["departments"] if d["department_name"] == "Neurology")
    assert len(neuro_entry["doctors"]) == 1
    # The extra department must still explain itself — reusing the model's own
    # note text, which already named both departments by reasoning.
    assert neuro_entry["note"] == "This could be evaluated by Cardiology for the chest pain and Neurology for the dizziness."


def test_run_symptom_agent_fetches_a_department_hinted_by_symptom_words_even_when_the_note_never_names_it(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live (2nd instance of this gap): "ear pain and itchy skin" then "chest
    # pain mild, itchy skin from past 1 day" got a reply that routed to Cardiology
    # and never mentioned Dermatology in its note at all this time — so the
    # note-scan fix above has nothing to find. This is the keyword-hint backstop:
    # the patient's own words ("itchy", "skin") plus a real active department whose
    # name matches the "derma" hint must still get fetched.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    derma_department = Department(clinic_id=clinic.id, name="Dermatology")
    db.add(derma_department)
    db.flush()
    derma_doctor = Doctor(
        clinic_id=clinic.id,
        department_id=derma_department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Sara Malik",
        is_active=True,
    )
    db.add(derma_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=derma_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    card = json.dumps(
        {
            "note": "Mild chest pain that is bearable and not worsening, likely Cardiology evaluation.",
            "department_name": "Cardiology",
            "doctors": [{"doctor_id": "d1", "doctor_name": doctor.full_name, "specialization": None, "slots": []}],
        }
    )
    reply_with_marker = DOCTOR_OPTIONS_MARKER + card
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: reply_with_marker)

    history = [
        _row("user", "i am having pain in my ear and also my skin in itchy"),
        _row("assistant", "Is the ear pain mild, moderate, or severe, and how long have you been experiencing the itchy skin?"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "the pain in chest is mild and bearable and facing itchy skin problem from past 1 day", "en", history
    )

    assert result.startswith(DEPARTMENT_LIST_MARKER)
    payload = json.loads(result[len(DEPARTMENT_LIST_MARKER):])
    names = {d["department_name"] for d in payload["departments"]}
    assert names == {"Cardiology", "Dermatology"}
    derma_entry = next(d for d in payload["departments"] if d["department_name"] == "Dermatology")
    assert len(derma_entry["doctors"]) == 1
    # The model's note never named Dermatology at all, so a synthesized reasoning
    # sentence must be shown instead of no explanation at all — and it must name
    # the specific symptom ("skin symptoms"), not a generic "this could also be
    # evaluated by X" with no reasoning attached.
    assert derma_entry["note"] == "Based on the skin symptoms described, this could also be evaluated by Dermatology."


def test_run_symptom_agent_fetches_head_and_leg_pain_departments_hinted_by_symptom_words(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "pain in my teeth, head and legs as well" only ever produced a
    # single Dentistry card — neither "head"/"headache" nor "leg"/"legs" existed as
    # a keyword in SYMPTOM_DEPARTMENT_HINTS at all, so this exact keyword-hint
    # backstop (already proven for skin/dermatology above) had nothing to find for
    # either one, even though the model's note named neither department either.
    # Follow-up report: once "leg" unconditionally hinted Orthopedics, a patient who
    # explicitly denied any leg symptoms still got an unwanted Orthopedics card —
    # bare limb pain with no injury signal now routes to General Medicine instead
    # (see symptom_hints.py's "limb/joint pain" entry), so Orthopedics is correctly
    # absent here.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    dentistry = Department(clinic_id=clinic.id, name="Dentistry")
    db.add(dentistry)
    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(gen_med)
    db.flush()
    dentist = Doctor(
        clinic_id=clinic.id, department_id=dentistry.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Qureshi", is_active=True,
    )
    db.add(dentist)
    gen_med_doctor = Doctor(
        clinic_id=clinic.id, department_id=gen_med.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza", is_active=True,
    )
    db.add(gen_med_doctor)
    db.flush()
    for doc in (dentist, gen_med_doctor):
        db.add(Slot(
            clinic_id=clinic.id, doctor_id=doc.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ))
    db.flush()

    card = json.dumps(
        {
            "note": "The mild tooth pain can be evaluated by Dentistry.",
            "department_name": "Dentistry",
            "doctors": [{"doctor_id": "d1", "doctor_name": dentist.full_name, "specialization": None, "slots": []}],
        }
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: DOCTOR_OPTIONS_MARKER + card)

    history = [
        _row("user", "i am having pain in my teeth,head and legs as well"),
        _row("assistant", "Is the pain severe, bearable, or mild for each area?"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx,
        "teeth pain started today morning and is mild\nhead pain is from 2 days and bearable\n"
        "no there is no such symptoms on legs",
        "en", history,
    )

    assert result.startswith(DEPARTMENT_LIST_MARKER)
    payload = json.loads(result[len(DEPARTMENT_LIST_MARKER):])
    names = {d["department_name"] for d in payload["departments"]}
    assert names == {"Dentistry", "General Medicine"}


def test_run_symptom_agent_names_the_specific_symptom_in_the_hinted_departments_note(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "pain in my head and in my chest as well" routed to General
    # Medicine (with a symptom-naming note the model composed itself) plus
    # Cardiology (hinted via the "chest" keyword) — but Cardiology's synthesized
    # note read as generic boilerplate ("this could also be evaluated by
    # Cardiology") with no mention of chest pain, an inconsistent, half-explained
    # reply next to General Medicine's real reasoning. The synthesized note must
    # name the specific symptom that drove this department, same as this table's
    # other categories.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    gen_med_department = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(gen_med_department)
    cardio_department = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(cardio_department)
    db.flush()
    gen_med_doctor = Doctor(
        clinic_id=clinic.id,
        department_id=gen_med_department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza",
        is_active=True,
    )
    db.add(gen_med_doctor)
    cardio_doctor = Doctor(
        clinic_id=clinic.id,
        department_id=cardio_department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ahmed Farooq",
        is_active=True,
    )
    db.add(cardio_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=cardio_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    card = json.dumps(
        {
            "note": "Based on what you've described, this sounds like something General Medicine should look at.",
            "department_name": gen_med_department.name,
            "doctors": [{"doctor_id": "d1", "doctor_name": gen_med_doctor.full_name, "specialization": None, "slots": []}],
        }
    )
    reply_with_marker = DOCTOR_OPTIONS_MARKER + card
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: reply_with_marker)

    history = [
        _row("user", "i am having pain in my head and in my chest as well"),
        _row(
            "assistant",
            "Is the head pain and chest discomfort severe, bearable, or mild? Also, did the chest feeling come on "
            "suddenly or worsen over time, and is the head pain constant or does it come and go?",
        ),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "the pain is mild and bearable and started today morning", "en", history
    )

    assert result.startswith(DEPARTMENT_LIST_MARKER)
    payload = json.loads(result[len(DEPARTMENT_LIST_MARKER):])
    cardio_entry = next(d for d in payload["departments"] if d["department_name"] == "Cardiology")
    assert cardio_entry["note"] == "Based on the chest pain described, this could also be evaluated by Cardiology."


def test_run_symptom_agent_does_not_add_a_spurious_general_medicine_card_for_head_as_a_location(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "i am feeling numb" -> "where?" -> "in the head" -> "also
    # having weakness" got a correct Neurology card from the LLM, PLUS a spurious
    # extra "Based on the head pain described, this could also be evaluated by
    # General Medicine" card — but the patient never described a headache at all;
    # "head" was their answer to WHERE the numbness was located, not a pain
    # complaint. Confirms the cross-check no longer misreads a bare body-location
    # word as its own separate "head pain" symptom.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    neuro = Department(clinic_id=clinic.id, name="Neurology")
    db.add(neuro)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=neuro.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Fatima Raza", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    card = json.dumps(
        {
            "note": "Based on your mild head numbness and weakness, Neurology would be appropriate for follow-up.",
            "department_name": "Neurology",
            "doctors": [{"doctor_id": "d1", "doctor_name": doctor.full_name, "specialization": None, "slots": []}],
        }
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: DOCTOR_OPTIONS_MARKER + card)

    history = [
        _row("user", "i am feeling numb right now"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and started 1 hour ago"),
        _row("assistant", "Where exactly are you feeling the numbness?"),
        _row("user", "in the head"),
        _row("assistant", "Are you also experiencing any headache, vision changes, dizziness, or weakness?"),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "having weakness", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Neurology"
    assert "General Medicine" not in result


def test_run_symptom_agent_does_not_falsely_hint_dentistry_for_ear_and_neck_pain(monkeypatch, db, ctx, clinic):
    # Reported live: "pain in my ear and neck" + "difficulty swallowing" correctly
    # routed to ENT, but a second, unrelated Dentistry card also showed up with no
    # dental symptom mentioned anywhere. Root cause: the "ear" keyword's hint
    # substring "ent" was matched with a plain `in` substring check against
    # department names, and "ent" happens to occur mid-word inside "Dentistry"
    # (d-ENT-istry). This confirms only ENT is returned, never Dentistry.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    ent_department = Department(clinic_id=clinic.id, name="ENT")
    db.add(ent_department)
    dentistry_department = Department(clinic_id=clinic.id, name="Dentistry")
    db.add(dentistry_department)
    db.flush()
    ent_doctor = Doctor(
        clinic_id=clinic.id,
        department_id=ent_department.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Raza",
        is_active=True,
    )
    db.add(ent_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=ent_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    card = json.dumps(
        {
            "note": "Based on your moderate ear and neck pain with difficulty swallowing, an ENT evaluation is appropriate.",
            "department_name": "ENT",
            "doctors": [{"doctor_id": "d1", "doctor_name": ent_doctor.full_name, "specialization": None, "slots": []}],
        }
    )
    reply_with_marker = DOCTOR_OPTIONS_MARKER + card
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: reply_with_marker)

    history = [
        _row("user", "i am having pain in my ear and neck since 2 days"),
        _row("assistant", "How would you describe the pain, and do you have any swelling, fever, or difficulty swallowing?"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "the pain is moderate and bearable and i am having difficulty swallowing", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "ENT"


def test_run_symptom_agent_answers_a_recommendation_request_deterministically_without_the_llm(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live: "isn't neurologist a better idea?" / "what do you think based on
    # my symptoms" got either a blind department switch (via appointment_agent, no
    # reasoning) or a free-text reply that HALLUCINATED a symptom ("ear pain") never
    # actually mentioned. A recommendation request must be answered entirely from the
    # real symptom-to-department mapping over what was actually said, never handed to
    # the LLM to freehand a reason.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(gen_med)
    db.flush()
    gen_med_doctor = Doctor(
        clinic_id=clinic.id,
        department_id=gen_med.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza",
        is_active=True,
    )
    db.add(gen_med_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=gen_med_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a recommendation request")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "I am having a bit of fever and body aches"),
        _row("assistant", "How long have you had the fever and body aches?"),
        _row("user", "from past 2 years, severity is mild"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "but i feel like going to neurologist isnt that a better idea?", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "General Medicine"
    assert "fever" in payload["note"].lower()


def test_run_symptom_agent_recommendation_request_names_low_mood_and_skin_departments_not_a_guessed_one(
    monkeypatch, db, ctx, clinic
):
    # Reported live: a patient described sadness (2 days) and mild itchy skin, was
    # shown a combined Dermatology/Psychiatry card, then asked to book with
    # Neurology instead ("i think i can goto neorologist,what do u suggest?") — a
    # department their own symptoms never pointed to at all. The informal "what do
    # u suggest" (not "you") is what let this fall through to a plain department
    # lookup instead of this deterministic short-circuit — see the message_classifier
    # fix alongside this test. "sad"/"sadness" are also now in the low-mood keyword
    # set (previously only "anxiety"/"depression"/"mental"/"stress" were covered).
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    for name in ("Dermatology", "Psychiatry", "Neurology"):
        dept = Department(clinic_id=clinic.id, name=name)
        db.add(dept)
        db.flush()
        doctor = Doctor(
            clinic_id=clinic.id,
            department_id=dept.id,
            external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
            full_name=f"Dr. {name}",
            is_active=True,
        )
        db.add(doctor)
        db.flush()
        db.add(Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a recommendation request")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i am feeling very sad today also i have some skin related issues as well"),
        _row("assistant", "How long have you been feeling sad? Can you describe the skin issue?"),
        _row("user", "no thoughts of harming myself, sadness is from 2 days. skin is itchy, not very severe"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "i think i can goto neorologist,what do u suggest?", "en", history
    )

    assert "Neurology" not in result
    assert "Dermatology" in result
    assert "Psychiatry" in result


def test_run_symptom_agent_recommendation_request_names_dentistry_for_jaw_pain_not_a_guessed_cardiology(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "i am having pain in my jaw" got a correct Dentistry card from
    # the LLM's own first-turn reasoning, but then "i think cardiologist can be
    # best fit for it?" got a Cardiology card instead — "jaw" was never a keyword
    # in symptom_hints.py at all, so this recommendation-request shortcut's scan of
    # prior history came back empty, fell through to the LLM with nothing to push
    # back with, and the model just went along with the patient's own guess. "jaw"
    # is now in the dental keyword set, same "missing keyword, not missing
    # mechanism" shape as the "teeths" typo fix above.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    for name in ("Dentistry", "Cardiology"):
        dept = Department(clinic_id=clinic.id, name=name)
        db.add(dept)
        db.flush()
        doctor = Doctor(
            clinic_id=clinic.id,
            department_id=dept.id,
            external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
            full_name=f"Dr. {name}",
            is_active=True,
        )
        db.add(doctor)
        db.flush()
        db.add(Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a recommendation request")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i am having pain in my jaw"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and moderate and started today morning"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "i think cardiologist can be best fit for it?", "en", history
    )

    assert "Cardiology" not in result
    assert "Dentistry" in result


def _make_dept_with_slot(db, clinic, name):
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    dept = Department(clinic_id=clinic.id, name=name)
    db.add(dept)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id,
        department_id=dept.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name=f"Dr. {name}",
        is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()


def test_run_symptom_agent_recommendation_request_names_general_medicine_for_bare_lightheadedness_not_a_guessed_cardiology(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "im very light headed" -> "its not that much severe..book me
    # an regular appointment" got a General Medicine card from the LLM's own
    # first-turn reasoning, then "but i think cardiology can be best fit for it"
    # got a real Cardiology availability card instead. Root cause: "light headed"
    # (patient typed it as two words) never matched the "lightheaded" keyword in
    # symptom_hints.py, so this recommendation-request shortcut's scan of prior
    # history came back empty, fell through to the LLM with nothing to push back
    # with, and the model just went along with the patient's own guess.
    # symptom_hints.py now normalizes the space/hyphen variants of "lightheaded"
    # before tokenizing, AND routes bare lightheadedness (no ear/vertigo/spinning
    # signal alongside it) to General Medicine specifically, matching how the
    # LLM's own first-turn reasoning had already (correctly) triaged it — see the
    # companion test below for the lightheaded+vertigo -> ENT case.
    for name in ("General Medicine", "Cardiology"):
        _make_dept_with_slot(db, clinic, name)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a recommendation request")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "im very light headed"),
        _row("assistant", "If you think this could be serious, please seek immediate medical attention..."),
        _row("user", "its not that much severe that i should goto emergency book me an regular appointment"),
        _row("assistant", "General Medicine\nDr. Ali Raza — Internal Medicine\n..."),
        _row("user", "is general medicine best fit for this case?"),
        _row("assistant", "Yes, General Medicine is the appropriate department..."),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "but i think cardiology can be best fit for it", "en", history
    )

    assert "Cardiology" not in result
    assert "General Medicine" in result


def test_run_symptom_agent_recommendation_request_names_ent_for_lightheadedness_with_vertigo(
    monkeypatch, db, ctx, clinic
):
    # Companion to the bare-lightheadedness test above: lightheadedness alongside
    # an actual ear/balance signal (vertigo, dizziness, spinning, or an ear
    # symptom) is the room-is-spinning kind, which IS ENT's territory — this must
    # still route to ENT, not General Medicine, even though bare lightheadedness
    # alone no longer does.
    for name in ("General Medicine", "ENT"):
        _make_dept_with_slot(db, clinic, name)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a recommendation request")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i feel light headed and the room keeps spining"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and started this morning"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "what do you recommend?", "en", history
    )

    assert "General Medicine" not in result
    assert "ENT" in result


def test_run_symptom_agent_recommendation_request_apologizes_for_height_growth_concerns_not_general_medicine(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "my height is not increasing i want to book an appointment"
    # had no keyword coverage in symptom_hints.py at all — "based on my symptoms
    # show me available dept that treats them" then fell all the way through to
    # the LLM with nothing grounded to reach for. This clinic has no dedicated
    # Endocrinology/growth department, so the LLM was left to guess and
    # apparently named one that doesn't exist here, producing a dead-end
    # "I couldn't find a department called that" instead of ever routing the
    # patient anywhere. A General Medicine fallback was tried next, then
    # instructed live to be removed too: General Medicine isn't actually
    # equipped for a growth/height concern either — a symptom with no genuine
    # matching specialty here must get an honest, gentle apology instead of
    # being rerouted to ANY department, real or not.
    _make_dept_with_slot(db, clinic, "General Medicine")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for an unsupported symptom category")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "my height is not increasing i want to book an appointment"),
        _row("assistant", "I'm happy to help you schedule a visit. Could you let me know which department..."),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "based on my symptoms show me available dept that treats them", "en", history
    )

    assert "General Medicine" not in result
    assert "sorry" in result.lower()
    assert "growth/height concerns" in result


def test_run_symptom_agent_recommendation_request_falls_through_when_nothing_hinted_yet(monkeypatch, db, ctx):
    # No real symptom has been described yet this session — nothing to base a
    # recommendation on, so this must fall through to the normal triage flow
    # (which will ask what's wrong) rather than returning an empty/broken reply.
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "What symptoms are you having?")

    result = symptom_agent.run_symptom_agent(db, ctx, "what do you recommend?", "en", [])

    assert result == "What symptoms are you having?"


def test_run_symptom_agent_first_message_combining_symptoms_and_recommendation_still_gets_normal_triage(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "I am having a bit of fever and body aches, what do you
    # recommend?" — symptom description AND the recommendation phrase in the SAME,
    # very first message — short-circuited straight to a department card, skipping
    # the normal PATH 2 screening questions (severity/duration) that must always come
    # first for a symptom nobody has triaged yet. Since this is the FIRST mention of
    # the symptom (nothing in prior history), this must still go through the normal
    # LLM triage flow, not the deterministic recommendation short-circuit.
    # A real General Medicine department is required so "fever and body aches"
    # resolves to something real — otherwise the general "no matching
    # department" apology (symptom_words_with_no_matching_department) would
    # short-circuit before any of this test's logic is even reached.
    _make_dept_with_slot(db, clinic, "General Medicine")

    monkeypatch.setattr(
        symptom_agent.llm,
        "run_tool_calling_agent",
        lambda *a, **k: "How long have you had the fever and body aches, and how severe are they?",
    )

    result = symptom_agent.run_symptom_agent(
        db, ctx, "I am having a bit of fever and body aches what do you recommend ?", "en", []
    )

    assert result == "How long have you had the fever and body aches, and how severe are they?"


@pytest.mark.parametrize("message", ["I am having pain in my jaw", "I am haiving nausea.."])
def test_run_symptom_agent_deterministically_asks_before_a_brand_new_first_message_symptom(
    monkeypatch, db, ctx, clinic, message
):
    # Reported live TWICE despite an explicit prompt instruction telling the model
    # not to do this ("MOST COMMON WAY THIS RULE GETS BROKEN" in llm.py's PATH 3):
    # a bare, mild-sounding symptom as the very first message of a brand new
    # session skipped straight to a department card with zero clarifying
    # questions. The prompt instruction alone wasn't a reliable enough guarantee,
    # so this is now a deterministic backstop — the LLM must not even be called
    # for this shape of message.
    # A real Dentistry department is required so "pain in my jaw" resolves to
    # something real — otherwise the general "no matching department" apology
    # would short-circuit before this backstop is even reached. Irrelevant to
    # the "nausea" case (that message hints nothing either way).
    _make_dept_with_slot(db, clinic, "Dentistry")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a brand-new first-message symptom")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = symptom_agent.run_symptom_agent(db, ctx, message, "en", [])

    assert "severe" in result.lower()
    assert "how long" in result.lower()


def test_run_symptom_agent_first_message_backstop_skips_when_duration_and_severity_already_given(
    monkeypatch, db, ctx, clinic
):
    # PATH 3's own stated exception: duration + severity already given together
    # means genuine clarification already happened, so the normal LLM triage flow
    # runs as usual instead of the deterministic backstop re-asking for the same
    # info the patient already provided. A real Dentistry department is required
    # so "jaw pain" resolves to something real — see the sibling test above.
    _make_dept_with_slot(db, clinic, "Dentistry")

    monkeypatch.setattr(
        symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Let me check availability for you."
    )

    result = symptom_agent.run_symptom_agent(db, ctx, "mild jaw pain for the past 2 days", "en", [])

    assert result == "Let me check availability for you."


@pytest.mark.parametrize("message", ["i have diabetes", "i think i have diabetes"])
def test_run_symptom_agent_first_message_backstop_skips_self_diagnosis_claims(monkeypatch, db, ctx, clinic, message):
    # Self-diagnosis claims are deliberately excluded from the backstop — asking
    # "is that mild, moderate, or severe?" in response to "I have diabetes"
    # reads as dismissive for something a patient is already alarmed about; the
    # established fix for this category is a fast, concise LLM-composed redirect
    # instead (see _is_self_diagnosis_claim's own comment in symptom_agent.py).
    # Uses diabetes rather than brain tumor/cancer here — those are now cancer-
    # orphan words that short-circuit to the honest apology before the LLM is
    # even called (see test_run_symptom_agent_apologizes_for_a_bare_cancer_or_
    # tumor_self_diagnosis_claim below), so they no longer exercise this
    # specific backstop-skip path. A real General Medicine department is
    # required so "diabetes" resolves to something real — otherwise the
    # general "no matching department" apology would short-circuit first.
    _make_dept_with_slot(db, clinic, "General Medicine")

    monkeypatch.setattr(
        symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Let me check availability for you."
    )

    result = symptom_agent.run_symptom_agent(db, ctx, message, "en", [])

    assert result == "Let me check availability for you."


def test_run_symptom_agent_first_message_backstop_does_not_apply_once_history_exists(monkeypatch, db, ctx, clinic):
    # The backstop is deliberately scoped to no REAL symptom having been described
    # yet — once a prior turn already described one (a headache here), the normal
    # LLM triage flow (which already has real conversation context to reason from)
    # applies, regardless of whether `history` is literally empty. A real General
    # Medicine department is required so "headache" resolves to something real —
    # otherwise the general "no matching department" apology would short-circuit
    # first (the "jaw" in the current message wouldn't resolve on its own, but
    # the headache from history resolving is enough to clear that gate).
    _make_dept_with_slot(db, clinic, "General Medicine")

    monkeypatch.setattr(
        symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Let me check availability for you."
    )

    history = [
        _row("user", "I have a headache"),
        _row("assistant", "How long have you had it, and how severe is it?"),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "I am also having pain in my jaw", "en", history)

    assert result == "Let me check availability for you."


def test_run_symptom_agent_first_message_backstop_applies_after_pure_small_talk(monkeypatch, db, ctx):
    # Reported live: "hey, my name is daud" -> greeting reply -> "i am having
    # nausea" — the backstop used to require `history` to be literally empty, so
    # this harmless small-talk exchange (no symptom content at all) was enough to
    # disqualify it from firing on what was genuinely the patient's first-ever
    # symptom message. "nausea" is PATH-3-eligible (not PATH-2), so the backstop's
    # own scope covers it once "brand new" correctly ignores preceding small talk.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a brand-new first symptom")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "hey,my name is daud"),
        # Deliberately doesn't end in "?" — a greeting reply that DOES (e.g.
        # "How can I assist you today?") trips a separate, unrelated PATH-2 signal
        # (_preceding_assistant_turn_looks_like_a_question), which isn't what this
        # test is about.
        _row("assistant", "Hello Daud! Nice to meet you."),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "i am having nausea", "en", history)

    assert "severe" in result.lower()
    assert "how long" in result.lower()


def test_run_symptom_agent_asks_before_a_new_symptom_category_introduced_later_in_the_conversation(
    monkeypatch, db, ctx, clinic
):
    # Reported live: after Orthopedics was correctly resolved for leg pain
    # (screened with 2 real questions), "i am also having pain in my eyes as
    # well" and then "...in my ear as well" both skipped straight to a
    # department card with ZERO clarifying questions. This backstop previously,
    # deliberately, only covered a session's first-ever symptom — a later,
    # genuinely new symptom category was assumed to be reliably screened by the
    # model's own judgment, which is exactly what failed here (twice in the same
    # conversation). "ear" pain is PATH-2-eligible (unlike "eyes") — confirms
    # this trigger fires regardless of PATH-2 status, unlike the first-message
    # backstop above. Real Department rows are required here (unlike most tests
    # in this file) — the new-category check resolves hints against real active
    # department NAMES, so with none in the DB it would silently find nothing.
    from app.models.department import Department

    for name in ("Orthopedics", "Ophthalmology", "ENT"):
        db.add(Department(clinic_id=clinic.id, name=name))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a newly introduced symptom")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i am having pain in my leg"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and bearable and been since morning"),
        _row("assistant", "Are you able to walk or put weight on the leg without increased pain or swelling?"),
        _row("user", "no im not able to put weight on it"),
        _row(
            "assistant",
            DOCTOR_OPTIONS_MARKER + json.dumps({"department_name": "Orthopedics", "note": "...", "doctors": []}),
        ),
    ]
    eyes_result = symptom_agent.run_symptom_agent(
        db, ctx, "i am also having pain in my eyes as well", "en", history
    )
    assert "severe" in eyes_result.lower()
    assert "how long" in eyes_result.lower()

    history_with_eyes = history + [
        _row("user", "i am also having pain in my eyes as well"),
        _row(
            "assistant",
            DOCTOR_OPTIONS_MARKER + json.dumps({"department_name": "Ophthalmology", "note": "...", "doctors": []}),
        ),
    ]
    ear_result = symptom_agent.run_symptom_agent(
        db, ctx, "i am also having pain in my ear as well", "en", history_with_eyes
    )
    assert "severe" in ear_result.lower()
    assert "how long" in ear_result.lower()


def test_run_symptom_agent_does_not_re_ask_severity_when_answering_its_own_screening_question(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "im sick" -> asked severity/duration -> answered ("3 days,
    # mild") -> asked "where exactly / what type of problem" -> answered "fever,
    # body ache" -> asked severity/duration AGAIN -> re-answered the same thing ->
    # asked "any other symptoms" -> answered "sore throat" -> asked severity/
    # duration a THIRD time. Each answer happened to hint a NEW department
    # (fever/ache -> General Medicine, sore throat -> ENT) that the prior turns
    # hadn't hinted, so _introduces_a_new_symptom_category correctly detected a
    # "new category" each time — but the patient was directly answering the
    # assistant's own immediately-preceding question, not volunteering an
    # unprompted new complaint. Real Department rows required, same as the test
    # above, since the new-category check resolves against real active names.
    from app.models.department import Department

    for name in ("General Medicine", "ENT"):
        db.add(Department(clinic_id=clinic.id, name=name))
    db.flush()

    # Unlike the "genuinely new category" test above, the backstop must NOT fire
    # here — normal flow proceeds to the LLM, so it's mocked to return a benign
    # reply (not made to fail-if-called) and the assertion is on THAT reply, not
    # on a hand-typed severity/duration question.
    monkeypatch.setattr(
        symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Let me check availability for you."
    )

    history = [
        _row("user", "im sick"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long you've had it?",
        ),
        _row("user", "its been 3 days, its mild rn"),
        _row(
            "assistant",
            "Where exactly are you feeling the symptoms (which part of your body) and what type of "
            "problem is it (e.g., pain, fever, cough, rash, etc.)?",
        ),
    ]
    fever_result = symptom_agent.run_symptom_agent(
        db, ctx, "all body feels warm, im feeling body ache, its fever", "en", history
    )
    assert fever_result == "Let me check availability for you."

    history_with_fever = history + [
        _row("user", "all body feels warm, im feeling body ache, its fever"),
        _row(
            "assistant",
            "Do you have any other symptoms such as cough, sore throat, rash, nausea, or vomiting?",
        ),
    ]
    throat_result = symptom_agent.run_symptom_agent(
        db, ctx, "sore throat", "en", history_with_fever
    )
    assert throat_result == "Let me check availability for you."


def test_run_symptom_agent_does_not_re_ask_for_the_same_symptom_category_being_screened(
    monkeypatch, db, ctx, clinic
):
    # A follow-up message continuing the SAME symptom (a swelling/weight-bearing
    # answer to the assistant's own screening question) must NOT be treated as a
    # "new symptom category" just because the resolved DEPARTMENT changes
    # (General Medicine -> Orthopedics) once the injury signal is present — that
    # would re-ask instead of ever reaching a real card.
    from app.models.department import Department

    for name in ("Orthopedics", "General Medicine"):
        db.add(Department(clinic_id=clinic.id, name=name))
    db.flush()

    card = json.dumps(
        {"note": "Based on your leg injury, Orthopedics would be appropriate.", "department_name": "Orthopedics",
         "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}]}
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: DOCTOR_OPTIONS_MARKER + card)

    history = [
        _row("user", "i am having pain in my leg"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and bearable and been since morning"),
        _row("assistant", "Are you able to walk or put weight on the leg without increased pain or swelling?"),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "no im not able to put weight on it", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)


def test_run_symptom_agent_corrects_the_primary_department_when_it_contradicts_the_hint_table(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "pain in my legs" -> screened (mild, since morning) ->
    # explicitly denied swelling/difficulty bearing weight -> the model's OWN
    # tool call still named Orthopedics ("...without swelling or difficulty
    # bearing weight, Orthopedics can evaluate this"). Bare limb pain with no
    # injury signal is General Medicine territory (see symptom_hints.py) — the
    # model directly contradicted its own established rule. This reply DID call
    # the real tool (a genuine DOCTOR_OPTIONS_MARKER card), so none of the
    # non-marker recovery nets (diagnosis-guard, faked-payload, advice-dump,
    # specialist-recommendation) could catch it — confirms the new primary-
    # department correction does.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    ortho = Department(clinic_id=clinic.id, name="Orthopedics")
    db.add_all([gen_med, ortho])
    db.flush()
    gen_med_doctor = Doctor(
        clinic_id=clinic.id, department_id=gen_med.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza", is_active=True,
    )
    ortho_doctor = Doctor(
        clinic_id=clinic.id, department_id=ortho.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Junaid Mirza", is_active=True,
    )
    db.add_all([gen_med_doctor, ortho_doctor])
    db.flush()
    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=gen_med_doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=ortho_doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    wrong_card = json.dumps(
        {
            "note": "Based on your mild leg pain without swelling or difficulty bearing weight, "
            "Orthopedics can evaluate this.",
            "department_name": "Orthopedics",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. Junaid Mirza", "specialization": None, "slots": []}],
        }
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: DOCTOR_OPTIONS_MARKER + wrong_card)

    history = [
        _row("user", "i am having pain in my legs"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and started since morning"),
        _row("assistant", "Is there any swelling, redness, or difficulty bearing weight on the leg?"),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "no there is no such symptoms", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "General Medicine"
    assert payload["doctors"][0]["doctor_name"] == "Dr. Ali Raza"


def test_run_symptom_agent_does_not_override_a_department_the_patient_explicitly_named(
    monkeypatch, db, ctx, clinic
):
    # The correction above must NOT fire when the patient themselves asked for
    # this department by name — that's a real, deliberate choice, not the model
    # freehanding one.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    ortho = Department(clinic_id=clinic.id, name="Orthopedics")
    db.add(ortho)
    db.flush()
    ortho_doctor = Doctor(
        clinic_id=clinic.id, department_id=ortho.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Junaid Mirza", is_active=True,
    )
    db.add(ortho_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=ortho_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    card = json.dumps(
        {
            "note": "Sure, here is Orthopedics availability.",
            "department_name": "Orthopedics",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. Junaid Mirza", "specialization": None, "slots": []}],
        }
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: DOCTOR_OPTIONS_MARKER + card)

    history = [
        _row("user", "i am having pain in my legs, no swelling or anything, mild, since morning"),
        _row("assistant", "Could you tell me more, or would you like to see Orthopedics directly?"),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "yes please show me Orthopedics availability", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Orthopedics"


def test_run_symptom_agent_recovers_a_real_department_when_the_model_advice_dumps_instead_of_routing(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "hey, my name is daud" -> "Hello Daud!... How can I assist
    # you today?" -> "i am having pain in my joints all along the body". The
    # greeting reply itself ending in "?" trips needs_path2_screening's
    # "preceding turn looks like a question" signal, making this PATH-2-eligible
    # — so the PATH-3 "zero questions" backstop deliberately doesn't apply (it
    # assumes PATH 2 already reliably asks a real question, exactly what failed
    # here). The model returned a long red-flag bullet list plus a
    # permission-seeking "would you like me to help you find a doctor?" tail,
    # never calling get_department_availability. It named no diagnosis and faked
    # no JSON, so neither existing recovery trigger caught it — this confirms the
    # new advice-dump detector does.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    ortho = Department(clinic_id=clinic.id, name="Orthopedics")
    db.add(ortho)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=ortho.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Junaid Mirza", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    advice_dump_reply = (
        "I'm sorry you're experiencing that discomfort. Joint pain throughout the body can have "
        "many causes, and it's important to have a professional evaluation.\n\n"
        "If you notice any of the following, please seek medical attention right away:\n\n"
        "- Sudden, severe swelling or redness in a joint\n"
        "- Fever, chills, or feeling very unwell\n"
        "- Inability to move a joint or bear weight\n\n"
        "Would you like me to help you find a doctor and check available appointment times?"
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: advice_dump_reply)

    history = [
        _row("user", "hey,my name is daud"),
        _row("assistant", "Hello Daud! Nice to meet you. How can I assist you today?"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "i am having pain in my joints all along the body", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Orthopedics"
    assert payload["doctors"][0]["doctor_name"] == "Dr. Junaid Mirza"
    assert "Would you like me to help" not in result


def test_run_symptom_agent_keeps_a_valid_path1_emergency_reply_intact(monkeypatch, db, ctx, clinic):
    # Reported live: "chest pain is mild and accompanied my sweating" — the model
    # correctly caught this as a real emergency-consistent combination via PATH 1's
    # EMERGENCY BACKSTOP secondary layer (no literal "severe" or other word-level
    # trigger fired for this one) and replied exactly as PATH 1 requires: a
    # one-sentence emergency statement, no tool call, first-aid steps formatted as
    # a numbered "1) ... 2) ..." list per the STRUCTURE RULE. That numbered-list
    # formatting is indistinguishable from _looks_like_an_advice_dump_instead_of_
    # routing's own trigger (2+ list-formatted lines) — LangSmith showed the model
    # produced this exact reply, but the patient received a Cardiology booking
    # card instead, because the advice-dump recovery discarded it and rebuilt a
    # department card from symptom keywords. This confirms the reply now survives
    # untouched instead. A real Cardiology department is required so "chest pain"
    # resolves to something real — otherwise the general "no matching
    # department" apology would short-circuit before the LLM is even called.
    _make_dept_with_slot(db, clinic, "Cardiology")

    emergency_reply = (
        "This sounds like an emergency—please call 1122 or go to the nearest ER right away.\n"
        "1) Sit down, stay calm, and loosen any tight clothing.\n"
        "2) If you feel faint, lie down with your legs raised slightly.\n"
        "3) Keep the phone handy and be ready to describe your symptoms to emergency responders."
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: emergency_reply)

    history = [
        _row("user", "i am having chest pain"),
        _row(
            "assistant",
            "Is the chest pain mild or moderate? Is it getting worse, spreading to your arm, "
            "jaw, or back, or accompanied by shortness of breath or sweating?",
        ),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "chest pain is mild and accompanied my sweating", "en", history
    )

    assert result == emergency_reply
    assert not result.startswith(DOCTOR_OPTIONS_MARKER)


def test_run_symptom_agent_overrides_path1_for_a_bare_self_diagnosis_claim(monkeypatch, db, ctx, clinic):
    # Reported live (original report used "brain tumour" — see the two cancer/
    # tumor-specific tests above for why that phrasing now short-circuits to an
    # apology instead; diabetes exercises the same PATH1-override mechanism
    # without tripping the newer cancer-orphan behavior): "i have diabetes" got
    # a full PATH 1 emergency reply while "i think i have diabetes" — the
    # identical claim, just hedged — correctly got a routine department
    # redirect. A bare self-diagnosis claim (nothing else describing severity/a
    # real red-flag) must get the same deterministic outcome regardless of the
    # "i think" hedge, so this confirms the override converts an
    # otherwise-valid PATH 1 reply into a real department card for the
    # unhedged phrasing.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    general_medicine = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(general_medicine)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=general_medicine.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Sana Qureshi", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    emergency_reply = (
        "This sounds like an emergency; please call 1122 or go to the nearest emergency department right away.\n"
        "1) Stay calm and sit or lie down in a safe position.\n"
        "2) Keep your airway clear and avoid any strenuous activity.\n"
        "3) If you feel faint or lose consciousness, have someone support your head and call emergency "
        "services immediately."
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: emergency_reply)

    result = symptom_agent.run_symptom_agent(db, ctx, "i have diabetes", "en", [])

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "General Medicine"
    assert "1122" not in result


def test_run_symptom_agent_keeps_path1_for_a_self_diagnosis_claim_with_a_real_red_flag(
    monkeypatch, db, ctx, clinic
):
    # A self-diagnosis claim genuinely accompanied by a real emergency signal
    # (here: seizure) must still be allowed to go through PATH 1 normally — the
    # override above is scoped to a BARE claim only, never a claim with real
    # red-flag content alongside it. A real Neurology department is added so
    # "seizure" hints something real — otherwise the top-level orphan-category
    # apology (from "brain tumour" with no matching Oncology department) would
    # short-circuit before the LLM is even called, which isn't what this test
    # is exercising.
    from app.models.department import Department

    db.add(Department(clinic_id=clinic.id, name="Neurology"))
    db.flush()

    emergency_reply = (
        "This sounds like an emergency; please call 1122 or go to the nearest emergency department right away.\n"
        "1) Keep the person safe from injury and do not restrain them.\n"
        "2) Time the seizure and note what you observe.\n"
        "3) Call emergency services if it lasts more than a few minutes."
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: emergency_reply)

    result = symptom_agent.run_symptom_agent(
        db, ctx, "i have a brain tumour and just had a seizure", "en", []
    )

    assert result == emergency_reply
    assert not result.startswith(DOCTOR_OPTIONS_MARKER)


def test_run_symptom_agent_recovers_when_the_model_recommends_a_specialist_in_plain_prose(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "issue in chewing any solid thing" got a reply in plain
    # PROSE, not a bulleted list ("I recommend scheduling an appointment with a
    # dentist or an oral-maxillofacial specialist..."), so it slipped past the
    # bullet-line advice-dump detector too — no diagnosis language, no faked JSON,
    # no bullets, just a paragraph naming a specialist type without ever calling
    # get_department_availability. Confirms the new specialist-recommendation
    # detector catches this prose-form variant.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    dentistry = Department(clinic_id=clinic.id, name="Dentistry")
    db.add(dentistry)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=dentistry.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Qureshi", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    prose_recommendation_reply = (
        "I'm sorry you're having trouble chewing. That can be caused by a number of issues such as "
        "dental problems, jaw joint discomfort, or muscle strain. I recommend scheduling an appointment "
        "with a dentist or an oral-maxillofacial specialist so they can examine you and determine the cause.\n\n"
        "If you'd like, I can help you find a suitable department and check the next available slots for "
        "an appointment. Just let me know how you'd like to proceed."
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: prose_recommendation_reply)

    history = [
        _row("user", "hey,my name is daud"),
        _row("assistant", "Hello Daud! Nice to meet you. How can I assist you today?"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "i am having issue in chewing any solid thing", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Dentistry"
    assert "I recommend scheduling" not in result


def test_router_rule0_5_find_me_a_suitable_dept_routes_to_symptom_general():
    # Reported live: "find me a suitable dept" (after already describing a real
    # symptom in the same session) matched none of the recommendation-request
    # phrasings, fell through the router's other rules, and landed on
    # general_info_agent — which has no tool bound to query real department
    # availability at all, only KB retrieval, so it just free-texted a department
    # name with no real card, no doctors, no slots ever shown.
    from app.services.orchestrator.router import _heuristic_classify

    history = [
        _row("user", "i am having issue in chewing any solid thing"),
        _row("assistant", "I recommend scheduling an appointment with a dentist."),
    ]
    assert _heuristic_classify("find me a suitable dept", history) == "symptom_general"


def test_run_symptom_agent_recovers_when_the_model_asks_which_department_in_plain_prose(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "i am having pain in my leg" -> screened (severe, 5 days) ->
    # emergency reply -> "no i dont need ER" -> "i want to book an appointment
    # here" got "Sure...Could you let me know which department..." in free text
    # instead of calling the tool, even though severity+duration were already
    # established and the hint table already supports General Medicine for bare
    # leg pain with no injury signal. The chatbot should compute this itself, not
    # defer the decision back to the patient.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    general_medicine = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(general_medicine)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=general_medicine.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Bilal Hashmi", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    department_question_reply = (
        "Sure, I can help you schedule an appointment. Could you let me know which department or "
        "specialist you'd like to see for your leg pain (for example, Orthopedics or a general "
        "physician)?"
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: department_question_reply)

    history = [
        _row("user", "i am having pain in my leg and its severe"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "severe since 5 days"),
        _row("assistant", "This sounds like an emergency. Call 1122 or go to the nearest ER right away."),
        _row("user", "no i dont need ER"),
        _row("assistant", "I understand you don't want to go to the ER, but this is still an emergency."),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "its not that bad i want to book an appointment here", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "General Medicine"
    assert "which department" not in result


def test_run_symptom_agent_recovers_when_the_model_names_a_real_department_in_plain_prose(
    monkeypatch, db, ctx, clinic
):
    # Reported live: same conversation, next turn — "i dont know what dept can be
    # best fir for it" got "the Orthopedics department is usually the most
    # appropriate" as free text, naming a REAL department (not a specialist
    # title, so the existing specialist-word detector missed it) without ever
    # calling get_department_availability. This contradicted the hint table's
    # own answer (General Medicine for bare leg pain), which later caused the
    # symptom-vs-department mismatch check in appointment_agent.py to fire
    # against the bot's own prior recommendation.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    general_medicine = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(general_medicine)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=general_medicine.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Bilal Hashmi", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    orthopedics_prose_reply = (
        "For leg pain, especially when it's severe and has lasted several days, the Orthopedics "
        "department is usually the most appropriate. Let me know which option you'd like."
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: orthopedics_prose_reply)

    history = [
        _row("user", "i am having pain in my leg and its severe"),
        _row("user", "severe since 5 days"),
        _row("user", "its not that bad i want to book an appointment here"),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "i dont know what dept can be best fir for it", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "General Medicine"
    assert "Orthopedics department" not in result


def test_run_symptom_agent_does_not_hint_a_department_with_no_matching_symptom_words(monkeypatch, db, ctx, clinic):
    # A real Cardiology department is created here matching what the model's own
    # card already claims — the general "no matching department" apology (see
    # symptom_words_with_no_matching_department) would otherwise short-circuit
    # "chest pain" before the LLM is even called, at a clinic with no Cardiology
    # at all; this test is specifically about the DOWNSTREAM correction-net
    # logic (hinted_for_primary), which needs a real matching department to
    # reach at all.
    _make_dept_with_slot(db, clinic, "Cardiology")

    card = json.dumps(
        {
            "note": "Based on your description, this sounds like something Cardiology should look at.",
            "department_name": "Cardiology",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}],
        }
    )
    reply_with_marker = DOCTOR_OPTIONS_MARKER + card
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: reply_with_marker)

    result = symptom_agent.run_symptom_agent(db, ctx, "chest pain", "en", [])

    assert result == reply_with_marker


def test_run_symptom_agent_recovers_a_real_department_when_the_model_diagnosed_instead_of_routing(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "fasting blood sugar 200+" then "excessive thirst and frequent
    # urination" (the diabetes triad) got a free-text reply naming the condition
    # outright instead of calling get_department_availability — PATH 3's own "always
    # ends with the tool call" rule was skipped. The diagnosis guard would normally
    # strip that down to a generic "tell me more" redirect with nothing to route to,
    # losing the patient's real turn entirely. This confirms the recovery: a real
    # department gets fetched and shown instead of the dead-end redirect.
    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot
    from datetime import datetime, timedelta, timezone

    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(gen_med)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id,
        department_id=gen_med.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza",
        is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    diagnostic_reply = "This sounds like it could be diabetes. You should get it checked soon."
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: diagnostic_reply)

    history = [
        _row("user", "my blood sugar level is 200+ while fasting"),
        _row(
            "assistant",
            "Do you have any other symptoms such as excessive thirst, frequent urination, or feeling unusually tired?",
        ),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "yes excessive thirst and frequent urination", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "General Medicine"
    assert payload["doctors"][0]["doctor_name"] == "Dr. Ali Raza"
    # The diagnostic free text must never reach the patient.
    assert "diabetes" not in result.lower()


def test_run_symptom_agent_recovers_when_the_model_fakes_a_tool_payload_instead_of_calling_it(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "pain in my chest and in testies as well" (mild, bearable, 2
    # days) got a reply that never called get_department_availability at all — the
    # model free-texted a recommendation paragraph, then hand-typed a fragment
    # mimicking the tool's own JSON shape ({"department_name": ..., "note": ...}) as
    # plain text, naming only ONE department (General Medicine, dropping the
    # chest-pain->Cardiology hint entirely). This reply contains no diagnostic
    # phrasing at all ("a General Medicine appointment would be appropriate" isn't
    # "you have X"), so the existing diagnosis-violation recovery net never fired —
    # confirms the new faked-payload detector catches this case too and rebuilds a
    # real card for the hinted department.
    #
    # This clinic has no Urology department, so — per the later "never reroute an
    # unsupported symptom to General Medicine just because it exists" fix — the
    # testicular-pain half of this complaint is no longer grounded to General
    # Medicine at all; only the genuine Cardiology hint (from the chest pain) gets
    # rebuilt into a real card here. The recovery mechanism only ever reconstructs
    # cards from real, grounded hints (never the model's own faked department
    # name), so this is the correct, narrower outcome, not a regression.
    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot
    from datetime import datetime, timedelta, timezone

    cardio = Department(clinic_id=clinic.id, name="Cardiology")
    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    db.add_all([cardio, gen_med])
    db.flush()
    cardio_doctor = Doctor(
        clinic_id=clinic.id, department_id=cardio.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}", full_name="Dr. Ahmed Farooq", is_active=True,
    )
    gen_med_doctor = Doctor(
        clinic_id=clinic.id, department_id=gen_med.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}", full_name="Dr. Ali Raza", is_active=True,
    )
    db.add_all([cardio_doctor, gen_med_doctor])
    db.flush()
    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=cardio_doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=gen_med_doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    faked_payload_reply = (
        "Based on what you've described, a General Medicine appointment would be "
        "appropriate to assess both the chest and testicular discomfort.\n\n"
        "Note: This is for a routine follow-up; if you notice worsening pain, new "
        "symptoms, or any concerning changes, seek immediate medical care.\n\n"
        '{\n  "department_name": "General Medicine",\n'
        '  "note": "Chest and testicular discomfort assessment"\n}'
    )
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: faked_payload_reply)

    history = [
        _row("user", "i am having pain in my chest and in testies as well"),
        _row(
            "assistant",
            "Is the chest pain severe, bearable, or mild? Also, is the testicular pain "
            "constant or does it come and go?",
        ),
    ]
    result = symptom_agent.run_symptom_agent(
        db, ctx, "the pain is mild and bearable and it is from 2 days with no other symptoms", "en", history
    )

    # The fake, hand-typed JSON fragment must never reach the patient verbatim.
    assert "Note: This is for a routine follow-up" not in result
    assert "Cardiology" in result
    assert "Dr. Ahmed Farooq" in result
    assert "General Medicine" not in result
    assert "Dr. Ali Raza" not in result


def test_run_symptom_agent_apologizes_for_a_bare_cancer_or_tumor_self_diagnosis_claim(monkeypatch, db, ctx, clinic):
    # Instructed live: this clinic has no Oncology department, so a truly BARE
    # cancer/tumor claim with no body part/organ named at all (nothing for the
    # symptom-hint table to route on) must get an honest "no matching
    # department" apology instead of being left to the LLM to freehand a guess.
    # A claim that DOES name a body part — "brain tumour" — is a different,
    # deliberately NOT-apologized case: see the sibling test below, since a
    # neurologist genuinely is a real first stop for that even without
    # Oncology. Only a Neurology department exists here (irrelevant to this
    # bare claim, since nothing hints Neurology without "brain"/"seizure"), so
    # this fires before the LLM is even called.
    from app.models.department import Department

    neuro = Department(clinic_id=clinic.id, name="Neurology")
    db.add(neuro)
    db.flush()

    monkeypatch.setattr(
        symptom_agent.llm,
        "run_tool_calling_agent",
        lambda *a, **k: pytest.fail("LLM should never be called — the apology short-circuits first"),
    )

    result = symptom_agent.run_symptom_agent(db, ctx, "i think i have cancer", "en", [])

    assert not result.startswith(DOCTOR_OPTIONS_MARKER)
    assert "Neurology" not in result
    assert "sorry" in result.lower()


def test_run_symptom_agent_apology_does_not_repeat_for_an_unrelated_later_message(monkeypatch, db, ctx, clinic):
    # Reported live: after the apology above fired for "I think I have cancer,"
    # a completely unrelated follow-up ("what's 2+2") got the EXACT SAME
    # apology again instead of the normal off-topic redirect. Root cause: the
    # apology check scans message+history TOGETHER, so "cancer" from an
    # earlier turn never left the combined word-set even though the CURRENT
    # message has nothing to do with it. The gate now requires the CURRENT
    # message itself to still be symptom-shaped (or an explicit "based on my
    # symptoms" recommendation request) — an unrelated message like this must
    # fall through to the normal LLM flow (and its off-topic guardrail)
    # instead of repeating stale apology text.
    off_topic_reply = "I don't have that information here — for anything outside the clinic, please ask elsewhere."
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: off_topic_reply)

    history = [
        _row("user", "i think i have cancer"),
        _row(
            "assistant",
            "I'm sorry, this clinic doesn't have a specialist department for cancer/tumor-related "
            "symptoms. I'd recommend visiting another hospital or clinic for this. Is there anything "
            "else I can help you with?",
        ),
    ]
    result = symptom_agent.run_symptom_agent(db, ctx, "whats 2+2", "en", history)

    assert result == off_topic_reply
    assert "sorry" not in result.lower()


def test_run_symptom_agent_routes_a_brain_tumor_claim_to_neurology_even_without_oncology(
    monkeypatch, db, ctx, clinic
):
    # Instructed live: "isn't brain tumour treated by neurologist?" — clinically
    # yes, a suspected brain tumor is genuinely first evaluated by Neurology
    # (Oncology treats it once confirmed, but Neurology does the initial
    # workup/referral). Unlike a bare "I have cancer" with no body part named
    # (see the apology test above), naming "brain" specifically still routes to
    # Neurology normally, even when this clinic has no Oncology department.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    neurology = Department(clinic_id=clinic.id, name="Neurology")
    db.add(neurology)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=neurology.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Sana Qureshi", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=20),
        status="open",
    ))
    db.flush()

    diagnostic_reply = "This sounds like it could be a brain tumor. You should get it checked soon."
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: diagnostic_reply)

    result = symptom_agent.run_symptom_agent(db, ctx, "i think i have a brain tumor", "en", [])

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Neurology"
    assert payload["doctors"][0]["doctor_name"] == "Dr. Sana Qureshi"


def test_run_symptom_agent_routes_a_cancer_claim_to_a_real_oncology_department_when_one_exists(
    monkeypatch, db, ctx, clinic
):
    # The apology above only fires when this clinic genuinely has no matching
    # department — a clinic that DOES have Oncology should still route there
    # normally, same as every other _ORPHAN_SYMPTOM_CATEGORIES entry (e.g.
    # Urology for urinary symptoms).
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    oncology = Department(clinic_id=clinic.id, name="Oncology")
    db.add(oncology)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id, department_id=oncology.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Amna Siddiqui", is_active=True,
    )
    db.add(doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=20),
        status="open",
    ))
    db.flush()

    diagnostic_reply = "This sounds like it could be a brain tumor. You should get it checked soon."
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: diagnostic_reply)

    result = symptom_agent.run_symptom_agent(db, ctx, "i have brain tumor", "en", [])

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Oncology"
    assert payload["doctors"][0]["doctor_name"] == "Dr. Amna Siddiqui"
    assert "checked soon" not in result.lower()


def test_run_symptom_agent_leaves_a_diagnostic_reply_alone_when_no_department_can_be_hinted(monkeypatch, db, ctx):
    # No symptom-word hint matches anything real here — nothing safe to route to,
    # so the diagnostic reply must pass through untouched (chat.py's own
    # enforce_no_diagnosis call still catches it afterward at the normal layer).
    # Duration + severity already given so this reaches the LLM normally instead
    # of the first-message backstop — this test is about the diagnosis-recovery
    # net's pass-through behavior, not the backstop itself.
    diagnostic_reply = "This sounds like it could be a rare condition. You should get it checked."
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: diagnostic_reply)

    result = symptom_agent.run_symptom_agent(
        db, ctx, "I have been feeling strange for the past few days, it's moderate", "en", []
    )

    assert result == diagnostic_reply


def test_run_symptom_agent_does_not_refetch_when_note_only_names_the_already_covered_department(
    monkeypatch, db, ctx, clinic
):
    # A real Cardiology department is created here so the general "no matching
    # department" apology doesn't short-circuit "chest pain" before the LLM is
    # even called — see the sibling test above for why.
    _make_dept_with_slot(db, clinic, "Cardiology")

    card = json.dumps(
        {
            "note": "Based on your description, this sounds like something Cardiology should look at.",
            "department_name": "Cardiology",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. A", "specialization": None, "slots": []}],
        }
    )
    reply_with_marker = DOCTOR_OPTIONS_MARKER + card
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: reply_with_marker)

    result = symptom_agent.run_symptom_agent(db, ctx, "chest pain", "en", [])

    # Single mention of its own department only — must pass through unchanged,
    # never re-wrapped into a DEPARTMENT_LIST_MARKER for just one department.
    assert result == reply_with_marker


# =====================================================================================
# appointment_agent — including the handoff mechanism (highest-risk new code)
# =====================================================================================


def test_appointment_agent_prompt_includes_appointment_action_rules_only():
    prompt = appointment_agent._build_system_prompt("English", None)
    assert "get_my_appointments returns structured data" in prompt
    assert "Only call book_appointment once the patient has clearly picked" in prompt
    # Symptom-triage and find_doctors_by_name-specific content must NOT leak in.
    assert "SYMPTOM TRIAGE RULE" not in prompt
    assert "NEVER treat a patient-typed doctor name as a confirmed match" not in prompt


def test_appointment_agent_prompt_includes_previously_shown_options_when_given():
    payload = {"doctors": [{"doctor_name": "Dr. Ahmed Khan", "department_name": "Cardiology"}]}
    prompt = appointment_agent._build_system_prompt("English", payload)
    assert "PREVIOUSLY SHOWN OPTIONS" in prompt
    assert "Dr. Ahmed Khan" in prompt


def test_appointment_agent_prompt_omits_previously_shown_options_when_none():
    prompt = appointment_agent._build_system_prompt("English", None)
    assert "PREVIOUSLY SHOWN OPTIONS" not in prompt


def test_most_recent_availability_marker_finds_the_latest_one():
    history = [
        _row("assistant", DEPARTMENT_LIST_MARKER + '{"departments": [{"department_name": "Cardiology"}]}'),
        _row("user", "book me with the second one"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + '{"doctors": [{"doctor_name": "Dr. X"}]}'),
    ]
    payload = appointment_agent._most_recent_availability_marker(history)
    assert payload == {"doctors": [{"doctor_name": "Dr. X"}]}


def test_most_recent_availability_marker_none_when_absent():
    history = [_row("user", "hi"), _row("assistant", "Hello!")]
    assert appointment_agent._most_recent_availability_marker(history) is None


def test_most_recent_availability_marker_ignores_booking_confirmed_card():
    # BOOKING_MARKER isn't a source of doctor/slot candidates (it's a done deal, not
    # options to pick from) — only DOCTOR_OPTIONS/DEPARTMENT_LIST are recovered.
    history = [_row("assistant", BOOKING_MARKER + '{"doctor_name": "Dr. X"}')]
    assert appointment_agent._most_recent_availability_marker(history) is None


def test_disambiguation_marker_reply_is_composed_from_real_rows_not_freehanded():
    matches = [
        SimpleNamespace(full_name="Dr. Ahmed Khan", department_name="Cardiology"),
        SimpleNamespace(full_name="Dr. Ahmed Raza", department_name="Neurology"),
    ]
    reply = appointment_agent._disambiguation_marker_reply(matches)
    assert reply.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(reply[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "doctor_name"
    assert payload["candidates"] == [
        {"doctor_name": "Dr. Ahmed Khan", "department_name": "Cardiology"},
        {"doctor_name": "Dr. Ahmed Raza", "department_name": "Neurology"},
    ]
    assert "Dr. Ahmed Khan" in payload["question"]
    assert "Dr. Ahmed Raza" in payload["question"]


def test_run_appointment_agent_ambiguous_doctor_name_short_circuits_before_any_llm_call(
    monkeypatch, db, ctx, doctor, other_doctor
):
    # Both fixtures are named "Dr. Ahmed ..." — a real, deterministic 2-match
    # ambiguity. The LLM must never be invoked for this turn at all.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called on an ambiguous handoff")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(db, ctx, "book me with Dr Ahmed", "en", [])

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    names = {c["doctor_name"] for c in payload["candidates"]}
    assert names == {"Dr. Ahmed Khan", "Dr. Ahmed Raza"}


def test_run_appointment_agent_unambiguous_doctor_name_proceeds_to_the_llm(monkeypatch, db, ctx, doctor):
    # Only one real "Ahmed" this time (other_doctor fixture not used) — must fall
    # through to the normal agent loop, not short-circuit.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["called"] = True
        return "some reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    result = appointment_agent.run_appointment_agent(db, ctx, "book me with Dr Ahmed at 4pm", "en", [])

    assert captured.get("called") is True
    assert result == "some reply"


def test_run_appointment_agent_tells_the_patient_plainly_when_a_named_doctor_does_not_exist(
    monkeypatch, db, ctx, doctor
):
    # Reported live: "dr zeeshan qureshi in pulmonology dept" (no such doctor at
    # this clinic — "doctor" fixture is "Dr. Ahmed Khan", and shares no word with
    # "Zeeshan Qureshi" at all, so find_doctors_by_name's word-overlap fallback
    # can't produce even a partial match) silently showed the WHOLE Pulmonology
    # department's real doctors instead of telling the patient plainly that "Dr.
    # Zeeshan Qureshi" isn't here. Must never reach the LLM at all — same "never
    # silently substitute a different doctor" principle as the ambiguous-name
    # short-circuit above. NOTE: this is deliberately a name with ZERO word
    # overlap with any real doctor — see
    # test_run_appointment_agent_suggests_a_partial_name_match_instead_of_flatly_refusing
    # just below for the (also deliberate) different behavior when the attempted
    # name DOES share a word with a real doctor.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the named doctor doesn't exist")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "dr zeeshan qureshi in pulmonology dept", "en", []
    )

    assert "Zeeshan Qureshi" in result
    assert "couldn't find a doctor" in result.lower()


def test_run_appointment_agent_suggests_a_partial_name_match_instead_of_flatly_refusing(
    monkeypatch, db, ctx, doctor
):
    """"dr ahmed rana in pulmonology dept" shares "ahmed" with the real "Dr. Ahmed
    Khan" fixture doctor — find_doctors_by_name's word-overlap fallback surfaces
    that as a partial match (see its own docstring), so the patient should be
    asked a one-line confirming question ("Did you mean Dr. Ahmed Khan?") rather
    than being flatly told no such doctor exists. This is deliberate, intended
    behavior — not the same case as the zero-overlap "doesn't exist" test above."""
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Did you mean Dr. Ahmed Khan in Cardiology?"
    )

    result = appointment_agent.run_appointment_agent(
        db, ctx, "dr ahmed rana in pulmonology dept", "en", []
    )

    assert "Ahmed Khan" in result
    assert "couldn't find a doctor" not in result.lower()


def test_run_appointment_agent_gives_the_deterministic_how_to_book_steps_after_slots_shown(
    monkeypatch, db, ctx
):
    """Requested directly: once a slot list is on screen, "how do I book an
    appointment" gets an exact, deterministic two-step answer, never left to the
    LLM to freehand."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a deterministic how-to-book reply")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + '{"doctors": []}')]
    result = appointment_agent.run_appointment_agent(db, ctx, "how do i book an appointment", "en", history)

    assert "1)" in result and "2)" in result
    assert "clickable" in result.lower()


def test_run_appointment_agent_how_to_book_falls_through_normally_with_no_slots_shown(
    monkeypatch, db, ctx
):
    """The deterministic how-to-book reply only fires once a real slot list is
    actually on screen — with nothing shown yet, "how do I book an appointment"
    has nothing to point back at "above", so it must fall through to the normal
    LLM/tool-calling flow instead."""
    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", lambda *a, **k: "some normal reply")

    result = appointment_agent.run_appointment_agent(db, ctx, "how do i book an appointment", "en", [])

    assert result == "some normal reply"


def test_run_appointment_agent_answers_most_recent_cancelled_appointment_from_real_db(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Reported live: "what's my most recent cancelled appointment" got a reply
    # naming a doctor, department, AND date that matched NOTHING in the
    # patient's real appointment history — the LLM fabricated plausible-
    # sounding but entirely fake details instead of using the tool's real
    # result. Answered entirely from the real DB now, never reaching the LLM.
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    sooner_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="booked",
    )
    later_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=10),
        end_utc=datetime.now(timezone.utc) + timedelta(days=10, minutes=30),
        status="booked",
    )
    db.add_all([sooner_slot, later_slot])
    db.flush()
    sooner_appt = Appointment(
        clinic_id=clinic.id, slot_id=sooner_slot.id, patient_id=patient.id, doctor_id=doctor.id, status="cancelled",
    )
    later_appt = Appointment(
        clinic_id=clinic.id, slot_id=later_slot.id, patient_id=patient.id, doctor_id=doctor.id, status="cancelled",
    )
    db.add_all([sooner_appt, later_appt])
    db.flush()
    # later_appt's slot is further in the future, but it was cancelled LAST
    # (most recently) — proves the reply follows cancellation recency, not
    # slot recency.
    sooner_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(hours=2)
    later_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a deterministic recency reply")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "what's my most recent cancelled appointment", "en", []
    )

    assert doctor.full_name in result
    assert "cancelled" in result.lower()
    # The later-scheduled appointment (later_appt) is the one actually cancelled
    # most recently — its slot is 10 days out, not sooner_appt's 1 day out.
    from app.services.chat_tools import _format_when

    tz = appointment_agent._clinic_timezone(db, ctx.clinic_id)
    assert _format_when(later_slot.start_utc, tz) in result
    assert _format_when(sooner_slot.start_utc, tz) not in result


def test_run_appointment_agent_answers_earliest_cancelled_appointment_from_real_db(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Reported live as a follow-up to the "most recent" fix above: "earliest"
    # asks for the OPPOSITE end of the same history — same two appointments/
    # timestamps as the test above, but "earliest" must surface sooner_appt
    # (cancelled FIRST), not later_appt (cancelled most recently).
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    sooner_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="booked",
    )
    later_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=10),
        end_utc=datetime.now(timezone.utc) + timedelta(days=10, minutes=30),
        status="booked",
    )
    db.add_all([sooner_slot, later_slot])
    db.flush()
    sooner_appt = Appointment(
        clinic_id=clinic.id, slot_id=sooner_slot.id, patient_id=patient.id, doctor_id=doctor.id, status="cancelled",
    )
    later_appt = Appointment(
        clinic_id=clinic.id, slot_id=later_slot.id, patient_id=patient.id, doctor_id=doctor.id, status="cancelled",
    )
    db.add_all([sooner_appt, later_appt])
    db.flush()
    sooner_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(hours=2)
    later_appt.cancelled_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a deterministic recency reply")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "what's my earliest cancelled appointment", "en", []
    )

    assert doctor.full_name in result
    assert "cancelled" in result.lower()
    from app.services.chat_tools import _format_when

    tz = appointment_agent._clinic_timezone(db, ctx.clinic_id)
    assert _format_when(sooner_slot.start_utc, tz) in result
    assert _format_when(later_slot.start_utc, tz) not in result


def test_run_appointment_agent_most_recent_cancelled_reply_handles_no_cancelled_appointments(
    monkeypatch, db, ctx
):
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    result = appointment_agent.run_appointment_agent(
        db, ctx, "what was my last cancelled appointment", "en", []
    )

    assert "don't have any cancelled appointments" in result.lower()


def test_run_appointment_agent_answers_most_recent_completed_appointment_from_real_db(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) - timedelta(days=3),
        end_utc=datetime.now(timezone.utc) - timedelta(days=3) + timedelta(minutes=30),
        status="booked",
    )
    db.add(slot)
    db.flush()
    appt = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="completed",
    )
    db.add(appt)
    db.flush()

    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    result = appointment_agent.run_appointment_agent(
        db, ctx, "what was my most recent completed appointment", "en", []
    )

    assert doctor.full_name in result
    assert "completed" in result.lower()


def test_run_appointment_agent_answers_most_recent_missed_appointment_from_real_db(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) - timedelta(days=3),
        end_utc=datetime.now(timezone.utc) - timedelta(days=3) + timedelta(minutes=30),
        status="booked",
    )
    db.add(slot)
    db.flush()
    appt = Appointment(
        clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="no_show",
    )
    db.add(appt)
    db.flush()

    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    result = appointment_agent.run_appointment_agent(db, ctx, "what appointment did i miss most recently", "en", [])

    assert doctor.full_name in result
    assert "missed" in result.lower()


def test_run_appointment_agent_plain_cancelled_list_request_falls_through_to_llm(monkeypatch, db, ctx):
    # No recency word ("most recent"/"latest"/"last") — a plain list request is
    # deliberately left to the normal LLM/tool-calling flow, which already
    # handles a full list fine.
    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", lambda *a, **k: "some normal reply")

    result = appointment_agent.run_appointment_agent(db, ctx, "show my cancelled appointments", "en", [])

    assert result == "some normal reply"


def test_run_appointment_agent_recovers_a_doctor_named_only_in_assistant_prose(monkeypatch, db, ctx, doctor):
    """Reported live: "i am having blury vision" got answered by general_info_agent
    with plain KB prose naming a real doctor (never a structured DOCTOR_OPTIONS
    card, and the PATIENT never typed the doctor's name themselves) — then "show
    me available slots for him" failed to resolve "him" at all, since
    _most_recently_named_doctor used to only scan the patient's own messages.
    Must now fall back to the assistant's last plain-prose reply and recover the
    same real doctor from it."""
    history = [
        _row("user", "i am having blury vision"),
        _row(
            "assistant",
            f"If you're experiencing blurry vision, you may want to see an ophthalmologist. "
            f"{doctor.full_name} sees patients for general ophthalmology on Tuesday, Thursday "
            f"and Saturday from 10:00 am to 6:00 pm.",
        ),
    ]
    monkeypatch.setattr(
        appointment_agent.llm,
        "run_tool_calling_agent",
        lambda *a, **k: f"Did you mean {doctor.full_name}?",
    )

    result = appointment_agent.run_appointment_agent(db, ctx, "show me available slots for him", "en", history)

    assert doctor.full_name in result
    assert "which department" not in result.lower()


def test_run_appointment_agent_reschedule_cap_reached_is_told_before_showing_new_slots(
    monkeypatch, db, ctx, clinic, department, doctor, patient
):
    # Same "check before asking" principle as the 2-hour cancel/reschedule
    # cutoff: if the daily reschedule cap for this appointment's own department/
    # day is already used up, the patient should be told directly instead of
    # being shown a full list of new times to pick from, only for the actual
    # reschedule to fail once they've chosen one. See
    # appointment_agent._reschedule_cap_reached.
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.appointment_department_day_reschedule_use import AppointmentDepartmentDayRescheduleUse
    from app.models.slot import Slot

    appt_start = datetime.now(timezone.utc) + timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=appt_start, end_utc=appt_start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    appointment = Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed")
    db.add(appointment)
    db.flush()

    # Pre-fill the reschedule cap (2) for (Cardiology, appt_start's date) — as if
    # two reschedules out of that department/day already happened.
    local_date = appt_start.date()
    db.add(AppointmentDepartmentDayRescheduleUse(clinic_id=clinic.id, patient_id=patient.id, department_id=department.id, appointment_id=appointment.id, local_date=local_date))
    db.add(AppointmentDepartmentDayRescheduleUse(clinic_id=clinic.id, patient_id=patient.id, department_id=department.id, appointment_id=appointment.id, local_date=local_date))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called once the reschedule cap is already reached")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(db, ctx, "i want to reschedule my appointment", "en", [])

    assert "Cardiology" in result
    assert "already been rescheduled" in result.lower()


def test_run_appointment_agent_suggests_booking_elsewhere_after_a_cancel_cutoff_reply(monkeypatch, db, ctx):
    # Requested: a patient told an appointment "can no longer be cancelled
    # online" (the 2-hour cutoff reply) who then asks "what can I do now?"
    # should get real next-step guidance, not a generic/off-topic answer.
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM for this deterministic reply")),
    )
    history = [
        _row("user", "cancel my appointment"),
        _row(
            "assistant",
            "Your appointment with Dr. Ahmed in Cardiology on Mon, Aug 24 at 9:00 AM is coming up in "
            "less than 2 hours, so it can no longer be cancelled online. Please contact the clinic "
            "directly for last-minute changes.",
        ),
    ]

    result = appointment_agent.run_appointment_agent(db, ctx, "what can i do now", "en", history)

    assert "browse" in result.lower()
    assert "book" in result.lower()


def test_run_appointment_agent_suggests_next_steps_after_a_reschedule_cutoff_reply(monkeypatch, db, ctx):
    # Same cutoff reply, reschedule-worded — cancelling is ALSO blocked by the
    # same 2-hour window here, so the guidance must not suggest cancel-and-
    # rebook (unlike the reschedule-CAP case below, a genuinely different limit
    # that doesn't block cancelling).
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM for this deterministic reply")),
    )
    history = [
        _row("user", "reschedule my appointment"),
        _row(
            "assistant",
            "Your appointment with Dr. Ahmed in Cardiology on Mon, Aug 24 at 9:00 AM is coming up in "
            "less than 2 hours, so it can no longer be rescheduled online. Please contact the clinic "
            "directly for last-minute changes.",
        ),
    ]

    result = appointment_agent.run_appointment_agent(db, ctx, "what should i do now?", "en", history)

    assert "contact the clinic" in result.lower()
    # Cancelling is explained as ALSO blocked by the same cutoff, never offered
    # as an actionable alternative the way the reschedule-cap case does.
    assert "cancel this appointment and book" not in result.lower()
    assert "cancel it and book" not in result.lower()


def test_run_appointment_agent_suggests_cancel_and_rebook_after_a_reschedule_cap_reply(monkeypatch, db, ctx):
    # The reschedule daily-cap reply is a genuinely different limit than the
    # cutoff above — cancelling is NOT subject to it, so "cancel and book new"
    # is real, actionable advice here (unlike the cutoff case above).
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM for this deterministic reply")),
    )
    history = [
        _row("user", "reschedule my appointment"),
        _row(
            "assistant",
            "Your appointment with Dr. Ahmed in Cardiology has already been rescheduled 2 times "
            "for that department for this day — please contact the clinic directly for any further changes to it.",
        ),
    ]

    result = appointment_agent.run_appointment_agent(db, ctx, "what can i do now", "en", history)

    assert "cancel" in result.lower()
    assert "contact the clinic" in result.lower()


def test_run_appointment_agent_booking_cap_reached_is_told_before_confirmation(
    monkeypatch, db, ctx, clinic, department, doctor, patient
):
    """Reported live: a patient picked a slot in a department already at its
    daily booking cap, got asked "Just to confirm — book...?", said "yes", and
    ONLY THEN was told the limit was already reached — a whole confirmation
    round-trip wasted on a booking that could never succeed. The cap must be
    checked at the moment the slot is picked, before the confirming question is
    ever asked. See appointment_agent._booking_cap_reached."""
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.appointment_department_day_use import AppointmentDepartmentDayUse
    from app.models.slot import Slot

    other_appt_start = datetime.now(timezone.utc) + timedelta(days=1)
    other_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id, start_utc=other_appt_start, end_utc=other_appt_start + timedelta(minutes=30)
    )
    db.add(other_slot)
    db.flush()
    other_appointment = Appointment(
        clinic_id=clinic.id, slot_id=other_slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed"
    )
    db.add(other_appointment)
    db.flush()

    # The slot being picked THIS turn, same department and same local day as
    # the two uses already recorded below.
    picked_start = other_appt_start.replace(hour=14, minute=0, second=0, microsecond=0)
    picked_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id, start_utc=picked_start, end_utc=picked_start + timedelta(minutes=30)
    )
    db.add(picked_slot)
    db.flush()

    local_date = other_appt_start.date()
    db.add(AppointmentDepartmentDayUse(
        clinic_id=clinic.id, patient_id=patient.id, department_id=department.id,
        appointment_id=other_appointment.id, local_date=local_date,
    ))
    db.add(AppointmentDepartmentDayUse(
        clinic_id=clinic.id, patient_id=patient.id, department_id=department.id,
        appointment_id=other_appointment.id, local_date=local_date,
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called once the booking cap is already reached")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(
        db, ctx, f"I'd like to book the appointment. slot_id: {picked_slot.id}", "en", []
    )

    assert "Cardiology" in result
    assert "reached the limit" in result.lower()
    assert "just to confirm" not in result.lower()


def test_run_appointment_agent_refuses_a_reschedule_request_naming_a_different_day(
    monkeypatch, db, ctx, clinic, department, doctor, patient
):
    # Same "check before asking" principle again: a reschedule can only move to
    # a different TIME on the SAME day (see booking_engine.reschedule_
    # appointment's own hard rule) — if the patient explicitly names a
    # different day, they're told to cancel-and-rebook immediately, never shown
    # slots for that other day. See appointment_agent._reschedule_different_day_reply.
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    appt_start = datetime.now(timezone.utc) + timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=appt_start, end_utc=appt_start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    appointment = Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed")
    db.add(appointment)
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_department_availability must not be queried for a different day")

    monkeypatch.setattr(appointment_agent, "get_department_availability", _fail_if_called)

    # resolve_bare_weekday_window only recognizes an actual weekday NAME (not a
    # raw date) — pick whichever weekday name is guaranteed to resolve to a
    # date other than the appointment's own, by walking forward from the
    # appointment's own weekday.
    weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    different_weekday = weekday_names[(appt_start.weekday() + 1) % 7]

    result = appointment_agent.run_appointment_agent(
        db, ctx, f"reschedule my appointment to {different_weekday}", "en", []
    )

    assert "same day" in result.lower()
    assert "cancel" in result.lower()


def test_run_appointment_agent_refuses_a_reschedule_naming_an_explicit_different_date(
    monkeypatch, db, ctx, clinic, department, doctor, patient
):
    # Same same-day-only rule, a genuinely different gap: resolve_bare_weekday_window
    # only recognizes a bare WEEKDAY NAME — "reschedule to august 26th"/"wed 26 aug"
    # (an explicit calendar date, no weekday word at all) fell through that check
    # entirely and reached the general LLM/tool-calling path instead of the
    # deterministic same-day-only refusal.
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    appt_start = datetime.now(timezone.utc) + timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=appt_start, end_utc=appt_start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    db.add(Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed"))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_department_availability must not be queried for a different day")

    monkeypatch.setattr(appointment_agent, "get_department_availability", _fail_if_called)

    # A date guaranteed different from the appointment's own day (a week later,
    # never the same calendar day regardless of when the test runs).
    different_date = appt_start + timedelta(days=7)
    month_name = different_date.strftime("%B").lower()

    result = appointment_agent.run_appointment_agent(
        db, ctx, f"reschedule my appointment to {different_date.day} {month_name}", "en", []
    )

    assert "same day" in result.lower()
    assert "cancel" in result.lower()


def test_run_appointment_agent_refuses_a_reschedule_naming_a_different_day_vaguely(
    monkeypatch, db, ctx, clinic, department, doctor, patient
):
    # Third gap in the same-day-only check: "reschedule to a different day" names
    # no specific date or weekday at all — must still refuse deterministically,
    # since ANY different day violates the same-day-only rule regardless of which
    # one, rather than falling through to the LLM with nothing telling it this.
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    appt_start = datetime.now(timezone.utc) + timedelta(days=1)
    slot = Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=appt_start, end_utc=appt_start + timedelta(minutes=30))
    db.add(slot)
    db.flush()
    db.add(Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status="confirmed"))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("get_department_availability must not be queried for a vague different-day request")

    monkeypatch.setattr(appointment_agent, "get_department_availability", _fail_if_called)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "can i reschedule it to a different day instead", "en", []
    )

    assert "same day" in result.lower()
    assert "cancel" in result.lower()


def test_run_appointment_agent_generic_doctor_availability_question_falls_through_normally(
    monkeypatch, db, ctx, doctor
):
    # "is there a doctor available in Cardiology" names no SPECIFIC doctor — must
    # NOT be misread as an attempted (and failed) name match; falls through to the
    # normal agent loop exactly like before this fix.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["called"] = True
        return "some reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "is there a doctor available in Cardiology", "en", []
    )

    assert captured.get("called") is True
    assert result == "some reply"


def test_run_appointment_agent_renders_resolved_single_match_department_into_the_prompt(
    monkeypatch, db, ctx, doctor
):
    # "doctor" fixture is "Dr. Ahmed Khan" in Cardiology — an exact, unambiguous
    # single match must have its real department rendered into the prompt so the
    # model doesn't have to ask the patient which department this doctor is in.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    appointment_agent.run_appointment_agent(db, ctx, "I'd like to book with Dr. Ahmed Khan", "en", [])

    assert "RESOLVED DOCTOR" in captured["system_prompt"]
    assert "Dr. Ahmed Khan" in captured["system_prompt"]
    assert "Cardiology" in captured["system_prompt"]
    # Reported gap: a resolved single match went straight to showing slots with no
    # confirmation at all — must instruct a one-line confirm-first question instead.
    assert "ask ONE direct confirming question" in captured["system_prompt"]
    assert "do not check availability in the same turn you resolve the name" in captured["system_prompt"]


def test_run_appointment_agent_renders_previously_shown_card_into_the_prompt_for_the_handoff(
    monkeypatch, db, ctx
):
    # No doctor-name reference in this message at all ("the second one") — the
    # handoff relies entirely on the rendered PREVIOUSLY SHOWN OPTIONS block, not
    # the deterministic name-match step.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "booked"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "who's free in cardiology"),
        _row(
            "assistant",
            DEPARTMENT_LIST_MARKER
            + json.dumps({"departments": [{"department_name": "Cardiology", "doctors": [{"doctor_name": "Dr. Jane"}]}]}),
        ),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "the second one please", "en", history)

    assert result == "booked"
    assert "PREVIOUSLY SHOWN OPTIONS" in captured["system_prompt"]
    assert "Dr. Jane" in captured["system_prompt"]


def test_doctor_already_shown_true_for_a_doctor_department_pair_in_a_department_list_card():
    history = [
        _row(
            "assistant",
            DEPARTMENT_LIST_MARKER
            + json.dumps(
                {
                    "departments": [
                        {"department_name": "Orthopedics", "doctors": [{"doctor_name": "Dr. Junaid Mirza"}]},
                        {"department_name": "Dermatology", "doctors": [{"doctor_name": "Dr. Mariam Farooq"}]},
                    ]
                }
            ),
        ),
    ]
    assert appointment_agent._doctor_already_shown(history, "Dr. Junaid Mirza", "Orthopedics") is True
    # Right doctor, wrong department — must not false-positive.
    assert appointment_agent._doctor_already_shown(history, "Dr. Junaid Mirza", "Dermatology") is False
    assert appointment_agent._doctor_already_shown(history, "Dr. Someone Else", "Orthopedics") is False


def test_doctor_already_shown_false_with_empty_history():
    assert appointment_agent._doctor_already_shown([], "Dr. Junaid Mirza", "Orthopedics") is False


def test_doctor_already_shown_true_for_a_plain_prose_assistant_reply_naming_the_doctor():
    # Reported live: a confirming reply about one specific doctor doesn't always come
    # back as a structured DOCTOR_OPTIONS/DEPARTMENT_LIST card — sometimes it's just
    # plain prose (e.g. confirming "Dr Farhan Malik" by describing his hours). That
    # used to leave _doctor_already_shown with nothing to match, silently failing a
    # doctor-scoped follow-up later in the same conversation.
    history = [
        _row(
            "assistant",
            "Dr Farhan Malik - General Cardiology - available on Saturday and Sunday "
            "from 10:00 am to 6:00 pm, 15-minute slots.",
        ),
    ]
    assert appointment_agent._doctor_already_shown(history, "Dr. Farhan Malik", "Cardiology") is True
    # A doctor never mentioned in any assistant reply must still come back False.
    assert appointment_agent._doctor_already_shown(history, "Dr. Ahmed Farooq", "Cardiology") is False


def test_doctor_already_shown_ignores_a_doctor_named_only_in_a_pending_disambiguation_question():
    # A DOCTOR_DISAMBIGUATION_MARKER card is a QUESTION the assistant is still
    # waiting on an answer to ("did you mean X or Y?") — naming a doctor there is
    # not the same as having confirmed them, so it must not count as "already shown".
    history = [
        _row(
            "assistant",
            DOCTOR_DISAMBIGUATION_MARKER
            + json.dumps(
                {
                    "kind": "doctor_name",
                    "question": "Did you mean Dr. Iqra Qureshi or Dr. Iqra Raza?",
                    "candidates": [
                        {"doctor_name": "Dr. Iqra Qureshi", "department_name": "ENT"},
                        {"doctor_name": "Dr. Iqra Raza", "department_name": "ENT"},
                    ],
                }
            ),
        ),
    ]
    assert appointment_agent._doctor_already_shown(history, "Dr. Iqra Raza", "ENT") is False


def test_run_appointment_agent_skips_confirmation_when_doctor_already_shown_in_history(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: the assistant re-asked "did you mean Dr. Ahmed Khan in
    # Cardiology?" for a doctor it had itself already listed in a card two turns
    # earlier — reads as not having listened. Once the same doctor+department
    # already appeared in a real card in history, no confirming question is needed.
    # Reported live (2nd report): once no confirming question was needed, the next
    # step still fell through to the LLM, which called get_department_availability
    # for the whole department instead of just this one already-identified doctor
    # — naming one specific doctor by name always means just that doctor. Must now
    # return a real, filtered, single-doctor card built directly from the DB.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved once a single doctor is already resolved")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row(
            "assistant",
            DEPARTMENT_LIST_MARKER
            + json.dumps({"departments": [{"department_name": "Cardiology", "doctors": [{"doctor_name": "Dr. Ahmed Khan"}]}]}),
        ),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "Book with Dr. Ahmed Khan", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert [d["doctor_name"] for d in payload["doctors"]] == ["Dr. Ahmed Khan"]


def test_run_appointment_agent_resolves_doctor_from_a_plain_yes_confirming_a_prior_question(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: "book with dr raza ali" -> "Did you mean Dr. Ali Raza in
    # General Medicine?" -> patient replies "yes" -> the next reply asked "which
    # department does Dr. Raza Ali work in?", as if the confirmation never
    # happened. "yes" names no doctor itself, so the deterministic name-match on
    # the CURRENT message alone finds nothing — this confirms the fallback
    # recovers the doctor from the patient's own PRIOR message instead.
    # Reported live (2nd report): even once recovered, the "yes" still fell
    # through to the LLM, which called get_department_availability for the whole
    # department — the entire reason that confirming question exists is to check
    # THAT doctor's availability, so confirming it is never a request to broaden
    # back out to the department. Must now return a real, filtered, single-doctor
    # card built directly from the DB, never routed through the LLM at all.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved once a single doctor is confirmed")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i wants to book an appointment with dr ahmed khan"),
        _row("assistant", "Did you mean Dr. Ahmed Khan in Cardiology?"),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert [d["doctor_name"] for d in payload["doctors"]] == ["Dr. Ahmed Khan"]


def test_run_appointment_agent_resolves_doctor_from_yes_through_an_intermediate_pronoun_message(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: "hook me up with dr ali raza" (plain KB-prose reply, no
    # card) -> "can u book an appointment for me with him?" (named no doctor
    # itself, just "him" — resolved via the pronoun path from the FIRST
    # message) -> "Did you mean Dr. Ali Raza in General Medicine?" -> "yes".
    # The bare-"yes" recovery used to check only the single most recent user
    # message ("can u book...him?"), which also names no doctor — so it found
    # nothing, resolved_match stayed None, and the model got zero doctor
    # context, producing a generic "book it yourself online" answer instead of
    # offering to show real slots. Must now see past the intermediate
    # pronoun-only message to the one that actually named the doctor, and return
    # a real, filtered, single-doctor card — never the whole department.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    db.add(Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved once a single doctor is confirmed")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "hook me up with dr ahmed khan"),
        _row(
            "assistant",
            "Dr. Ahmed Khan sees patients for cardiology Monday through Thursday "
            "from 2:00 pm to 8:00 pm with 15-minute appointment slots.",
        ),
        _row("user", "can u book an appointment for me with him?"),
        _row("assistant", "Did you mean Dr. Ahmed Khan in Cardiology?"),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert [d["doctor_name"] for d in payload["doctors"]] == ["Dr. Ahmed Khan"]


def test_run_appointment_agent_plain_yes_with_no_preceding_question_does_not_force_a_match(
    monkeypatch, db, ctx, doctor
):
    # No pending confirming question to answer — "yes" must not spuriously resolve
    # a doctor from an unrelated earlier message.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "I'd like to book with Dr. Ahmed Khan"),
        _row("assistant", "Sure, what time works for you?"),
        _row("user", "anytime tomorrow"),
        _row("assistant", "Got it, checking now."),
    ]
    appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert "RESOLVED DOCTOR" not in captured["system_prompt"]


def test_run_appointment_agent_resolves_doctor_from_pronoun_referencing_prior_message(
    monkeypatch, db, ctx, doctor
):
    # Reported live: "hook me up with dr waqas" got a plain KB-prose reply (routed
    # to general_info_agent, no card shown) describing him as "otolaryngology" —
    # then "show me his available slots on fri" named no doctor at all, just the
    # pronoun "his", and wasn't a bare "yes" either. Neither existing recovery path
    # fired, leaving the LLM to guess a department_name from raw history — it
    # apparently echoed "otolaryngology" (a specialization, not the real department
    # name), and get_department_availability's exact-match lookup rejected it. This
    # confirms the doctor is now recovered deterministically from the patient's own
    # prior message via the pronoun.
    #
    # Reported live (2nd report): a further real conversation showed the assistant's
    # own prior reply (naming this exact doctor in plain prose, as below) wasn't
    # recognized as "already shown" at all — _doctor_already_shown only checked
    # structured DOCTOR_OPTIONS/DEPARTMENT_LIST cards, so a later pronoun follow-up
    # like this one fell all the way through to the doctor-blind LLM tool loop and
    # showed the WHOLE department instead of just this doctor. _doctor_already_shown
    # now also recognizes a free-form assistant reply naming the doctor, so this
    # resolves straight to a real, single-doctor answer — no redundant confirming
    # question, since the patient has already effectively been told who this is.
    monkeypatch.setattr(
        appointment_agent.llm,
        "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fall through to the LLM tool loop")),
    )

    history = [
        _row("user", "hook me up with dr ahmed khan"),
        _row(
            "assistant",
            "Dr. Ahmed Khan sees patients for cardiology Monday through Thursday "
            "from 2:00 pm to 8:00 pm with 15-minute appointment slots.",
        ),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "show me his available slots on fri", "en", history)

    # No slots ever seeded for this fixture doctor, so the deterministic
    # short-circuit's own "nothing open" reply is what comes back — the point
    # being asserted here is that it fired at all (real, resolved-to-one-doctor
    # deterministic code), not that it found an open slot.
    assert "Dr. Ahmed Khan" in result
    assert "doesn't have any open slots" in result


def test_run_appointment_agent_pronoun_with_no_prior_named_doctor_does_not_force_a_match(
    monkeypatch, db, ctx, doctor
):
    # No doctor named anywhere in recent history — "his" must not spuriously
    # resolve to an unrelated doctor.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "what are your opening hours"),
        _row("assistant", "We're open 9am to 5pm, Monday through Saturday."),
    ]
    appointment_agent.run_appointment_agent(db, ctx, "show me his available slots on fri", "en", history)

    assert "RESOLVED DOCTOR" not in captured["system_prompt"]


def test_run_appointment_agent_resolves_department_from_pronoun_referencing_prior_message(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "i want to see a dentist" got a plain KB-prose reply naming
    # the department as "General Dentistry" (prose text, not this clinic's real
    # department name, "Dentistry") — then "is there any doctor available
    # there?or just 1 doctor" referred back to it with "there", named no
    # department directly and no doctor either. The LLM was left to guess a
    # department_name from raw history and apparently echoed the KB prose's
    # wrong name, which get_department_availability's exact-match lookup
    # rejected ("I couldn't find a department called that"). Must now recover
    # the REAL department name deterministically and answer with a real,
    # current availability card — never routed through the LLM at all.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    dentistry = Department(clinic_id=clinic.id, name="Dentistry")
    db.add(dentistry)
    db.flush()
    qureshi = Doctor(
        clinic_id=clinic.id, department_id=dentistry.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Qureshi", is_active=True,
    )
    db.add(qureshi)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=qureshi.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved in resolving a department pronoun reference")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i want to see a dentist"),
        _row(
            "assistant",
            "You can book an appointment with Dr. Iqra Qureshi in the General Dentistry "
            "department. She sees patients on Monday, Wednesday and Friday from 10 am to "
            "6 pm, with 15-minute appointment slots.",
        ),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "is there any doctor available there?or just 1 doctor", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Dentistry"
    assert [d["doctor_name"] for d in payload["doctors"]] == ["Dr. Iqra Qureshi"]


def test_run_appointment_agent_department_pronoun_with_no_prior_named_department_falls_through_normally(
    monkeypatch, db, ctx, doctor
):
    # No department named anywhere in recent history — "there" must not
    # spuriously resolve to an unrelated department.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "what are your opening hours"),
        _row("assistant", "We're open 9am to 5pm, Monday through Saturday."),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "is there any doctor available there", "en", history
    )

    assert result == "reply"


def test_run_appointment_agent_answers_a_doctor_count_question_referencing_this_dept(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: "show me available slots for cardiology" showed a real
    # two-doctor card, then "are there only 2 doctors in this dept?" — "this
    # dept" is the same kind of back-reference as "there", just phrased
    # differently, and named no specific doctor. Fell through to the LLM with
    # only the (correct) previously-shown card as context; the system prompt
    # already tells the model to answer a genuinely separate question like this
    # via the card's own `note`, but it just re-showed the same card with no
    # explicit answer. Must now attach a real, code-computed doctor-count
    # answer, never routed through the LLM at all.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    other = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Farhan Malik", is_active=True,
    )
    db.add(other)
    db.flush()
    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=other.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved in answering a doctor-count question")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [_row("user", "show me available slots for cardiology")]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "are there only 2 doctors in this dept?", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["department_name"] == "Cardiology"
    assert len(payload["doctors"]) == 2
    assert payload["note"] == "There are currently 2 doctors in Cardiology:"


def test_run_appointment_agent_gives_doctor_info_without_slots_when_asked(
    monkeypatch, db, ctx, clinic, department, doctor
):
    """Reported live: "just give me general info about them not show slots" —
    asked right after a Cardiology DOCTOR_OPTIONS card was shown — got the exact
    same slots card shown back again instead of an answer, since
    appointment_agent's only tool (get_department_availability) always returns a
    full card with slots and has no way to omit them. Must answer with a real,
    code-computed doctor listing with NO slots, never routed through the LLM at
    all. Also exercises the "them" fallback path specifically: neither a
    department named directly in this message nor a "there" pronoun applies
    here, so it must recover Cardiology from the most recently shown card."""
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    other = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Farhan Malik", is_active=True,
    )
    db.add(other)
    db.flush()
    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=other.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved in answering a no-slots doctor-info question")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("assistant", DOCTOR_OPTIONS_MARKER + json.dumps({"department_name": "Cardiology", "doctors": []})),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "just give me general info about them not show slots", "en", history
    )

    assert not result.startswith(DOCTOR_OPTIONS_MARKER)
    assert doctor.full_name in result
    assert other.full_name in result
    assert "Cardiology" in result


def test_run_appointment_agent_pronoun_followup_stays_filtered_to_that_doctor_with_date(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: "is dr farhan malik available on sat?" -> confirmed -> shown
    # his real Saturday slots -> "is he available on mon?" — "he" correctly
    # resolved back to Dr. Farhan Malik, but this message has no "only"/"just"
    # narrowing word, so get_department_availability was called unfiltered for
    # the whole department on Monday and the reply silently showed a DIFFERENT
    # doctor's (Dr. Ahmed Farooq's) Monday slots instead of ever answering
    # whether Dr. Farhan Malik himself is free. Must stay filtered to the one
    # doctor "he" refers to, and clearly say he has nothing that day rather than
    # substituting someone else's availability.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    malik = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Farhan Malik", is_active=True,
    )
    db.add(malik)
    db.flush()

    forced_window = appointment_agent.resolve_bare_weekday_window("is he available on mon?")
    monday_date = datetime.fromisoformat(forced_window[0]).replace(tzinfo=timezone.utc)

    db.add_all([
        # Dr. Farhan Malik: a Saturday slot (already shown), nothing on Monday,
        # but a real slot further out so a "next available" answer is possible.
        Slot(
            clinic_id=clinic.id, doctor_id=malik.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=malik.id,
            start_utc=monday_date + timedelta(days=14),
            end_utc=monday_date + timedelta(days=14, minutes=30),
            status="open",
        ),
        # doctor (Dr. Ahmed Khan, from the fixture) DOES have a Monday slot —
        # this must never be substituted in for Dr. Farhan Malik's answer.
        Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=monday_date + timedelta(hours=9),
            end_utc=monday_date + timedelta(hours=9, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not reach the LLM for a pronoun follow-up about an already-resolved doctor")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    card = json.dumps(
        {"department_name": "Cardiology", "doctors": [{"doctor_name": "Dr. Farhan Malik", "specialization": None, "slots": []}]}
    )
    history = [
        _row("user", "is dr farhan malik available on sat?"),
        _row("assistant", "Did you mean Dr. Farhan Malik in Cardiology?"),
        _row("user", "yes"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + card),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "is he available on mon?", "en", history)

    assert "Dr. Ahmed Khan" not in result
    assert "Dr. Farhan Malik" in result
    assert "doesn't have any open slots" in result


def test_run_appointment_agent_answers_doctor_count_question_and_followup_shows_both(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live, full conversation: "how many cardiologist are there in this
    # clinic??" -> answered from static KB prose ("There are two cardiologists")
    # instead of real data -> "show me their information" -> only ONE doctor's
    # info came back (the KB document didn't happen to mention the second one),
    # with "No other cardiologists are mentioned in the available information."
    # Both turns must now be answered deterministically from the real DB,
    # showing BOTH doctors both times.
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    other = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Farhan Malik", is_active=True,
    )
    db.add(other)
    db.flush()
    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=other.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not reach the LLM for a real-DB doctor-count/listing question")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    # Turn 1.
    turn1 = appointment_agent.run_appointment_agent(
        db, ctx, "how many cardiologist are there in this clinic??", "en", []
    )
    assert turn1.startswith(DOCTOR_OPTIONS_MARKER)
    payload1 = json.loads(turn1[len(DOCTOR_OPTIONS_MARKER):])
    assert payload1["department_name"] == "Cardiology"
    assert {d["doctor_name"] for d in payload1["doctors"]} == {"Dr. Ahmed Khan", "Dr. Farhan Malik"}
    assert payload1["note"] == "There are currently 2 doctors in Cardiology:"

    # Turn 2: a pronoun follow-up ("their") to the real card turn 1 just showed.
    history = [
        _row("user", "how many cardiologist are there in this clinic??"),
        _row("assistant", turn1),
    ]
    turn2 = appointment_agent.run_appointment_agent(db, ctx, "show me their information", "en", history)
    assert turn2.startswith(DOCTOR_OPTIONS_MARKER)
    payload2 = json.loads(turn2[len(DOCTOR_OPTIONS_MARKER):])
    assert {d["doctor_name"] for d in payload2["doctors"]} == {"Dr. Ahmed Khan", "Dr. Farhan Malik"}


def test_run_appointment_agent_narrows_shown_card_by_doctor_and_time_of_day(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: "only show me available slots of dr farhan rehman after 12
    # pm on monday" and its follow-up "show me his available slots after 12
    # pm" both returned the same top-5-earliest-of-the-day slots (10:00 am
    # onward) — the time-of-day request was silently ignored entirely, since
    # there was no time-filtering mechanism anywhere in this deterministic
    # short-circuit path at all.
    from datetime import date, datetime, time, timedelta, timezone

    from app.models.doctor import Doctor
    from app.models.slot import Slot

    rehman = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Farhan Rehman", is_active=True,
    )
    db.add(rehman)
    db.flush()

    forced_window = appointment_agent.resolve_bare_weekday_window(
        "only show me available slots of dr farhan rehman after 12 pm on monday"
    )
    monday = date.fromisoformat(forced_window[0])

    def _make_slot(start_utc):
        slot = Slot(clinic_id=clinic.id, doctor_id=rehman.id, start_utc=start_utc, end_utc=start_utc + timedelta(minutes=30), status="open")
        db.add(slot)
        db.flush()
        return slot

    morning_slot = _make_slot(datetime.combine(monday, time(10, 0), tzinfo=timezone.utc))
    afternoon_slot = _make_slot(datetime.combine(monday, time(13, 0), tzinfo=timezone.utc))

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not reach the LLM for a deterministic doctor+time-of-day filter")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    card = json.dumps(
        {"department_name": "Cardiology", "doctors": [{"doctor_name": "Dr. Farhan Rehman", "specialization": None, "slots": []}]}
    )
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + card)]

    result = appointment_agent.run_appointment_agent(
        db, ctx, "only show me available slots of dr farhan rehman after 12 pm on monday", "en", history
    )

    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    slot_ids = {s["slot_id"] for s in payload["doctors"][0]["slots"]}
    assert slot_ids == {str(afternoon_slot.id)}
    assert str(morning_slot.id) not in slot_ids

    # Follow-up: "show me his available slots after 12 pm" (pronoun, no date
    # named) — must stay filtered to the afternoon slot regardless of phrasing.
    history2 = [
        _row("user", "only show me available slots of dr farhan rehman after 12 pm on monday"),
        _row("assistant", result),
    ]
    result2 = appointment_agent.run_appointment_agent(
        db, ctx, "show me his available slots after 12 pm", "en", history2
    )
    payload2 = json.loads(result2[len(DOCTOR_OPTIONS_MARKER):])
    slot_ids2 = {s["slot_id"] for s in payload2["doctors"][0]["slots"]}
    assert slot_ids2 == {str(afternoon_slot.id)}


def test_run_appointment_agent_narrows_shown_card_to_one_named_doctor_on_request(
    monkeypatch, db, ctx, clinic, department, doctor
):
    # Reported live: a DOCTOR_OPTIONS card showed two doctors (Dr. Ali Raza, Dr.
    # Farhan Rehman) in General Medicine. "only show me available slots for dr
    # farhan rehman" re-showed the exact same unfiltered two-doctor card —
    # get_department_availability has no doctor-filtering capability at all, so
    # even a correctly-resolved single doctor had nothing downstream to act on it.
    # Must now return a real, filtered, single-doctor card built directly from the
    # DB, never routed through the LLM at all.
    from datetime import datetime, timedelta, timezone

    from app.models.doctor import Doctor
    from app.models.slot import Slot

    other = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Farhan Rehman", is_active=True,
    )
    db.add(other)
    db.flush()
    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=doctor.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=other.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved in filtering an already-shown card")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    card = json.dumps(
        {
            "departments": [
                {
                    "department_name": "Cardiology",
                    "doctors": [
                        {"doctor_name": "Dr. Ahmed Khan", "specialization": None, "slots": []},
                        {"doctor_name": "Dr. Farhan Rehman", "specialization": None, "slots": []},
                    ],
                }
            ]
        }
    )
    history = [_row("assistant", DEPARTMENT_LIST_MARKER + card)]

    result = appointment_agent.run_appointment_agent(
        db, ctx, "only show me available slots for dr farhan rehman", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert [d["doctor_name"] for d in payload["doctors"]] == ["Dr. Farhan Rehman"]


def test_run_appointment_agent_book_with_specific_slot_falls_through_to_llm_instead_of_renarrowing(
    monkeypatch, db, ctx, clinic, department
):
    # Reported live: after a card showed Dr. Shahid Sheikh's Dermatology slots,
    # "book with dr shahid sheikh on sat aug 22 at 8 am" — an explicit booking
    # instruction naming the doctor, date, AND time — matched
    # direct_name_match_this_turn and fired the doctor-narrowing short-circuit,
    # which has no way to resolve a specific slot_id from natural language at
    # all: it just re-fetched and re-displayed the SAME availability card
    # instead of ever booking anything. The word "book" now steers this to the
    # LLM/tool-calling path instead — the one place PREVIOUSLY SHOWN OPTIONS
    # actually gets matched against the patient's wording to find the real
    # slot_id and call book_appointment.
    from app.models.doctor import Doctor

    sheikh = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Shahid Sheikh", is_active=True,
    )
    db.add(sheikh)
    db.flush()

    calls = []

    def _record_call(system_prompt, message, history, tools):
        calls.append(message)
        return "booked"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _record_call)

    card = json.dumps(
        {
            "department_name": "Cardiology",
            "doctors": [{"doctor_name": "Dr. Shahid Sheikh", "specialization": None, "slots": []}],
        }
    )
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + card)]

    result = appointment_agent.run_appointment_agent(
        db, ctx, "book with dr shahid sheikh on sat aug 22 at 8 am", "en", history
    )

    assert result == "booked"
    assert len(calls) == 1


def test_run_appointment_agent_narrowing_without_book_word_still_short_circuits(
    monkeypatch, db, ctx, clinic, department
):
    # Companion to the "book" fix above: a plain re-narrowing request with NO
    # booking verb (just re-asking for the same doctor's slots at a specific
    # time) must still hit the deterministic short-circuit as before — the fix
    # is scoped narrowly to messages that actually say "book", not a blanket
    # bypass of the whole narrowing mechanism.
    from datetime import date, datetime, time, timedelta, timezone

    from app.models.doctor import Doctor
    from app.models.slot import Slot

    sheikh = Doctor(
        clinic_id=clinic.id, department_id=department.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Shahid Sheikh", is_active=True,
    )
    db.add(sheikh)
    db.flush()

    forced_window = appointment_agent.resolve_bare_weekday_window("at 8 am on monday")
    monday = date.fromisoformat(forced_window[0]) if forced_window else (
        datetime.now(timezone.utc).date() + timedelta(days=1)
    )
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=sheikh.id,
        start_utc=datetime.combine(monday, time(8, 0), tzinfo=timezone.utc),
        end_utc=datetime.combine(monday, time(8, 30), tzinfo=timezone.utc),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("must not reach the LLM for a plain re-narrowing request with no booking verb")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    card = json.dumps(
        {
            "department_name": "Cardiology",
            "doctors": [{"doctor_name": "Dr. Shahid Sheikh", "specialization": None, "slots": []}],
        }
    )
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + card)]

    result = appointment_agent.run_appointment_agent(
        db, ctx, "show me available slots of dr shahid sheikh at 8 am", "en", history
    )

    assert result.startswith(DOCTOR_OPTIONS_MARKER)


def test_run_appointment_agent_narrowing_phrase_without_prior_card_falls_through_normally(
    monkeypatch, db, ctx, doctor
):
    # "only"/"just" with a doctor named for the first time (never shown in a card
    # yet) must not spuriously short-circuit — doctor_already_shown is False here.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "only show me available slots for dr ahmed khan", "en", []
    )

    assert result == "reply"
    assert "ask ONE direct confirming question" in captured["system_prompt"]


def test_run_appointment_agent_narrows_after_resolving_a_name_disambiguation_reply(
    monkeypatch, db, ctx, clinic
):
    # Reported live: "only show available slots for dr iqra" matched two doctors
    # (Dr. Iqra Qureshi, Dr. Iqra Raza) and correctly asked "did you mean X or Y?".
    # The patient's reply, "I mean Dr. Iqra Raza (ENT).", named no narrowing word
    # of its own — the ORIGINAL narrowing intent, stated in the message that
    # triggered the name disambiguation, was lost, and the full unfiltered
    # two-doctor card was shown again instead of a filtered single-doctor one.
    from app.models.department import Department
    from app.models.doctor import Doctor

    ent = Department(clinic_id=clinic.id, name="ENT")
    dentistry = Department(clinic_id=clinic.id, name="Dentistry")
    db.add_all([ent, dentistry])
    db.flush()
    iqra_raza = Doctor(
        clinic_id=clinic.id, department_id=ent.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Raza", is_active=True,
    )
    waqas = Doctor(
        clinic_id=clinic.id, department_id=ent.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Waqas Farooq", is_active=True,
    )
    iqra_qureshi = Doctor(
        clinic_id=clinic.id, department_id=dentistry.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Qureshi", is_active=True,
    )
    db.add_all([iqra_raza, waqas, iqra_qureshi])
    db.flush()

    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    db.add_all([
        Slot(
            clinic_id=clinic.id, doctor_id=iqra_raza.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
        Slot(
            clinic_id=clinic.id, doctor_id=waqas.id,
            start_utc=datetime.now(timezone.utc) + timedelta(days=1),
            end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
            status="open",
        ),
    ])
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be involved in filtering an already-shown card")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    card = json.dumps(
        {
            "note": "Based on what you've described, this sounds like something ENT should look at",
            "department_name": "ENT",
            "doctors": [
                {"doctor_name": "Dr. Iqra Raza", "specialization": "Head & Neck Surgery", "slots": []},
                {"doctor_name": "Dr. Waqas Farooq", "specialization": "Otolaryngology", "slots": []},
            ],
        }
    )

    # Step 1: the ambiguous-name request correctly asks which Iqra is meant.
    history = [_row("assistant", DOCTOR_OPTIONS_MARKER + card)]
    disambiguation = appointment_agent.run_appointment_agent(
        db, ctx, "only show available slots for dr iqra", "en", history
    )
    assert disambiguation.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    disambiguation_payload = json.loads(disambiguation[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert disambiguation_payload["kind"] == "doctor_name"

    # Step 2: answering which doctor is meant must still apply the original
    # narrowing request, returning a real, filtered single-doctor card.
    history = [
        _row("assistant", DOCTOR_OPTIONS_MARKER + card),
        _row("user", "only show available slots for dr iqra"),
        _row("assistant", disambiguation),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "I mean Dr. Iqra Raza (ENT).", "en", history)

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert [d["doctor_name"] for d in payload["doctors"]] == ["Dr. Iqra Raza"]


# --- action-intent negation (_detect_action_intent) ---------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I want to reschedule my appointment", "reschedule"),
        ("please cancel my appointment", "cancel"),
        ("reshedule my upcmoing appointment", "reschedule"),  # typo tolerance preserved
        # Reported live: a real conversation opened with "no no not reschedule but
        # book" — the negation sits right next to "reschedule", but the old plain
        # word-SET check had no concept of order/negation and matched it anyway.
        ("no no not reschedule but book", None),
        ("don't reschedule, just cancel it", "cancel"),
        ("never mind rescheduling", None),
        ("book with dr farhan malik at 11am", None),
    ],
)
def test_detect_action_intent_respects_local_negation(message, expected):
    assert appointment_agent._detect_action_intent(message) == expected


# --- new-booking intent supersedes a stale reschedule/cancel (_most_recent_action_intent) --


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("i want to book a new slot", True),
        ("book me another appointment", True),
        ("I'd like to book a separate appointment", True),
        ("can you book a different appointment for me", True),
        # A plain "book <time>" — the ordinary way a patient continues an
        # already-in-progress reschedule — must NOT count, or a live reschedule
        # would be wrongly dropped the moment the patient names a time for it.
        ("book on aug 21 at 8.30 am", False),
        ("book with dr farhan malik at 11am", False),
        # "new"/"another" alone, with no booking verb, isn't a booking statement at all.
        ("what's new with my appointment", False),
    ],
)
def test_detect_new_booking_supersede_intent(message, expected):
    assert appointment_agent._detect_new_booking_supersede_intent(message) == expected


def test_most_recent_action_intent_stops_at_a_superseding_new_booking_turn():
    # Reported live (see the full scenario replicated in
    # test_run_appointment_agent_new_booking_request_supersedes_a_stale_reschedule
    # below): "i wanna reschedule it" (turn 1) ... "i want to book a new slot"
    # (turn 2, a retraction with no reschedule/cancel keyword of its own) — the old
    # plain backward scan skipped straight past turn 2 (nothing to match) and
    # resurrected turn 1's stale "reschedule". The scan must stop at turn 2 instead.
    history = [
        _row("user", "i wanna reschedule it"),
        _row("assistant", "Here are the available slots..."),
        _row("user", "i want to book a new slot"),
        _row("assistant", "Here's what's available..."),
    ]
    assert appointment_agent._most_recent_action_intent(history) is None


def test_most_recent_action_intent_still_recovers_a_stale_action_with_no_intervening_supersede():
    # Unchanged behavior: a doctor-name-only reply a turn or two after a real
    # reschedule/cancel request must still recover that action when nothing in
    # between retracted it.
    history = [
        _row("user", "i wanna reschedule it"),
        _row("assistant", "Which doctor?"),
    ]
    assert appointment_agent._most_recent_action_intent(history) == "reschedule"


def test_run_appointment_agent_new_booking_request_supersedes_a_stale_reschedule(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Full reported scenario: patient has a confirmed appointment with `doctor`,
    # asks to reschedule it (slots shown), then changes their mind mid-flow and
    # says they want to book a NEW, separate slot instead (keeping the original),
    # then names the doctor again with a specific time. This must be treated as a
    # fresh booking request — never silently folded back into the original
    # reschedule (previously produced a "Just to confirm — reschedule..." reply
    # for what the patient explicitly asked to be a new booking).
    _future_appointment(db, clinic, patient, doctor, days_from_now=1)

    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "Here are the available slots for a new booking."

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "i wanna reschedule it"),
        _row("assistant", "Here are the available slots..."),
        _row("user", "book on aug 21 at 8.30 am"),
        _row("assistant", "Would you like to change it to the 8:30 AM slot? Reply yes or no."),
        _row("user", "i want to book a new slot"),
        _row("assistant", "Here's what's available..."),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, f"I'd like to book the appointment with {doctor.full_name} at 8:30 AM.", "en", history
    )

    assert "reschedule" not in result.lower()
    # The system prompt built for the LLM must not carry _build_system_prompt's
    # RESOLVED APPOINTMENT / "This is a RESCHEDULE:" block — that's what forces
    # book_appointment calls to be silently redirected into reschedule_appointment
    # (see build_tools' reschedule_redirect_appointment_id in chat_tools.py). Checked
    # via these exact marker strings (not a blanket "reschedule" substring check) —
    # the prompt's general tool-usage rules mention reschedule_appointment by name
    # unconditionally, so only the RESOLVED-appointment block's own distinctive
    # wording actually proves the stale action didn't get resolved.
    system_prompt = captured.get("system_prompt", "")
    assert "RESOLVED APPOINTMENT" not in system_prompt
    assert "This is a RESCHEDULE:" not in system_prompt


def test_run_appointment_agent_new_booking_request_supersedes_a_stale_cancel_disambiguation(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Reported live: "cancel both of my appointments" (2 active, same doctor) asked
    # "which one would you like to cancel: 9:00 AM, 9:30 AM?" — then "i want to book
    # a new appointment" just repeated the exact same cancel question verbatim,
    # since _match_candidate found no match (the reply names neither time) and the
    # old code always re-asked on a non-match with no way to recognize the topic
    # had changed. Must fall through to a fresh (non-cancel) request instead.
    _future_appointment(db, clinic, patient, doctor, days_from_now=1)
    _future_appointment(db, clinic, patient, doctor, days_from_now=2)

    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: "Here are the available slots for a new booking.",
    )

    disambiguation = appointment_agent.run_appointment_agent(db, ctx, "cancel both of my appointments", "en", [])
    assert disambiguation.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(disambiguation[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "appointment"
    assert payload["action"] == "cancel"

    history = [
        _row("user", "cancel both of my appointments"),
        _row("assistant", disambiguation),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "i want to book a new appointment", "en", history)

    assert "which one would you like to cancel" not in result.lower()
    assert not (result.startswith(DOCTOR_DISAMBIGUATION_MARKER) and '"action": "cancel"' in result)


def test_run_appointment_agent_different_action_supersedes_a_stale_cancel_disambiguation(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Same live report, second reply tried: "reschedule my upcoming appointments"
    # (a genuinely different action, not just a non-answer) also just repeated the
    # stale CANCEL question. A reply naming a different action than the one being
    # disambiguated must re-disambiguate for THAT action, not parrot the old one.
    _future_appointment(db, clinic, patient, doctor, days_from_now=1)
    _future_appointment(db, clinic, patient, doctor, days_from_now=2)

    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    disambiguation = appointment_agent.run_appointment_agent(db, ctx, "cancel both of my appointments", "en", [])
    history = [
        _row("user", "cancel both of my appointments"),
        _row("assistant", disambiguation),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "reschedule my upcoming appointments", "en", history)

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "appointment"
    assert payload["action"] == "reschedule"
    assert "which one would you like to reschedule" in payload["question"].lower()


def test_run_appointment_agent_reschedule_with_zero_active_appointments_and_no_doctor_named(
    monkeypatch, db, ctx, patient
):
    # Reported live: a patient was shown Cardiology availability by symptom_agent
    # for a chest-pain complaint (never booked anything — just a triage card), then
    # said "i want to reschedule my appointment" (no doctor named, no real active
    # appointment to reschedule). This used to fall through every branch with no
    # deterministic reply, reach the LLM primed only with the stale, symptom-agent-
    # authored Cardiology card as context, and come back with a diagnosis_guard
    # redirect ("I'm not able to diagnose conditions...") — a bizarre non-answer to
    # a plain reschedule request. Must now short-circuit deterministically instead,
    # same as the already-handled named-doctor-with-no-appointment case.
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    # The stale symptom-agent card from triage — present in history exactly as it
    # would be live, to prove it's no longer treated as relevant appointment context.
    stale_card = DOCTOR_OPTIONS_MARKER + json.dumps({
        "department_name": "Cardiology",
        "note": "Based on what you've described, this mild chest pain related to physical "
        "activity could be evaluated by Cardiology.",
        "doctors": [{"doctor_name": "Dr. Ahmed Farooq", "slots": []}],
    })
    history = [
        _row("user", "i have mild chest pain during physical activity"),
        _row("assistant", stale_card),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "i want to reschedule my appointment", "en", history)

    assert result == "You don't have an upcoming appointment to reschedule."


def test_run_appointment_agent_reschedule_slot_card_has_a_headline_note(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Instructed live: the reschedule slot-pick card used to render with no note
    # at all — a bare list of times with nothing telling the patient what picking
    # one of them actually does, unlike every symptom-triage card elsewhere in
    # this module. Must now name the doctor and say to pick a time to reschedule.
    from datetime import timedelta

    from app.models.slot import Slot

    appt = _future_appointment(db, clinic, patient, doctor, days_from_now=1)
    old_slot = db.get(Slot, appt.slot_id)
    # Reschedule only ever offers same-day times — shifted a couple hours,
    # same calendar day as the appointment's own slot, never a different day.
    shift = timedelta(hours=-2) if old_slot.start_utc.hour >= 20 else timedelta(hours=2)
    new_start = old_slot.start_utc + shift
    db.add(Slot(clinic_id=clinic.id, doctor_id=doctor.id, start_utc=new_start, end_utc=new_start + timedelta(minutes=30), status="open"))
    db.flush()

    result = appointment_agent.run_appointment_agent(db, ctx, "i want to reschedule my appointment", "en", [])

    assert result.startswith(DOCTOR_OPTIONS_MARKER)
    payload = json.loads(result[len(DOCTOR_OPTIONS_MARKER):])
    assert payload["note"] == f"Select a time below to reschedule your appointment with {doctor.full_name}."


def test_run_appointment_agent_cancel_with_zero_active_appointments_and_no_doctor_named(
    monkeypatch, db, ctx, patient
):
    # Same gap, cancel side.
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    result = appointment_agent.run_appointment_agent(db, ctx, "i want to cancel my appointment", "en", [])

    assert result == "You don't have an upcoming appointment to cancel."


def test_run_appointment_agent_bare_yes_after_a_stale_confirmation_cannot_book(
    monkeypatch, db, clinic, doctor, ctx
):
    # Full reported scenario: "book with Dr. X at 3pm" -> confirm question ->
    # "what are clinic opening hours?" (correctly answered, unrelated) -> "Yeah"
    # (correctly generic, not booked) -> "Yes" — booked for real live, from a
    # confirmation question that was two turns stale by then. The tools list
    # appointment_agent hands to the LLM for this final "Yes" must have
    # book_appointment gated shut (see build_tools' suppress_bare_confirmation_booking),
    # since none of this turn's own deterministic checks resolved anything live —
    # this is a bare yes/no answering nothing.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["tools"] = tools
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", f"Book with {doctor.full_name} at Mon, Aug 24 at 3:00 PM"),
        _row(
            "assistant",
            f"Just to confirm — book your appointment with {doctor.full_name} in Cardiology on "
            "Mon, Aug 24 at 3:00 PM?",
        ),
        _row("user", "What are clinic opening hours?"),
        _row("assistant", "The clinic is open every day of the week, 8:00 am to 9:00 pm."),
        _row("user", "Yeah"),
        _row(
            "assistant",
            "Glad to hear that! Let me know if you'd like to book, reschedule, or cancel an "
            "appointment. I'm here to help.",
        ),
    ]
    appointment_agent.run_appointment_agent(db, ctx, "Yes", "en", history)

    book_tool = next(t for t in captured["tools"] if t.name == "book_appointment")
    result = book_tool.invoke({"slot_id": str(uuid.uuid4())})
    assert "booked" not in result.lower()


@pytest.mark.parametrize("message", ["do it", "go ahead", "please do", "confirm it", "yes", "sure thing"])
def test_run_appointment_agent_any_confirming_phrase_after_a_stale_confirmation_cannot_book(
    monkeypatch, db, clinic, doctor, ctx, message
):
    # Second live report on the same bug: "do it" reached the exact same path as
    # "Yes" above and booked anyway — _is_short_affirmative_reply/
    # _is_short_negative_reply don't recognize "do it"/"go ahead"/"please do"/
    # "confirm it" at all, so the FIRST fix's flag (narrowly keyed on yes/no
    # wording) never activated for them. The real fix is keyed on the absence of
    # anything resolved this turn, not on matching a fixed list of confirming
    # phrases — must hold for any wording, not just yes/no.
    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        return next(t for t in tools if t.name == "book_appointment").invoke({"slot_id": str(uuid.uuid4())})

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", f"Book with {doctor.full_name} at Sun, Aug 23 at 8:00 AM"),
        _row(
            "assistant",
            f"Just to confirm — book your appointment with {doctor.full_name} in Dermatology on "
            "Sun, Aug 23 at 8:00 AM?",
        ),
        _row("user", "i am having pain in my leg"),
        _row(
            "assistant",
            "Could you tell me how severe this is (mild, moderate, or severe) and how long you've had it?",
        ),
        _row("user", "where is this clinic located"),
        _row("assistant", "The clinic is located at 123 Main Boulevard, Gulberg III, Lahore, Punjab."),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, message, "en", history)

    assert "appointment confirmed" not in result.lower()
    assert "booked" not in result.lower()


def test_run_appointment_agent_does_not_treat_a_retracted_reschedule_as_a_live_action(
    monkeypatch, db, ctx, doctor
):
    # Reported live: the patient's very first message ("no no not reschedule but
    # book") was misread as a reschedule request despite explicitly retracting it.
    # Later, tapping a slot for Dr. Ahmed Khan (who has no active appointment) named
    # exactly one doctor with no action word of its own, so _most_recent_action_intent
    # scanned back through recent turns and picked the stale, retracted "reschedule"
    # back up — replying "You don't have an upcoming appointment... to reschedule"
    # to what was actually a brand new booking request. The fix makes
    # _detect_action_intent itself negation-aware, so the retracted mention is never
    # even a candidate for that lookback to find.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["called"] = True
        return "showing slots"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "no no not reschedule but book"),
        _row("assistant", "Sure! Which department or doctor would you like to see?"),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "I'd like to book the appointment with Dr. Ahmed Khan at 11:00 AM.", "en", history
    )

    assert "to reschedule" not in result
    assert "don't have an upcoming appointment" not in result
    assert captured.get("called") is True


# --- appointment-ambiguity handoff (cancel/reschedule against real appointments) ----


def _future_appointment(db, clinic, patient, doctor, days_from_now=1, status="confirmed"):
    from datetime import datetime, timedelta, timezone

    from app.models.appointment import Appointment
    from app.models.slot import Slot

    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=days_from_now),
        end_utc=datetime.now(timezone.utc) + timedelta(days=days_from_now, minutes=30),
        status="booked",
    )
    db.add(slot)
    db.flush()
    appt = Appointment(clinic_id=clinic.id, slot_id=slot.id, patient_id=patient.id, doctor_id=doctor.id, status=status)
    db.add(appt)
    db.flush()
    return appt


def test_run_appointment_agent_asks_confirmation_before_cancelling_when_exactly_one_active_appointment(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Reported live: "show me my upcoming appointment" (one shown), then "can i
    # cancel that?" — a QUESTION, not a command — cancelled the appointment
    # outright with zero confirmation. Cancel must always ask first, deterministically,
    # and never reach the LLM/tool layer until the patient explicitly confirms.
    appt = _future_appointment(db, clinic, patient, doctor)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called before the patient confirms a cancel")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(db, ctx, "can i cancel that?", "en", [])

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "cancel_confirm"
    assert payload["candidate"]["appointment_id"] == str(appt.id)


def test_run_appointment_agent_asks_which_action_when_message_names_both_cancel_and_reschedule(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Reported live: "cancel my appointment or reschedule" silently ALWAYS
    # resolved to cancel — _detect_action_intent checks every token for a
    # cancel keyword before ever considering reschedule, so the "or
    # reschedule" alternative was dropped entirely with zero disambiguation,
    # regardless of which action the patient named first in the sentence.
    _future_appointment(db, clinic, patient, doctor)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called while the action is still ambiguous")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    for message in [
        "cancel my appointment or reschedule",
        "reschedule or cancel my appointment",
        "should i cancel or reschedule my appointment?",
    ]:
        result = appointment_agent.run_appointment_agent(db, ctx, message, "en", [])
        assert "cancel" in result.lower() and "reschedule" in result.lower()
        assert not result.startswith(DOCTOR_DISAMBIGUATION_MARKER)


def test_run_appointment_agent_phrases_confirmation_as_an_answer_to_a_capability_question(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    # Reported live: the confirm-before-act gate itself was correct, but "can i
    # also cancel my appointment?" got the same command-toned "Just to confirm —
    # you'd like to cancel..." wording as a direct "cancel my appointment", which
    # reads as ignoring the question being asked. Same safety gate, different
    # wording only when the message is phrased as a capability question.
    appt = _future_appointment(db, clinic, patient, doctor)
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM")),
    )

    result = appointment_agent.run_appointment_agent(db, ctx, "can i also cancel my appointment?", "en", [])

    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "cancel_confirm"
    assert payload["question"].startswith("Yes, you can")
    assert "Just to confirm" not in payload["question"]

    # A direct command still gets the original "Just to confirm" wording.
    direct_result = appointment_agent.run_appointment_agent(db, ctx, "cancel my appointment", "en", [])
    direct_payload = json.loads(direct_result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert direct_payload["question"].startswith("Just to confirm")

    # "yes" after the capability-question wording still actually cancels — the
    # underlying gate/marker mechanism is unchanged.
    history = [_row("assistant", result)]
    cancel_result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)
    assert "has been cancelled" in cancel_result
    db.refresh(appt)
    assert appt.status == "cancelled"


def test_run_appointment_agent_cancels_only_after_explicit_yes_to_confirmation(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    appt = _future_appointment(db, clinic, patient, doctor)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must never be involved in the actual cancel action")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "appointment_id": str(appt.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "cancel_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert "has been cancelled" in result
    db.refresh(appt)
    assert appt.status == "cancelled"


def test_run_appointment_agent_declining_confirmation_leaves_appointment_untouched(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    appt = _future_appointment(db, clinic, patient, doctor)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called when the patient declines the cancel")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "appointment_id": str(appt.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "cancel_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(db, ctx, "no", "en", history)

    assert "was not cancelled" in result
    db.refresh(appt)
    assert appt.status == "confirmed"


# --- book/reschedule confirmation (instructed live: same code-enforced confirm-
# then-act gate as cancel, applied to booking and rescheduling too) ---


def _future_slot(db, clinic, doctor, days_from_now=2):
    from datetime import datetime, timedelta, timezone

    from app.models.slot import Slot

    slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=days_from_now),
        end_utc=datetime.now(timezone.utc) + timedelta(days=days_from_now, minutes=30),
        status="open",
    )
    db.add(slot)
    db.flush()
    return slot


def test_run_appointment_agent_asks_to_confirm_before_booking_a_fresh_slot_pick(
    monkeypatch, db, ctx, clinic, doctor
):
    slot = _future_slot(db, clinic, doctor)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not book directly off a slot pick — must ask to confirm first")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    message = f"I'd like to book the appointment with Dr. Ahmed Khan at Mon, Aug 10 at 9:00 AM (slot_id: {slot.id})."
    result = appointment_agent.run_appointment_agent(db, ctx, message, "en", [])

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "book_confirm"
    assert payload["candidate"]["slot_id"] == str(slot.id)
    assert payload["candidate"]["doctor_name"] == "Dr. Ahmed Khan"
    assert payload["candidate"]["department_name"] == "Cardiology"

    db.refresh(slot)
    assert slot.status == "open"


def test_run_appointment_agent_books_only_after_explicit_yes_to_book_confirmation(
    monkeypatch, db, ctx, clinic, doctor
):
    slot = _future_slot(db, clinic, doctor)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must never be involved in the actual booking action")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "slot_id": str(slot.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "book_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert "confirmed" in result.lower()
    db.refresh(slot)
    assert slot.status == "booked"


def test_run_appointment_agent_declining_book_confirmation_leaves_slot_open(
    monkeypatch, db, ctx, clinic, doctor
):
    slot = _future_slot(db, clinic, doctor)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called when the patient declines the booking")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "slot_id": str(slot.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "book_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(db, ctx, "no", "en", history)

    assert "not booked" in result


# --- _is_short_negative_reply ---------------------------------------------------
# Reported live: "no,book me with dr waqas on mon aug 24 at 3.30 instead" — a
# decline plus a real new request in one message — matched the plain "no" prefix
# check and got treated as a flat decline, silently discarding "book me ... 3.30
# instead" entirely.


@pytest.mark.parametrize(
    "message",
    [
        "no",
        "no thanks",
        "nope, thanks",
        "never mind",
        "do not book that",
        "don't book it",
        "cancel that",
    ],
)
def test_is_short_negative_reply_true_for_pure_declines(message):
    assert appointment_agent._is_short_negative_reply(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "no,book me with dr waqas on mon aug 24 at 3.30 instead",
        "nope, I want dr waqas at 4pm instead",
        "no not that one, book the friday slot instead",
    ],
)
def test_is_short_negative_reply_false_when_a_new_request_follows(message):
    assert appointment_agent._is_short_negative_reply(message) is False


def test_run_appointment_agent_declining_book_confirmation_with_a_new_request_does_not_swallow_it(
    monkeypatch, db, ctx, clinic, doctor
):
    slot = _future_slot(db, clinic, doctor)
    db.commit()

    candidate = {
        "slot_id": str(slot.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "book_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(
        db, ctx, "no, book me with dr ahmed khan on mon instead", "en", history
    )

    # The candidate must not be silently declined with the canned message — the
    # trailing new request has to actually be looked at (falls through to normal
    # handling, which resolves the named doctor rather than returning the flat
    # "not booked" sentence for the original slot).
    assert "was not booked" not in result
    db.refresh(slot)
    assert slot.status == "open"
    db.refresh(slot)
    assert slot.status == "open"


def test_run_appointment_agent_asks_to_confirm_before_rescheduling_to_a_picked_slot(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    appt = _future_appointment(db, clinic, patient, doctor, days_from_now=1)
    new_slot = _future_slot(db, clinic, doctor, days_from_now=3)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not reschedule directly off a slot pick — must ask to confirm first")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "appointment_id": str(appt.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending_disambiguation = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "appointment", "action": "reschedule", "candidates": [candidate]}
    )
    history = [
        _row("user", "reschedule my appointment"),
        _row("assistant", pending_disambiguation),
        _row("user", "the one with Dr. Ahmed Khan"),
        _row(
            "assistant",
            DOCTOR_OPTIONS_MARKER
            + json.dumps(
                {
                    "department_name": "Cardiology",
                    "doctors": [
                        {
                            "doctor_id": str(doctor.id),
                            "doctor_name": "Dr. Ahmed Khan",
                            "specialization": None,
                            "slots": [{"slot_id": str(new_slot.id), "when": "Wed, Aug 12 at 9:00 AM"}],
                        }
                    ],
                }
            ),
        ),
    ]

    message = f"I'd like to book the appointment with Dr. Ahmed Khan at Wed, Aug 12 at 9:00 AM (slot_id: {new_slot.id})."
    result = appointment_agent.run_appointment_agent(db, ctx, message, "en", history)

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "reschedule_confirm"
    assert payload["candidate"]["appointment_id"] == str(appt.id)
    assert payload["candidate"]["new_slot_id"] == str(new_slot.id)

    db.refresh(appt)
    assert appt.slot_id != new_slot.id


def test_run_appointment_agent_reschedules_only_after_explicit_yes_to_reschedule_confirmation(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    from datetime import timedelta

    from app.models.slot import Slot

    appt = _future_appointment(db, clinic, patient, doctor, days_from_now=1)
    # A reschedule may only move to a different TIME on the SAME day (see
    # booking_engine.reschedule_appointment) — a couple hours off, same day as
    # appt's own slot, not _future_slot's default cross-day offset. Shifted
    # earlier or later depending on the hour so the +/- 2h never crosses a
    # calendar-day boundary regardless of what time "now" happens to be.
    old_slot = db.get(Slot, appt.slot_id)
    shift = timedelta(hours=-2) if old_slot.start_utc.hour >= 20 else timedelta(hours=2)
    new_start = old_slot.start_utc + shift
    new_slot = Slot(
        clinic_id=clinic.id, doctor_id=doctor.id,
        start_utc=new_start,
        end_utc=new_start + timedelta(minutes=30),
        status="open",
    )
    db.add(new_slot)
    db.flush()
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must never be involved in the actual reschedule action")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "appointment_id": str(appt.id),
        "new_slot_id": str(new_slot.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Wed, Aug 12 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "reschedule_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert "rescheduled" in result.lower()
    db.refresh(appt)
    assert appt.slot_id == new_slot.id


def test_run_appointment_agent_declining_reschedule_confirmation_leaves_appointment_untouched(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    appt = _future_appointment(db, clinic, patient, doctor, days_from_now=1)
    original_slot_id = appt.slot_id
    new_slot = _future_slot(db, clinic, doctor, days_from_now=3)
    db.commit()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called when the patient declines the reschedule")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    candidate = {
        "appointment_id": str(appt.id),
        "new_slot_id": str(new_slot.id),
        "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology",
        "when": "Wed, Aug 12 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "reschedule_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    result = appointment_agent.run_appointment_agent(db, ctx, "no", "en", history)

    assert "not rescheduled" in result
    db.refresh(appt)
    assert appt.slot_id == original_slot_id


def test_run_appointment_agent_short_circuits_asking_which_when_multiple_active_and_no_doctor_named(
    monkeypatch, db, ctx, clinic, doctor, other_doctor, patient
):
    # Reported live: with 2 real upcoming appointments, "cancel my upcoming
    # appointments" silently cancelled the most recently booked one instead of
    # asking which — must now short-circuit with a real disambiguation, no LLM call.
    _future_appointment(db, clinic, patient, doctor)
    _future_appointment(db, clinic, patient, other_doctor)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when appointment is genuinely ambiguous")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(db, ctx, "cancel my upcoming appointments", "en", [])

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "appointment"
    assert payload["action"] == "cancel"
    names = {c["doctor_name"] for c in payload["candidates"]}
    assert names == {"Dr. Ahmed Khan", "Dr. Ahmed Raza"}


def test_run_appointment_agent_narrows_multiple_active_via_named_doctor_in_same_message(
    monkeypatch, db, ctx, clinic, doctor, other_doctor, patient
):
    _future_appointment(db, clinic, patient, doctor)
    other_appt = _future_appointment(db, clinic, patient, other_doctor)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called before the patient confirms a cancel")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "cancel my appointment with Dr. Ahmed Raza", "en", []
    )

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "cancel_confirm"
    assert payload["candidate"]["appointment_id"] == str(other_appt.id)


def test_run_appointment_agent_named_doctor_with_no_active_appointment_never_falls_back_to_another_one(
    monkeypatch, db, ctx, clinic, doctor, other_doctor, patient
):
    # Reported live: patient had 2 upcoming appointments (Dr. Ahmed Khan, Dr. Ahmed
    # Raza), cancelled Dr. Ahmed Khan's via the disambiguation flow, then repeated
    # "I mean Dr. Ahmed Khan." again (no "cancel" keyword this time, and the last
    # assistant turn was a plain cancellation-confirmation sentence, not a
    # DOCTOR_DISAMBIGUATION_MARKER card) — the old `elif len(active) == 1` branch
    # picked the one appointment left (Dr. Ahmed Raza's) blindly, ignoring that the
    # named doctor was Khan, and silently cancelled the WRONG one. Only Dr. Ahmed
    # Raza's appointment exists here (Khan's was already cancelled/never booked);
    # naming Khan again must get a clear "you don't have one" reply, never resolve
    # to Raza's appointment, and never even reach the LLM.
    _future_appointment(db, clinic, patient, other_doctor)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the named doctor has no active appointment")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "cancel my appointment"),
        _row(
            "assistant",
            "You have more than one upcoming appointment — which one would you like to cancel: "
            "Dr. Ahmed Khan (Cardiology, Thu, Aug 6 at 4:00 PM), Dr. Ahmed Raza (Neurology, Sat, Aug 8 at 8:00 AM)?",
        ),
        _row("user", "I mean Dr. Ahmed Khan."),
        _row("assistant", "Your appointment with Dr. Ahmed Khan in Cardiology on Thu, Aug 6 at 4:00 PM has been cancelled."),
    ]
    result = appointment_agent.run_appointment_agent(db, ctx, "I mean Dr. Ahmed Khan.", "en", history)

    assert result == "You don't have an upcoming appointment with Dr. Ahmed Khan to cancel."


def test_run_appointment_agent_resolves_a_reply_to_pending_appointment_disambiguation(
    monkeypatch, db, ctx, clinic, doctor, other_doctor, patient
):
    _future_appointment(db, clinic, patient, doctor)
    other_appt = _future_appointment(db, clinic, patient, other_doctor)
    candidates = [
        {"appointment_id": "wrong-would-be-a-bug", "doctor_name": "Dr. Ahmed Khan", "department_name": "Cardiology", "when": "Mon"},
        {"appointment_id": str(other_appt.id), "doctor_name": "Dr. Ahmed Raza", "department_name": "Neurology", "when": "Tue"},
    ]
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "appointment", "action": "cancel", "question": "which one?", "candidates": candidates}
    )
    history = [_row("assistant", pending)]

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called before the patient confirms a cancel")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(db, ctx, "Dr. Ahmed Raza", "en", history)

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    # Resolves to the candidate the reply actually matched (Dr. Ahmed Raza's real
    # id), not the other, non-matching candidate in the same payload — then asks a
    # fresh confirm-before-cancel question rather than acting immediately.
    assert payload["kind"] == "cancel_confirm"
    assert payload["candidate"]["appointment_id"] == str(other_appt.id)
    assert "wrong-would-be-a-bug" not in result


def test_run_appointment_agent_reasks_when_reply_to_pending_disambiguation_matches_nothing(
    monkeypatch, db, ctx
):
    candidates = [
        {"appointment_id": "id-1", "doctor_name": "Dr. Ahmed Khan", "department_name": "Cardiology", "when": "Mon"},
        {"appointment_id": "id-2", "doctor_name": "Dr. Ahmed Raza", "department_name": "Neurology", "when": "Tue"},
    ]
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "appointment", "action": "cancel", "question": "which one?", "candidates": candidates}
    )
    history = [_row("assistant", pending)]

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the reply still doesn't match a candidate")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = appointment_agent.run_appointment_agent(db, ctx, "the first appointment please", "en", history)

    assert result.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    payload = json.loads(result[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert payload["kind"] == "appointment"
    assert len(payload["candidates"]) == 2


def test_run_appointment_agent_warns_when_named_department_contradicts_session_symptoms(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live: patient described fever + body aches (routed correctly to
    # General Medicine), then said "isn't neurologist a better idea" / directly asked
    # to book with a department their own symptoms don't support — appointment_agent
    # blindly showed that department's availability with zero reasoning. This
    # confirms the deterministic mismatch check fires BEFORE the LLM ever runs.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(gen_med)
    neuro = Department(clinic_id=clinic.id, name="Neurology")
    db.add(neuro)
    db.flush()
    gen_med_doctor = Doctor(
        clinic_id=clinic.id,
        department_id=gen_med.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza",
        is_active=True,
    )
    db.add(gen_med_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=gen_med_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the named department contradicts symptoms")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    card = json.dumps(
        {
            "note": "Patient reports mild fever and body aches.",
            "department_name": "General Medicine",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. Ali Raza", "specialization": None, "slots": []}],
        }
    )
    history = [
        _row("user", "I am having a bit of fever and body aches"),
        _row("assistant", "How long have you had the fever and body aches?"),
        _row("user", "from past 2 years"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + card),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "book appointment in Neurology", "en", history
    )

    assert "General Medicine" in result
    assert "Neurology" in result
    assert result != DOCTOR_OPTIONS_MARKER + card


def test_run_appointment_agent_warns_when_general_doc_names_a_mismatched_department(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live: after describing severe leg pain (an Orthopedics-hinting
    # symptom), "can u refer me to general doc" went straight to a General
    # Medicine availability card with zero mention of the earlier symptoms —
    # "general doc" wasn't in _DEPARTMENT_TITLE_HINTS at all (only "general
    # physician"/"general practitioner" were), so _departments_named_directly_in_
    # message found nothing to warn about and the mismatch check never ran. See
    # the "general doc"/"gp" additions to DEPARTMENT_TITLE_HINTS in
    # message_classifier.py.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    ortho = Department(clinic_id=clinic.id, name="Orthopedics")
    db.add(ortho)
    general = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(general)
    db.flush()
    general_doctor = Doctor(
        clinic_id=clinic.id, department_id=general.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Ali Raza", is_active=True,
    )
    db.add(general_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=general_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the named department contradicts symptoms")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    # The card the symptom-triage agent's own reasoning actually showed for the
    # leg pain — real ground truth the fallback check compares against, since
    # the keyword-hint table alone (bare "leg"+"pain", no explicit injury verb)
    # would otherwise guess General Medicine here too and see no contradiction.
    ortho_card = json.dumps(
        {
            "note": "This sounds like something Orthopedics should look at.",
            "department_name": "Orthopedics",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. Junaid Mirza", "specialization": None, "slots": []}],
        }
    )
    history = [
        _row("user", "i have severe pain in my leg"),
        _row("assistant", "Is the pain severe, moderate, or mild, and how long have you had it?"),
        _row("user", "severe"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + ortho_card),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "can u refer me to general doc", "en", history
    )

    assert "Orthopedics" in result
    assert "General Medicine" in result


def test_run_appointment_agent_two_independent_direct_specialty_requests_do_not_warn(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live (found during verification, not a symptom-triage case at
    # all): "available slots for cardiologist" then, later, "available slots for
    # dermatologist" — two unrelated, independently-named specialties, no
    # symptoms ever described. The fallback mismatch check (see the test above)
    # must NOT fire here just because a DIFFERENT department was shown earlier —
    # that earlier card's `note` is null (the patient named Cardiology
    # themselves, nothing was inferred), so there's no real recommendation to
    # contradict. Only a `note`-carrying (genuinely symptom-inferred) earlier
    # card should ever trigger this fallback.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    cardiology = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(cardiology)
    derma = Department(clinic_id=clinic.id, name="Dermatology")
    db.add(derma)
    db.flush()
    derma_doctor = Doctor(
        clinic_id=clinic.id, department_id=derma.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Sara Khan", is_active=True,
    )
    db.add(derma_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=derma_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    # Expected to reach the normal LLM/tool-calling path — this fixture just
    # needs to prove that path was reached (no deterministic mismatch
    # short-circuit fired first), not exercise the LLM call itself.
    def _fake_reply(*args, **kwargs):
        return "Here's Dermatology availability."

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fake_reply)

    # A direct request's own card — `note` is null, since the patient named the
    # department themselves (see chat_tools._doctor_options_payload).
    cardiology_card = json.dumps(
        {
            "note": None,
            "department_name": "Cardiology",
            "doctors": [{"doctor_id": "d1", "doctor_name": "Dr. Ahmed Farooq", "specialization": None, "slots": []}],
        }
    )
    history = [
        _row("user", "available slots for cardiologist"),
        _row("assistant", DOCTOR_OPTIONS_MARKER + cardiology_card),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "available slots for dermatologist", "en", history
    )

    assert result == "Here's Dermatology availability."
    assert "might be a better fit" not in result


def test_run_appointment_agent_warns_when_a_professional_title_names_a_mismatched_department(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live: "i think cardiologist can be a best fit for it?" (after
    # describing leg pain + ear pain, nothing cardiac at all) skipped the mismatch
    # check entirely and went straight to a Cardiology availability card — the
    # literal string "cardiology" isn't even a substring of "cardiologist", so the
    # old exact-department-name check found nothing to warn about. See
    # _DEPARTMENT_TITLE_HINTS in appointment_agent.py.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    ent = Department(clinic_id=clinic.id, name="ENT")
    db.add(ent)
    cardiology = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(cardiology)
    db.flush()
    ent_doctor = Doctor(
        clinic_id=clinic.id, department_id=ent.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Raza", is_active=True,
    )
    db.add(ent_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=ent_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the named department contradicts symptoms")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "i am having pain in my leg and in my ear as well"),
        _row("assistant", "Is the leg pain severe, bearable, or mild? Is the ear pain accompanied by anything else?"),
        _row("user", "the pain is mild and bearable, there is a little swelling on the leg, no other ear symptoms"),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "i think cardiologist can be a best fit for it?", "en", history
    )

    assert "Cardiology" in result
    assert "ENT" in result


def test_run_appointment_agent_warns_when_british_spelled_title_names_a_mismatched_department(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live: after describing burning chest pain (Cardiology hint),
    # "I want to see gynaecologist instead of cardiologist" (British spelling)
    # went straight to a Gynecology card with no mismatch warning at all. Root
    # cause: DEPARTMENT_TITLE_HINTS' "gynaecologist" entry used to map to hint
    # substring "gynaecolog", which never matches this clinic's real department
    # name "Gynecology" (American spelling, no "a") — so the deterministic
    # matcher didn't recognize the message as naming Gynecology at all, and the
    # mismatch check never got the chance to fire. Now reuses the same "gynec"
    # hint as the American "gynecologist" entry, which matches either spelling
    # of the real department name.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    cardiology = Department(clinic_id=clinic.id, name="Cardiology")
    db.add(cardiology)
    gynecology = Department(clinic_id=clinic.id, name="Gynecology")
    db.add(gynecology)
    db.flush()
    gynecology_doctor = Doctor(
        clinic_id=clinic.id, department_id=gynecology.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Rukhsana Hashmi", is_active=True,
    )
    db.add(gynecology_doctor)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=gynecology_doctor.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called when the named department contradicts symptoms")

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", _fail_if_called)

    history = [
        _row("user", "I am having pain in chest"),
        _row("assistant", "Is the chest pain severe, moderate, or mild?"),
        _row("user", "Mild and since 3 days with no other symptoms"),
        _row("assistant", "Can you describe the type of pain?"),
        _row("user", "Burning type"),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "I want to see gynaecologist instead of cardiologist", "en", history
    )

    assert "Gynecology" in result
    assert "Cardiology" in result


def test_run_appointment_agent_does_not_warn_when_a_typo_still_matches_a_real_symptom(
    monkeypatch, db, ctx, clinic, patient
):
    # Reported live: "pain in my teeths and in heart as well" (a typo — extra "s" on
    # the already-plural "teeth") wasn't recognized by the dental hint keyword set
    # at all, so Dentistry was missing from `hinted` even though the LLM itself had
    # already correctly shown a Dentistry card for it earlier in the same session —
    # asking about dentist availability then wrongly triggered the mismatch warning
    # ("Cardiology might be a better fit than Dentistry"). See the "teeths" addition
    # to symptom_hints.py's dental keyword set.
    from datetime import datetime, timedelta, timezone

    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot

    dentistry = Department(clinic_id=clinic.id, name="Dentistry")
    db.add(dentistry)
    db.flush()
    dentist = Doctor(
        clinic_id=clinic.id, department_id=dentistry.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Iqra Qureshi", is_active=True,
    )
    db.add(dentist)
    db.flush()
    db.add(Slot(
        clinic_id=clinic.id, doctor_id=dentist.id,
        start_utc=datetime.now(timezone.utc) + timedelta(days=1),
        end_utc=datetime.now(timezone.utc) + timedelta(days=1, minutes=30),
        status="open",
    ))
    db.flush()

    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Here's Dentistry availability."
    )

    history = [
        _row("user", "i am having pain in my teeths and in heart as well"),
        _row("assistant", "Could you tell me how severe this is and how long you've had it?"),
        _row("user", "its mild and bearable with no other symptoms"),
        _row("assistant", "How long have you been experiencing the tooth pain and the chest discomfort?"),
        _row("user", "its from past 2 days"),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "isnt there any dentist available on tue?", "en", history
    )

    assert result == "Here's Dentistry availability."


def test_run_appointment_agent_proceeds_after_the_mismatch_warning_was_already_given(
    monkeypatch, db, ctx, clinic, patient
):
    # A patient who repeats the same request after already seeing the mismatch
    # warning is making an informed choice — must proceed normally the second time,
    # not refuse forever.
    from app.models.department import Department
    from app.models.doctor import Doctor

    gen_med = Department(clinic_id=clinic.id, name="General Medicine")
    db.add(gen_med)
    neuro = Department(clinic_id=clinic.id, name="Neurology")
    db.add(neuro)
    db.flush()
    db.add(Doctor(
        clinic_id=clinic.id,
        department_id=neuro.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Sana Qureshi",
        is_active=True,
    ))
    db.flush()

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", lambda *a, **k: "reply")

    history = [
        _row("user", "I am having a bit of fever and body aches"),
        _row(
            "assistant",
            "Based on the fever/body aches you described, General Medicine might be a better fit than Neurology. "
            "Would you like me to show General Medicine availability instead, or would you still like to see "
            "Neurology's availability?",
        ),
    ]
    result = appointment_agent.run_appointment_agent(
        db, ctx, "book appointment in Neurology", "en", history
    )

    assert result == "reply"


def test_run_appointment_agent_only_binds_its_five_tools(monkeypatch, db, ctx):
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["tools"] = {t.name for t in tools}
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    # Deliberately a message with no cancel/reschedule action word and no doctor
    # name — tools are bound unconditionally for every appointment_agent call
    # regardless of message content, so this only needs to reach the LLM/tool-
    # calling path at all, not exercise any particular action-resolution branch.
    # ("cancel my appointment" used to serve this purpose too, but with zero real
    # appointments in this test's DB it now correctly short-circuits deterministically
    # instead of reaching the LLM — see
    # test_run_appointment_agent_cancel_with_zero_active_appointments_and_no_doctor_named.)
    appointment_agent.run_appointment_agent(db, ctx, "can you help me with my appointment?", "en", [])

    assert captured["tools"] == {
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "get_my_appointments",
        "get_department_availability",
    }


# =====================================================================================
# Scenario matrix: pending-state x message-type combinations
# =====================================================================================
# Nearly every appointment_agent bug fixed in this file was a PENDING STATE (a marker
# question the assistant just asked — cancel_confirm / appointment disambiguation /
# doctor_name disambiguation) crossed with a MESSAGE TYPE (how the patient's reply is
# phrased) that had never been exercised together before — e.g. a narrowing request
# stated before a name-disambiguation question, then answered with a plain name that
# carries no narrowing word of its own. This matrix drives all three pending kinds
# this module supports against several distinct reply shapes and checks the one
# invariant that must hold no matter which combination fires: a real appointment is
# only ever mutated by an unambiguous "yes" to its own cancel_confirm question — never
# by an unrelated message, a "no", or a message answering a different pending
# question.


def test_scenario_matrix_cancel_confirm_pending_only_yes_mutates(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    appt = _future_appointment(db, clinic, patient, doctor)
    db.commit()
    candidate = {
        "appointment_id": str(appt.id), "doctor_name": "Dr. Ahmed Khan",
        "department_name": "Cardiology", "when": "Mon, Aug 10 at 9:00 AM",
    }
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "cancel_confirm", "question": "confirm?", "candidate": candidate}
    )
    history = [_row("assistant", pending)]

    # "no" -> declined, never mutated.
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM for a clear no")),
    )
    appointment_agent.run_appointment_agent(db, ctx, "no", "en", history)
    db.refresh(appt)
    assert appt.status == "confirmed"

    # An unrelated reply -> falls through to normal handling, still not mutated by
    # this turn alone (no cancel keyword in this message, no active resolution).
    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", lambda *a, **k: "reply")
    appointment_agent.run_appointment_agent(db, ctx, "what are your opening hours", "en", history)
    db.refresh(appt)
    assert appt.status == "confirmed"

    # "yes" -> and only "yes" -> actually cancels.
    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("cancelling must never go through the LLM")),
    )
    result = appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)
    assert "has been cancelled" in result
    db.refresh(appt)
    assert appt.status == "cancelled"


def test_scenario_matrix_appointment_disambiguation_pending_only_a_real_match_resolves(
    monkeypatch, db, ctx, clinic, doctor, other_doctor, patient
):
    _future_appointment(db, clinic, patient, doctor)
    other_appt = _future_appointment(db, clinic, patient, other_doctor)
    candidates = [
        {"appointment_id": "id-1", "doctor_name": "Dr. Ahmed Khan", "department_name": "Cardiology", "when": "Mon"},
        {"appointment_id": str(other_appt.id), "doctor_name": "Dr. Ahmed Raza", "department_name": "Neurology", "when": "Tue"},
    ]
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "appointment", "action": "cancel", "question": "which one?", "candidates": candidates}
    )
    history = [_row("assistant", pending)]

    monkeypatch.setattr(
        appointment_agent.llm, "run_tool_calling_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the LLM before the patient confirms a cancel")),
    )

    # A reply matching neither candidate -> re-asks, no resolution at all.
    reask = appointment_agent.run_appointment_agent(db, ctx, "the first one please", "en", history)
    assert reask.startswith(DOCTOR_DISAMBIGUATION_MARKER)
    assert json.loads(reask[len(DOCTOR_DISAMBIGUATION_MARKER):])["kind"] == "appointment"

    # A reply naming the real candidate -> resolves to a cancel_confirm question for
    # THAT SPECIFIC appointment, still not an actual cancellation yet.
    resolved = appointment_agent.run_appointment_agent(db, ctx, "Dr. Ahmed Raza", "en", history)
    resolved_payload = json.loads(resolved[len(DOCTOR_DISAMBIGUATION_MARKER):])
    assert resolved_payload["kind"] == "cancel_confirm"
    assert resolved_payload["candidate"]["appointment_id"] == str(other_appt.id)


def test_scenario_matrix_doctor_name_disambiguation_pending_only_a_real_match_resolves(
    monkeypatch, db, ctx, clinic
):
    from app.models.department import Department
    from app.models.doctor import Doctor

    ent = Department(clinic_id=clinic.id, name="ENT")
    dentistry = Department(clinic_id=clinic.id, name="Dentistry")
    db.add_all([ent, dentistry])
    db.flush()
    db.add_all([
        Doctor(
            clinic_id=clinic.id, department_id=ent.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
            full_name="Dr. Iqra Raza", is_active=True,
        ),
        Doctor(
            clinic_id=clinic.id, department_id=dentistry.id, external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
            full_name="Dr. Iqra Qureshi", is_active=True,
        ),
    ])
    db.flush()

    candidates = [
        {"doctor_name": "Dr. Iqra Qureshi", "department_name": "Dentistry"},
        {"doctor_name": "Dr. Iqra Raza", "department_name": "ENT"},
    ]
    pending = appointment_agent.DOCTOR_DISAMBIGUATION_MARKER + json.dumps(
        {"kind": "doctor_name", "question": "which one?", "candidates": candidates}
    )
    history = [_row("assistant", pending)]

    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    # A reply naming neither doctor by name at all still reaches the LLM normally
    # (no forced/fake resolution) — this pending kind has no dedicated re-ask
    # short-circuit of its own, unlike the other two.
    appointment_agent.run_appointment_agent(db, ctx, "actually never mind", "en", history)
    assert "RESOLVED DOCTOR" not in captured["system_prompt"]

    # A reply naming the real doctor resolves to exactly that one, correct
    # department included, and reaches the LLM with a real RESOLVED DOCTOR block
    # (no prior DOCTOR_OPTIONS/DEPARTMENT_LIST card in this history, so it isn't
    # treated as already-shown — full narrowing behavior for that combination is
    # covered by test_run_appointment_agent_narrows_after_resolving_a_name_
    # disambiguation_reply above).
    result = appointment_agent.run_appointment_agent(db, ctx, "I mean Dr. Iqra Raza (ENT).", "en", history)
    assert result == "reply"
    assert "RESOLVED DOCTOR" in captured["system_prompt"]
    assert "Dr. Iqra Raza" in captured["system_prompt"]


# =====================================================================================
# general_info_agent
# =====================================================================================


@pytest.mark.parametrize(
    "message",
    [
        "what does dermatologist treat",
        "so what symptoms does dermatologist treats?",
        "what does cardiology handle",
    ],
)
def test_run_general_info_agent_redirects_a_department_scope_question_instead_of_explaining_it(
    monkeypatch, db, ctx, message
):
    # Instructed live: a department-scope question ("what does X treat/handle")
    # used to be answered from KB content describing that department's role —
    # product decision to stop entirely. This bot's job is triage-by-symptom, not
    # an encyclopedia of what each specialty does. Deterministic — never the
    # LLM/KB, regardless of which department is asked about.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM/KB must not be called for a department-scope question")

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", _fail_if_called)
    monkeypatch.setattr(
        general_info_agent, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve"))
    )

    result = general_info_agent.run_general_info_agent(db, ctx, message, "en", [])

    assert result == general_info_agent._DEPARTMENT_ROLE_REDIRECT_EN


def test_run_general_info_agent_department_scope_redirect_is_in_urdu_when_language_is_ur(monkeypatch, db, ctx):
    monkeypatch.setattr(
        general_info_agent.llm, "run_plain_reply",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call the LLM")),
    )
    result = general_info_agent.run_general_info_agent(db, ctx, "what does cardiology handle", "ur", [])
    assert result == general_info_agent._DEPARTMENT_ROLE_REDIRECT_UR


def test_run_general_info_agent_answers_department_list_request_deterministically(monkeypatch, db, ctx, department):
    # Reported live: "what are the available depts" wasn't recognized at all.
    # Answered directly from real, active department names — never the LLM, and
    # never KB retrieval (departments are admin-managed data, not static KB text).
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called for a department-list request")

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", _fail_if_called)
    monkeypatch.setattr(
        general_info_agent, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve"))
    )

    result = general_info_agent.run_general_info_agent(db, ctx, "what are the available depts", "en", [])

    assert result == f"Here are the departments available at this clinic: {department.name}."


def test_run_general_info_agent_department_list_request_says_so_when_none_configured(monkeypatch, db, ctx):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called for a department-list request")

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", _fail_if_called)

    result = general_info_agent.run_general_info_agent(db, ctx, "show me available departments", "en", [])

    assert result == "There are no departments configured at this clinic right now."


def test_run_general_info_agent_department_list_with_explanation_answers_both_halves(
    monkeypatch, db, ctx, department
):
    # Reported live: "show me list of depts in this clinic, and also explanation of
    # each dept that which symptoms does they treat in detail" got ONLY the bare
    # department list back — the second half of the same message (what each
    # department treats) was silently dropped, since the department-list
    # short-circuit always returned immediately regardless of what else was asked.
    # Must now answer both halves in one deterministic reply, built from the same
    # real, vetted symptom_hints table used to route a patient's own symptoms,
    # never an LLM freehanding department descriptions.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called for a department-list request")

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", _fail_if_called)
    monkeypatch.setattr(
        general_info_agent, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve"))
    )

    result = general_info_agent.run_general_info_agent(
        db,
        ctx,
        "show me list of depts in this clinic,and also explanation of each dept that "
        "which symptoms does they treat in detail",
        "en",
        [],
    )

    assert department.name in result
    assert "chest pain" in result  # Cardiology's real, vetted symptom label


def test_run_general_info_agent_department_list_without_explanation_stays_bare(
    monkeypatch, db, ctx, department
):
    # Companion to the test above: a plain list request (no "explain"/"symptoms"/
    # "treat" wording) must still get exactly the original bare list, unchanged.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("the LLM must not be called for a department-list request")

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", _fail_if_called)

    result = general_info_agent.run_general_info_agent(db, ctx, "show me list of depts", "en", [])

    assert result == f"Here are the departments available at this clinic: {department.name}."


def test_run_general_info_agent_answers_personal_recall_question_via_the_llm(monkeypatch, db, ctx):
    # Cross-session memory was removed entirely (see app.services.chat's module
    # docstring) — a personal-recall question like "what are the things i told u"
    # now falls straight through to the normal retrieval+LLM path, same as any
    # other message, instead of a deterministic memory short-circuit.
    from app.rag.retrieval import RetrievalResult

    monkeypatch.setattr(
        general_info_agent,
        "retrieve",
        lambda db, clinic_id, query: RetrievalResult(matched=False, best_score=0.0, chunks=[], fallback_message=None),
    )
    monkeypatch.setattr(general_info_agent, "rewrite_query", lambda message, history: message)

    captured = {}

    def fake_run_plain_reply(system_prompt, message, history):
        captured["called"] = True
        return "I don't have anything stored."

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", fake_run_plain_reply)

    result = general_info_agent.run_general_info_agent(db, ctx, "what are the things i told u", "en", [])

    assert captured.get("called") is True
    assert result == "I don't have anything stored."


def test_run_general_info_agent_uses_system_prompt_unchanged(monkeypatch, db, ctx):
    from app.rag.retrieval import RetrievalResult

    monkeypatch.setattr(
        general_info_agent,
        "retrieve",
        lambda db, clinic_id, query: RetrievalResult(
            matched=True, best_score=0.9, chunks=["Clinic hours: 9-5."], fallback_message=None
        ),
    )

    captured = {}

    def fake_run_plain_reply(system_prompt, message, history):
        captured["system_prompt"] = system_prompt
        return "We're open 9-5."

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", fake_run_plain_reply)

    result = general_info_agent.run_general_info_agent(db, ctx, "what are your hours", "en", [])

    assert result == "We're open 9-5."
    assert "Clinic hours: 9-5." in captured["system_prompt"]
    assert "STRICT GROUNDING RULE" in captured["system_prompt"]
    assert "CONVERSATIONAL EXCEPTION" in captured["system_prompt"]
    assert "DEPARTMENT VS SPECIALIZATION" in captured["system_prompt"]


def test_run_general_info_agent_omits_patient_memory_section_entirely(monkeypatch, db, ctx):
    # PATIENT MEMORY was removed from the prompt entirely — see
    # app.services.chat's module docstring for why leaving it in with an
    # always-empty value was actively harmful, not just wasted tokens.
    from app.rag.retrieval import RetrievalResult

    monkeypatch.setattr(
        general_info_agent,
        "retrieve",
        lambda db, clinic_id, query: RetrievalResult(matched=False, best_score=0.0, chunks=[], fallback_message="n/a"),
    )
    monkeypatch.setattr(general_info_agent, "rewrite_query", lambda message, history: message)

    captured = {}
    monkeypatch.setattr(
        general_info_agent.llm,
        "run_plain_reply",
        lambda system_prompt, message, history: captured.setdefault("system_prompt", system_prompt) or "hi",
    )

    general_info_agent.run_general_info_agent(db, ctx, "hi", "en", [])

    assert "PATIENT MEMORY" not in captured["system_prompt"]


def test_run_general_info_agent_rewrite_rescues_a_query_that_fails_on_its_own(monkeypatch, db, ctx):
    """The raw, conversationally-diluted message would score below threshold on its
    own; rewrite_query cleans it into a standalone question that clears it — proves
    run_general_info_agent actually uses the rewritten query for retrieval when the
    raw message fails, not just always the raw message."""
    from app.rag.retrieval import RetrievalResult

    raw_message = "not on the info but overall in total how many departments does this clinic have"
    rewritten = "How many departments does this clinic have?"

    monkeypatch.setattr(general_info_agent, "rewrite_query", lambda message, history: rewritten)

    def fake_retrieve(db, clinic_id, query):
        if query == rewritten:
            return RetrievalResult(matched=True, best_score=0.9, chunks=["5 departments."], fallback_message=None)
        return RetrievalResult(matched=False, best_score=0.1, chunks=[], fallback_message="n/a")

    monkeypatch.setattr(general_info_agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", lambda system_prompt, message, history: "5 departments.")

    result = general_info_agent.run_general_info_agent(db, ctx, raw_message, "en", [])

    assert result == "5 departments."


@pytest.mark.parametrize(
    "message",
    [
        "hello bye",  # reported live: leaked real (irrelevant) KB chunks instead of "(none)"
        "hi",
        "thanks",
        "ok great",
    ],
)
def test_run_general_info_agent_skips_retrieval_entirely_for_small_talk(monkeypatch, db, ctx, message):
    """retrieve()'s own similarity gate only inspects the single best chunk's raw
    score, which can accidentally clear the threshold for a short, low-content
    message purely by chance — classify_message_intent's deterministic CONVERSATIONAL
    check must short-circuit before retrieve() is ever called, not rely on that gate."""
    monkeypatch.setattr(
        general_info_agent, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve"))
    )
    monkeypatch.setattr(
        general_info_agent, "rewrite_query", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not rewrite"))
    )

    captured = {}

    def fake_run_plain_reply(system_prompt, message, history):
        captured["system_prompt"] = system_prompt
        return "Hey there!"

    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", fake_run_plain_reply)

    result = general_info_agent.run_general_info_agent(db, ctx, message, "en", [])

    assert result == "Hey there!"
    assert "(none)" in captured["system_prompt"]


def test_run_general_info_agent_skips_retrieval_for_a_personal_recall_question(monkeypatch, db, ctx):
    monkeypatch.setattr(
        general_info_agent, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retrieve"))
    )
    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", lambda system_prompt, message, history: "reply")

    result = general_info_agent.run_general_info_agent(db, ctx, "what is my name", "en", [])

    assert result == "reply"


def test_run_general_info_agent_still_retrieves_for_a_real_question_that_happens_to_be_short(monkeypatch, db, ctx):
    """Guards against the fix above becoming over-broad: a short message is only
    exempted from retrieval when it's genuinely small talk/personal-recall shaped —
    a short real question must still hit retrieve()."""
    from app.rag.retrieval import RetrievalResult

    monkeypatch.setattr(
        general_info_agent,
        "retrieve",
        lambda db, clinic_id, query: RetrievalResult(
            matched=True, best_score=0.9, chunks=["Clinic hours: 9-5."], fallback_message=None
        ),
    )
    monkeypatch.setattr(general_info_agent.llm, "run_plain_reply", lambda system_prompt, message, history: "9-5.")

    result = general_info_agent.run_general_info_agent(db, ctx, "what are your hours?", "en", [])

    assert result == "9-5."


def test_run_general_info_agent_never_rewrites_a_clean_standalone_question_even_with_unrelated_history(
    monkeypatch, db, ctx
):
    """Reproduces a previously-fixed bug: history is loaded cross-session by
    chat.py, so a brand-new, unambiguous question used to get silently rewritten
    using leftover context from an unrelated prior conversation even though the raw
    question already clears the threshold on its own. rewrite_query must not run at
    all when the raw message already matches."""
    from app.rag.retrieval import RetrievalResult

    calls = {"rewrite": 0}

    def counting_rewrite(message, history):
        calls["rewrite"] += 1
        return "a rewritten query that must never be used"

    monkeypatch.setattr(general_info_agent, "rewrite_query", counting_rewrite)

    clean_question = "just tell me the names of doctors in cardiology"

    def fake_retrieve(db, clinic_id, query):
        assert query == clean_question, "retrieve() must receive the raw message untouched"
        return RetrievalResult(
            matched=True, best_score=0.95, chunks=["Cardiology: Dr. Ahmed Farooq, Dr. Farhan Malik."],
            fallback_message=None,
        )

    monkeypatch.setattr(general_info_agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        general_info_agent.llm,
        "run_plain_reply",
        lambda system_prompt, message, history: "Dr. Ahmed Farooq and Dr. Farhan Malik.",
    )

    # Unrelated prior conversation from a DIFFERENT session — this is exactly what a
    # "new chat" doesn't leave behind, but the regression happened because history
    # was loaded cross-session before that boundary existed.
    unrelated_history = [
        _row("user", "I have chest tightness and a cough, is that cardiology or pulmonology?"),
        _row("assistant", "That could be either — can you tell me more about your breathing?"),
    ]

    result = general_info_agent.run_general_info_agent(db, ctx, clean_question, "en", unrelated_history)

    assert calls["rewrite"] == 0, "rewrite_query must not run when the raw message already clears the threshold"
    assert result == "Dr. Ahmed Farooq and Dr. Farhan Malik."

