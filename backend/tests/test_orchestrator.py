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


def test_run_symptom_agent_recommendation_request_falls_through_when_nothing_hinted_yet(monkeypatch, db, ctx):
    # No real symptom has been described yet this session — nothing to base a
    # recommendation on, so this must fall through to the normal triage flow
    # (which will ask what's wrong) rather than returning an empty/broken reply.
    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "What symptoms are you having?")

    result = symptom_agent.run_symptom_agent(db, ctx, "what do you recommend?", "en", [])

    assert result == "What symptoms are you having?"


def test_run_symptom_agent_first_message_combining_symptoms_and_recommendation_still_gets_normal_triage(
    monkeypatch, db, ctx
):
    # Reported live: "I am having a bit of fever and body aches, what do you
    # recommend?" — symptom description AND the recommendation phrase in the SAME,
    # very first message — short-circuited straight to a department card, skipping
    # the normal PATH 2 screening questions (severity/duration) that must always come
    # first for a symptom nobody has triaged yet. Since this is the FIRST mention of
    # the symptom (nothing in prior history), this must still go through the normal
    # LLM triage flow, not the deterministic recommendation short-circuit.
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
def test_run_symptom_agent_deterministically_asks_before_a_brand_new_first_message_symptom(monkeypatch, db, ctx, message):
    # Reported live TWICE despite an explicit prompt instruction telling the model
    # not to do this ("MOST COMMON WAY THIS RULE GETS BROKEN" in llm.py's PATH 3):
    # a bare, mild-sounding symptom as the very first message of a brand new
    # session skipped straight to a department card with zero clarifying
    # questions. The prompt instruction alone wasn't a reliable enough guarantee,
    # so this is now a deterministic backstop — the LLM must not even be called
    # for this shape of message.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_tool_calling_agent must not be called for a brand-new first-message symptom")

    monkeypatch.setattr(symptom_agent.llm, "run_tool_calling_agent", _fail_if_called)

    result = symptom_agent.run_symptom_agent(db, ctx, message, "en", [])

    assert "severe" in result.lower()
    assert "how long" in result.lower()


def test_run_symptom_agent_first_message_backstop_skips_when_duration_and_severity_already_given(monkeypatch, db, ctx):
    # PATH 3's own stated exception: duration + severity already given together
    # means genuine clarification already happened, so the normal LLM triage flow
    # runs as usual instead of the deterministic backstop re-asking for the same
    # info the patient already provided.
    monkeypatch.setattr(
        symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Let me check availability for you."
    )

    result = symptom_agent.run_symptom_agent(db, ctx, "mild jaw pain for the past 2 days", "en", [])

    assert result == "Let me check availability for you."


@pytest.mark.parametrize("message", ["i have brain tumor", "i think i have cancer"])
def test_run_symptom_agent_first_message_backstop_skips_self_diagnosis_claims(monkeypatch, db, ctx, message):
    # Self-diagnosis claims are deliberately excluded from the backstop — asking
    # "is that mild, moderate, or severe?" in response to "I have a brain tumor"
    # reads as dismissive for something a patient is already alarmed about; the
    # established fix for this category is a fast, concise LLM-composed redirect
    # instead (see _is_self_diagnosis_claim's own comment in symptom_agent.py).
    monkeypatch.setattr(
        symptom_agent.llm, "run_tool_calling_agent", lambda *a, **k: "Let me check availability for you."
    )

    result = symptom_agent.run_symptom_agent(db, ctx, message, "en", [])

    assert result == "Let me check availability for you."


def test_run_symptom_agent_first_message_backstop_does_not_apply_once_history_exists(monkeypatch, db, ctx):
    # The backstop is deliberately scoped to no REAL symptom having been described
    # yet — once a prior turn already described one (a headache here), the normal
    # LLM triage flow (which already has real conversation context to reason from)
    # applies, regardless of whether `history` is literally empty.
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


def test_run_symptom_agent_does_not_hint_a_department_with_no_matching_symptom_words(monkeypatch, db, ctx):
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
    # confirms the new faked-payload detector catches this case too and rebuilds
    # real cards for BOTH hinted departments.
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
    assert "General Medicine" in result
    assert "Dr. Ahmed Farooq" in result
    assert "Dr. Ali Raza" in result


def test_run_symptom_agent_recovers_neurology_when_the_model_diagnosed_a_brain_tumor(monkeypatch, db, ctx, clinic):
    # Reported live: "i have brain tumor" got a long free-text breakdown (department
    # + doctor schedule + booking/reschedule mechanics) from general_info_agent
    # instead of a concise department card, because the message had no matching
    # symptom keyword at all and never reached symptom_agent. Now that routing is
    # fixed (message_classifier._SYMPTOM_KEYWORDS includes "tumor"/"tumour"/
    # "cancer"), this confirms symptom_agent's own diagnosis-recovery net also has a
    # real department to fall back to if the model still names the condition in
    # free text instead of calling the tool.
    from app.models.department import Department
    from app.models.doctor import Doctor
    from app.models.slot import Slot
    from datetime import datetime, timedelta, timezone

    neuro = Department(clinic_id=clinic.id, name="Neurology")
    db.add(neuro)
    db.flush()
    doctor = Doctor(
        clinic_id=clinic.id,
        department_id=neuro.id,
        external_doctor_id=f"DOC-{uuid.uuid4().hex[:8]}",
        full_name="Dr. Sana Qureshi",
        is_active=True,
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
    assert payload["department_name"] == "Neurology"
    assert payload["doctors"][0]["doctor_name"] == "Dr. Sana Qureshi"
    assert "tumor" not in result.lower()


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
    monkeypatch, db, ctx
):
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


def test_run_appointment_agent_skips_confirmation_when_doctor_already_shown_in_history(
    monkeypatch, db, ctx, doctor
):
    # Reported live: the assistant re-asked "did you mean Dr. Ahmed Khan in
    # Cardiology?" for a doctor it had itself already listed in a card two turns
    # earlier — reads as not having listened. Once the same doctor+department
    # already appeared in a real card in history, no confirming question is needed.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row(
            "assistant",
            DEPARTMENT_LIST_MARKER
            + json.dumps({"departments": [{"department_name": "Cardiology", "doctors": [{"doctor_name": "Dr. Ahmed Khan"}]}]}),
        ),
    ]
    appointment_agent.run_appointment_agent(db, ctx, "Book with Dr. Ahmed Khan", "en", history)

    assert "RESOLVED DOCTOR" in captured["system_prompt"]
    assert "already shown to the patient earlier this conversation" in captured["system_prompt"]
    assert "ask ONE direct confirming question" not in captured["system_prompt"]


def test_run_appointment_agent_resolves_doctor_from_a_plain_yes_confirming_a_prior_question(
    monkeypatch, db, ctx, doctor
):
    # Reported live: "book with dr raza ali" -> "Did you mean Dr. Ali Raza in
    # General Medicine?" -> patient replies "yes" -> the next reply asked "which
    # department does Dr. Raza Ali work in?", as if the confirmation never
    # happened. "yes" names no doctor itself, so the deterministic name-match on
    # the CURRENT message alone finds nothing — this confirms the fallback
    # recovers the doctor from the patient's own PRIOR message instead.
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    history = [
        _row("user", "i wants to book an appointment with dr ahmed khan"),
        _row("assistant", "Did you mean Dr. Ahmed Khan in Cardiology?"),
    ]
    appointment_agent.run_appointment_agent(db, ctx, "yes", "en", history)

    assert "RESOLVED DOCTOR" in captured["system_prompt"]
    assert "Dr. Ahmed Khan" in captured["system_prompt"]
    assert "Cardiology" in captured["system_prompt"]
    # The "yes" itself IS the confirmation — must not ask the same question again.
    assert "ask ONE direct confirming question" not in captured["system_prompt"]


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


def test_run_appointment_agent_auto_resolves_when_exactly_one_active_appointment(
    monkeypatch, db, ctx, clinic, doctor, patient
):
    appt = _future_appointment(db, clinic, patient, doctor)
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    result = appointment_agent.run_appointment_agent(db, ctx, "cancel my appointment", "en", [])

    assert result == "reply"
    assert "RESOLVED APPOINTMENT" in captured["system_prompt"]
    assert str(appt.id) in captured["system_prompt"]
    assert "Call cancel_appointment with this exact appointment_id" in captured["system_prompt"]


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
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    result = appointment_agent.run_appointment_agent(
        db, ctx, "cancel my appointment with Dr. Ahmed Raza", "en", []
    )

    assert result == "reply"
    assert "RESOLVED APPOINTMENT" in captured["system_prompt"]
    assert str(other_appt.id) in captured["system_prompt"]


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
    captured = {}

    def fake_run_tool_calling_agent(system_prompt, message, history, tools):
        captured["system_prompt"] = system_prompt
        return "reply"

    monkeypatch.setattr(appointment_agent.llm, "run_tool_calling_agent", fake_run_tool_calling_agent)

    result = appointment_agent.run_appointment_agent(db, ctx, "Dr. Ahmed Raza", "en", history)

    assert result == "reply"
    assert "RESOLVED APPOINTMENT" in captured["system_prompt"]
    # Resolves to the candidate the reply actually matched (Dr. Ahmed Raza's real
    # id), not the other, non-matching candidate in the same payload.
    assert str(other_appt.id) in captured["system_prompt"]
    assert "wrong-would-be-a-bug" not in captured["system_prompt"]
    # No redundant RESOLVED DOCTOR confirmation for the same doctor on top of this.
    assert "ask ONE direct confirming question" not in captured["system_prompt"]


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

    appointment_agent.run_appointment_agent(db, ctx, "cancel my appointment", "en", [])

    assert captured["tools"] == {
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "get_my_appointments",
        "get_department_availability",
    }


# =====================================================================================
# general_info_agent
# =====================================================================================


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
