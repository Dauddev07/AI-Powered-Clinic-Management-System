import json

import httpx
import pytest
from groq import APIStatusError, RateLimitError

from app.services import llm


def _rate_limit_error(model: str = "test-model") -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError(f"rate limited on {model}", response=response, body=None)


def _token_limit_error(model: str = "test-model") -> APIStatusError:
    # Reproduces a real observed failure: Groq rejects a request with a plain 413
    # (not the 429 RateLimitError subclass) when it exceeds a model's tokens-per-
    # minute cap — a long conversation/system prompt can trip this same as an actual
    # rate limit, and it needs the same same-provider-model-swap handling.
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=413, request=request)
    return APIStatusError(f"request too large for {model}", response=response, body=None)


class _FakeToolCallResponse:
    def __init__(self, tool_calls):
        self.content = ""
        self.tool_calls = tool_calls


class _FakeFinalResponse:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class _FakeTool:
    def __init__(self, name, result="tool result"):
        self.name = name
        self._result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self._result


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id}


class _FakeLLM:
    """Stand-in for a ChatGroq instance, wrapping a shared per-model iterator (see
    _patch_chat_groq) rather than owning its own — real code builds a brand-new
    ChatGroq(...).bind_tools(tools) on every call to _invoke_with_fallback, including
    across separate run_chat_agent loop iterations reusing the same model, so the
    canned response queue for a given model must persist across multiple
    constructions, not restart from the top each time. bind_tools() is a no-op
    returning self, matching real ChatGroq usage."""

    def __init__(self, responses_iter):
        self._responses = responses_iter

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        item = next(self._responses)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_chat_groq(monkeypatch, responses_by_model):
    """`responses_by_model`: dict of model name -> list of responses/exceptions for
    that model, consumed in order across however many times that model gets
    (re)constructed. A model not in the dict gets an empty queue (any .invoke() call
    on it raises StopIteration, surfacing a test bug immediately rather than
    silently returning None)."""
    shared_iters = {model: iter(responses) for model, responses in responses_by_model.items()}

    def _fake_chat_groq(model, **kwargs):
        return _FakeLLM(shared_iters.setdefault(model, iter([])))

    monkeypatch.setattr(llm, "ChatGroq", _fake_chat_groq)


# --- item: Groq-only retry sequence (gpt-oss-120b -> qwen/qwen3.6-27b -> gpt-oss-120b) ---


def test_retry_sequence_is_gpt_oss_then_qwen_then_gpt_oss_again():
    first, second, third = llm._MODEL_RETRY_SEQUENCE
    assert first == "openai/gpt-oss-120b"
    assert second == "qwen/qwen3.6-27b"
    assert third == "openai/gpt-oss-120b"


# --- token-reduction: reasoning_effort is model-specific, never one-size-fits-all --


def test_reasoning_effort_is_low_for_gpt_oss_models():
    # Verified directly against Groq's API: gpt-oss models reject "none"/"default"
    # and accept only "low"/"medium"/"high" — "low" cuts hidden chain-of-thought
    # reasoning tokens dramatically with no effect on the visible reply, since those
    # tokens are already stripped from `content` by reasoning_format="hidden" and
    # were never shown to the patient in the first place.
    assert llm._reasoning_effort_for("openai/gpt-oss-120b") == "low"
    assert llm._reasoning_effort_for("openai/gpt-oss-20b") == "low"


def test_reasoning_effort_is_omitted_for_qwen():
    # Reproduces a real 400 error verified directly against Groq's API: qwen rejects
    # "low"/"medium"/"high" outright (the opposite constraint from gpt-oss) — passing
    # it would break the qwen fallback attempt in the retry sequence every time it's
    # actually needed (i.e. right when the primary model is rate-limited), silently
    # degrading _MODEL_RETRY_SEQUENCE down to two real attempts instead of three.
    assert llm._reasoning_effort_for("qwen/qwen3.6-27b") is None


def test_get_chat_reply_and_run_chat_agent_pass_per_model_reasoning_effort(monkeypatch):
    seen_kwargs = []

    def fake_chat_groq(**kwargs):
        seen_kwargs.append(kwargs)
        return _FakeLLM(iter([_FakeFinalResponse("hi")]))

    monkeypatch.setattr(llm, "ChatGroq", fake_chat_groq)
    llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    assert seen_kwargs[0]["reasoning_effort"] == "low"
    assert seen_kwargs[0]["model"] == "openai/gpt-oss-120b"


def test_no_rate_limit_uses_the_primary_model_only_happy_path_unchanged(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    calls = []

    def _fake_chat_groq(model, **kwargs):
        calls.append(model)
        return _FakeLLM(iter([_FakeFinalResponse("Hello there.")]))

    monkeypatch.setattr(llm, "ChatGroq", _fake_chat_groq)

    result = llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    assert result == "Hello there."
    assert calls == [first]
    assert second not in calls


def test_rate_limit_on_gpt_oss_only_succeeds_via_qwen(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_rate_limit_error(first)],
            second: [_FakeFinalResponse("Served by qwen/qwen3.6-27b.")],
        },
    )

    result = llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    # Succeeds via the second attempt — no error surfaced to the patient.
    assert result == "Served by qwen/qwen3.6-27b."


def test_token_limit_413_on_gpt_oss_falls_back_to_qwen_same_as_a_rate_limit(monkeypatch):
    # Regression: a plain groq.APIStatusError (413, tokens-per-minute cap) used to
    # NOT be caught here at all (only the 429 RateLimitError subclass was), so it
    # crashed the whole request instead of advancing to the next model — surfacing
    # as an unhandled 500 to the patient instead of the graceful fallback reply.
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_token_limit_error(first)],
            second: [_FakeFinalResponse("Served by qwen/qwen3.6-27b.")],
        },
    )

    result = llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    assert result == "Served by qwen/qwen3.6-27b."


def test_token_limit_413_on_every_attempt_gives_the_graceful_fallback_not_a_crash(monkeypatch):
    first, second, third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_token_limit_error(first), _token_limit_error(first)],
            second: [_token_limit_error(second)],
        },
    )
    assert third == first  # sanity: the sequence really does circle back to `first`

    result = llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    assert result == llm._RATE_LIMIT_REPLY


def test_rate_limit_on_gpt_oss_and_qwen_succeeds_on_the_second_gpt_oss_attempt(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    # `first` (gpt-oss-120b) appears at both position 1 and position 3 of the
    # sequence, so its queue holds two entries: the first exhausted at position 1,
    # the second consumed when the sequence circles back at position 3.
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_rate_limit_error(first), _FakeFinalResponse("Served by the second gpt-oss-120b attempt.")],
            second: [_rate_limit_error(second)],
        },
    )

    result = llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    assert result == "Served by the second gpt-oss-120b attempt."


def test_rate_limit_on_all_three_attempts_returns_the_plain_message_not_an_error(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_rate_limit_error(first), _rate_limit_error(first)],
            second: [_rate_limit_error(second)],
        },
    )

    result = llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])

    assert result == llm._RATE_LIMIT_REPLY


def test_a_non_rate_limit_error_is_not_retried_against_the_next_attempt(monkeypatch):
    first, _second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(monkeypatch, {first: [RuntimeError("some other failure")]})

    with pytest.raises(RuntimeError):
        llm.get_chat_reply(message="hi", language="en", context_chunks=[], history=[])


def test_run_chat_agent_rate_limit_on_gpt_oss_only_succeeds_via_qwen(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_rate_limit_error(first)],
            second: [_FakeFinalResponse("Agent reply from qwen/qwen3.6-27b.")],
        },
    )

    result = llm.run_chat_agent(message="hi", language="en", context_chunks=[], history=[], tools=[])

    assert result == "Agent reply from qwen/qwen3.6-27b."


def test_run_chat_agent_rate_limit_on_gpt_oss_and_qwen_succeeds_on_second_gpt_oss_attempt(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_rate_limit_error(first), _FakeFinalResponse("Agent reply from the second gpt-oss-120b attempt.")],
            second: [_rate_limit_error(second)],
        },
    )

    result = llm.run_chat_agent(message="hi", language="en", context_chunks=[], history=[], tools=[])

    assert result == "Agent reply from the second gpt-oss-120b attempt."


def test_run_chat_agent_returns_plain_message_when_all_three_attempts_are_rate_limited(monkeypatch):
    first, second, _third = llm._MODEL_RETRY_SEQUENCE
    _patch_chat_groq(
        monkeypatch,
        {
            first: [_rate_limit_error(first), _rate_limit_error(first)],
            second: [_rate_limit_error(second)],
        },
    )

    result = llm.run_chat_agent(message="hi", language="en", context_chunks=[], history=[], tools=[])

    assert result == llm._RATE_LIMIT_REPLY


def test_no_gemini_anywhere_in_llm_module():
    # Checks for concrete code markers, not the bare word "gemini" — this module's
    # own comments legitimately explain that there is no Gemini fallback anymore,
    # which would false-positive on a plain substring check.
    import app.services.llm as llm_module

    source = open(llm_module.__file__).read()
    assert "ChatGoogleGenerativeAI" not in source
    assert "langchain_google_genai" not in source
    assert "GEMINI_API_KEY" not in source
    assert "GEMINI_MODEL" not in source
    assert not hasattr(llm_module.settings, "GEMINI_API_KEY")
    assert not hasattr(llm_module.settings, "GEMINI_MODEL")


# --- regression: a tool call that raises must never 500 the whole turn -------------


def test_a_raising_tool_call_never_crashes_the_turn_and_gives_a_safe_reply(monkeypatch):
    class _RaisingTool:
        name = "get_department_availability"

        def invoke(self, args):
            raise ValueError("simulated schema validation failure on a bad date argument")

    _patch_chat_groq(
        monkeypatch,
        {
            llm._MODEL_RETRY_SEQUENCE[0]: [
                _FakeToolCallResponse([_tool_call("get_department_availability", {"earliest_date": "not-real"}, "call-1")]),
                _FakeFinalResponse("I'm not sure what happened there."),
            ]
        },
    )

    result = llm.run_chat_agent(
        message="is there no doctor available for tommorow?",
        language="en",
        context_chunks=[],
        history=[],
        tools=[_RaisingTool()],
    )

    assert "sorry" in result.lower()
    assert "rephrasing" in result.lower()


def test_a_raising_unknown_named_tool_still_never_crashes_the_turn(monkeypatch):
    # A terminal tool raising must be just as safe as a non-terminal one.
    class _RaisingTool:
        name = "book_appointment"

        def invoke(self, args):
            raise RuntimeError("boom")

    _patch_chat_groq(
        monkeypatch,
        {llm._MODEL_RETRY_SEQUENCE[0]: [_FakeToolCallResponse([_tool_call("book_appointment", {"slot_id": "abc"}, "call-1")])]},
    )

    result = llm.run_chat_agent(
        message="book that slot",
        language="en",
        context_chunks=[],
        history=[],
        tools=[_RaisingTool()],
    )

    assert "sorry" in result.lower()


# --- item 1: non-red-flag symptom triage also screens for urgency ------------------


def test_agent_prompt_instructs_an_emergency_screening_clarifying_question():
    # A genuinely live LLM call isn't exercised in this test suite (see other tests'
    # mocked-LLM approach) — this asserts the prompt-engineering requirement itself
    # is actually present in the system prompt the model is given, since that's the
    # only lever this feature is implemented through.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "help you tell an urgent presentation apart from a routine one" in prompt


def test_agent_prompt_instructs_plain_urgent_recommendation_on_a_worrying_answer():
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "plainly and directly tell the patient this sounds like an emergency" in prompt
    assert "do not call get_department_availability" in prompt.lower()


def test_agent_prompt_emergency_backstop_applies_to_any_message_not_just_answers():
    # The backstop must not be scoped to "an answer to a clarifying question" only —
    # a first message describing a severe/emergency presentation (that the
    # same-message regex check happened to miss, e.g. a typo or an unusual injury
    # combination) must trigger it too.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "not only an answer to your own clarifying question" in prompt
    assert "including the very first message" in prompt


def test_agent_prompt_handles_emergency_with_no_matching_department():
    # If the clinic has no department appropriate for a genuine emergency, the model
    # must still say so plainly rather than silently forcing a department match or
    # treating it as routine triage.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "if none of them are actually the right fit for a genuine emergency" in prompt
    assert "rather than silently forcing a" in prompt


def test_agent_prompt_defers_to_the_same_message_emergency_check_as_authoritative():
    # The secondary screening layer must explicitly describe itself as
    # non-authoritative and subordinate to red_flag.py's same-message check — never
    # implying it could override or delay that check.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "non-authoritative" in prompt
    assert "takes priority" in prompt


def test_agent_prompt_requires_first_aid_guidance_on_confirmed_emergency():
    # PATH 1 must tell the patient to go to the ER AND give 2-3 safe, generic
    # first-aid tips for the wait — never medication or a home remedy.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "give 2–3 short, generic, safe first-aid actions" in prompt
    assert "Never suggest medication" in prompt
    assert "while you're on your way" in prompt


def test_agent_prompt_requires_severity_screening_for_ambiguous_symptoms():
    # PATH 2: chest pain/tightness/head pain must be screened with a direct
    # severity question ("severe, bearable, or mild?") before any department
    # routing decision — this is the product-level change from auto-firing on
    # "chest pain" alone (see red_flag.py) to asking first.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 2" in prompt
    assert "is it severe, bearable, or mild?" in prompt
    assert "Do NOT decide a department, do NOT call get_department_availability" in prompt
    assert "chest pain, chest tightness or pressure; head pain or a severe headache" in prompt


def test_agent_prompt_path2_covers_broad_real_world_symptom_list_not_just_examples():
    # PATH 2 must explicitly be a non-exhaustive, general rule ("any presentation
    # that plausibly ranges from routine to a genuine emergency"), not just the
    # three original named examples — plus cover several concrete additional
    # real-world categories (abdominal pain, high fever, dizziness, etc.) with
    # their own differentiator guidance.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "this is not a short fixed list" in prompt
    assert "severe or persistent abdominal" in prompt
    assert "stomach pain" in prompt
    assert "high or persistent fever" in prompt
    assert "dizziness, lightheadedness, or feeling faint" in prompt
    assert "pain or bleeding during pregnancy" in prompt


def test_agent_prompt_intro_lists_expanded_auto_fire_categories():
    # The intro note should mention the newly-added auto-fire categories (see
    # red_flag.py) so the model's own mental model of "what's already covered
    # before I see it" stays in sync with the actual regex gate.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "choking, poisoning/overdose" in prompt
    assert "severe burns, electrocution, drowning" in prompt
    assert "gunshot/stab wounds" in prompt


def test_agent_prompt_includes_broken_bone_in_ambiguous_symptom_screening():
    # Reported bug: "i got the bone of my arm broken" fell through PATH 2 entirely
    # (only chest/head pain were listed) and PATH 3's routine flow led the model to
    # try naming the injury ("this is a fracture"), which diagnosis_guard correctly
    # blocked and swapped for the unhelpful generic redirect. A suspected fracture
    # must be in PATH 2's list, with its own differentiators, so the model asks
    # about severity/deformity/numbness first instead of asserting a diagnosis.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "a broken bone or suspected fracture" in prompt
    assert "visible deformity, or bone visible through the skin" in prompt
    assert 'asserting "this is a fracture" is not' in prompt


def test_agent_prompt_requires_two_to_three_questions_for_routine_symptoms():
    # PATH 3: routine (non-ambiguous) symptoms now require 2-3 clarifying
    # questions before routing, up from the earlier 1-2.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 3" in prompt
    assert "ask 2–3 real clarifying questions" in prompt


def test_agent_prompt_caps_path2_at_one_screening_round():
    # Reported bug: for chest pain, the model asked severity, then a SEPARATE reply
    # asking about shortness of breath/sweating/nausea, then ANOTHER asking about
    # pressure vs tightness, then ANOTHER asking about activity/illness — four
    # rounds of questions and it never once called get_department_availability,
    # eventually hitting a rate limit. PATH 2 must be capped at exactly one
    # screening reply before a mandatory decision.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 2 IS EXACTLY ONE ROUND — HARD LIMIT" in prompt
    assert "Do not ask a second round of differentiator questions" in prompt
    assert "decide on your very next reply, no exceptions" in prompt


def test_agent_prompt_caps_path3_at_three_questions_hard_ceiling():
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "3 QUESTIONS IS A HARD CEILING, NOT A TARGET" in prompt
    assert "your very next reply MUST call get_department_availability" in prompt
    assert "counting any PATH 2 screening reply as one of them" in prompt


def test_agent_prompt_treats_a_string_of_denied_red_flags_as_enough_to_route_early():
    # Reproduces a reported bug: mild neck pain, patient denied every differentiator
    # asked (no radiation, no stiffness, no known trigger) across 3 separate
    # questions, and the model kept probing instead of routing once it already had a
    # clearly mild, stable picture.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "a string of \"no\"s across several questions is the ceiling being hit early" in prompt


def test_agent_prompt_forbids_ending_path3_with_freehand_advice_instead_of_the_tool_call():
    # Reproduces a reported bug: a mild-neck-pain conversation ended with a long
    # freehand paragraph of home-remedy/OTC-medication advice and never once called
    # get_department_availability — no department was ever offered.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 3 ALWAYS ENDS WITH THE TOOL CALL — NEVER WITH FREE-TEXT ADVICE INSTEAD OF IT" in prompt
    assert "never as a multi-point home-remedy essay" in prompt


def test_agent_prompt_forbids_inventing_a_nonexistent_department():
    # Reproduces a reported bug: "hit by a stone on my head" with minor bleeding
    # (screened as non-emergency) led the model to call get_department_availability
    # with a department name like "Emergency" that isn't one of this clinic's real
    # 12 departments, producing a broken "I couldn't find a department called that"
    # reply instead of routing to a real department (e.g. Neurology).
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NEVER call get_department_availability with a department_name you've silently" in prompt
    assert '"Emergency" or "ER"' in prompt


def test_agent_prompt_gives_first_aid_and_er_escalation_for_notable_injuries():
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NOTABLE-BUT-NOT-CLEARLY-EMERGENCY INJURIES" in prompt
    assert "go to the nearest ER right away instead of a scheduled visit" in prompt


def test_agent_prompt_forbids_red_flag_jargon_in_patient_facing_text():
    # Reproduces a reported bug: a wound-care note ended with "...not a substitute
    # for emergency care if red-flag signs develop" — "red flag" is our own internal
    # screening terminology (used throughout this prompt to instruct the model),
    # not something a patient reading the reply knows the meaning of.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PLAIN-LANGUAGE RULE" in prompt
    assert 'never use internal clinical/ triage shorthand like "red flag(s)"' in prompt
    assert "never the bare term \"red flag(s)\"" in prompt


def test_agent_prompt_rechecks_current_appointment_status_before_assuming_conflict():
    # Reproduces a reported bug: patient booked Cardiology Mon 8am via chat, then
    # cancelled it manually (not through chat) — the model still had the old
    # BOOKING_CONFIRMED card in conversation history and, asked to book a new slot,
    # assumed the old appointment was still active and asked whether to reschedule/
    # cancel it instead of just booking the new one.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NEVER treat a booking/reschedule confirmation from earlier in this conversation's" in prompt
    assert "STOP and call get_my_appointments first" in prompt


def test_agent_prompt_rechecks_before_disambiguating_which_past_appointment_is_meant():
    # Reproduces a reported bug: patient booked two appointments (8am and 9am) via
    # chat, manually cancelled the 8am one outside the chat, then said "reschedule
    # this appointment to Monday" — the model offered a choice between BOTH the 8am
    # and 9am appointments from its memory of the two earlier booking cards, still
    # including the one that had already been cancelled.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert 'figuring out which appointment the patient means when they say' in prompt
    assert "drop any that come back already cancelled or" in prompt


def test_agent_prompt_forbids_freehand_slot_lists_even_when_repeating_earlier_options():
    # Reproduces a reported bug: the model composed its own bulleted list of "other
    # available slots" — each line including a raw slot_id UUID — instead of calling
    # get_department_availability and relaying its real card. Exactly the raw-slot_id
    # leak the DOCTOR_OPTIONS:: card system exists to prevent.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NEVER compose a list of appointment times or slots yourself in prose" in prompt
    assert "do NOT retype the list from" in prompt


def test_agent_prompt_forbids_asking_patient_for_appointment_id():
    # Reproduces a reported bug: the model described the patient's existing
    # appointment earlier in the conversation (from a get_my_appointments call), but
    # when it came time to actually call reschedule_appointment it asked the patient
    # "could you please provide that ID?" instead of re-calling get_my_appointments to
    # resolve the appointment_id itself.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NEVER ask the patient to provide an appointment_id" in prompt
    assert "call get_my_appointments first" in prompt


def test_agent_prompt_forbids_writing_out_appointment_id_like_slot_id():
    # Companion fix to the appointment_id-in-tool-output change: now that
    # get_my_appointments' JSON includes a real appointment_id, the model must be told
    # not to leak it in prose, same as the existing slot_id rule.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "never write an appointment_id out in your reply text" in prompt


def test_agent_prompt_rejects_unrecognizable_department_strings_instead_of_guessing():
    # Reproduces a reported bug: patient replied "Cars dept" (not a real specialty,
    # not even a recognizable typo of one) and the model silently ignored it instead
    # of flagging it as not a real department.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "typo, or nonsense" in prompt
    assert "call get_department_availability with the department name exactly as the patient wrote it" in prompt


def test_agent_prompt_forbids_composing_the_not_found_department_list_from_context():
    # Reproduces a worse version of the same reported bug: for "geology dept" (not a
    # real department), the model composed its own "our active departments are..."
    # list from Retrieved context/memory instead of actually calling
    # get_department_availability — and that composed list turned out to be doctors'
    # SPECIALIZATION values, not real departments at all. The tool's own not-found
    # response (built from the real Department table) must be the only source.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "do NOT compose your own" in prompt
    assert "is the ONLY reliable source for" in prompt


def test_agent_prompt_requires_confirming_fuzzy_matched_doctor_names():
    # Reproduces a reported bug: patient wrote "Ali Baber", the model silently
    # "corrected" it to the real "Dr. Babar Ali" and proceeded, instead of confirming
    # the match first. The fix routes this through the find_doctors_by_name tool
    # (word-level, order-independent match) rather than context-only guessing.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NEVER treat a patient-typed doctor name as a confirmed match" in prompt
    assert "call find_doctors_by_name" in prompt
    assert "Did you mean Dr. Iqra" in prompt


def test_agent_prompt_requires_listing_multiple_name_matches():
    # Reproduces a reported bug: "dr raza iqra" (reversed word order for a real
    # "Dr. Iqra Raza") was fed straight into get_department_availability's
    # department_name argument instead of being resolved via a doctor-name search —
    # and when a typed name could match several real doctors, the bot must list all
    # of them and ask which one, not silently pick one.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "NEVER pass a person's name" in prompt
    assert "more than one match" in prompt
    assert "list every matching doctor with their department and ask" in prompt


def test_agent_prompt_flags_doctor_department_mismatch_instead_of_silent_swap():
    # Reproduces a reported bug: the model silently swapped in a named doctor's real
    # department without ever telling the patient their stated department was wrong.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "does NOT actually work in" in prompt
    assert "say plainly that this doctor isn't in the department they mentioned" in prompt


def test_agent_prompt_requires_exact_real_department_name_on_mismatch():
    # Reproduces a reported bug: the model told a patient "Dr. Iqra Raza... sees
    # patients for head and neck surgery (the Head and Neck Surgery department)" — a
    # composed descriptive phrase rather than the clinic's actual department name
    # (e.g. "ENT"), which would fail to match on the very next tool call.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "per the department_name-source rule above" in prompt


def test_agent_prompt_forbids_confusing_specialization_with_department():
    # Reproduces a worse version of the same reported bug: the model went on to
    # invent a whole fake "our active departments are..." list built entirely out of
    # doctors' SPECIALIZATION values (e.g. "Head and Neck Surgery", "Respiratory
    # Medicine", "Neonatology") rather than the clinic's real Department rows (e.g.
    # "ENT", "Pulmonology", "Pediatrics") — specialization and department are
    # different fields, and only a real tool result may be used as the department name
    # source, never Retrieved context prose or a doctor's specialization text.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "is NOT the same thing as their DEPARTMENT" in prompt
    assert "NEVER state a specialization as if it were the department" in prompt
    assert "never build a claimed \"active departments\" list out of" in prompt


def test_agent_prompt_major_weight_bearing_bone_broken_skips_path2_straight_to_path1():
    # Reproduces a reported bug: "my leg got broken" still went through a PATH 2
    # screening question instead of being treated as an immediate emergency.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "A MAJOR/WEIGHT-BEARING BONE STATED AS BROKEN SKIPS PATH 2 ENTIRELY" in prompt
    assert '"my leg is broken", "I broke my leg" — go straight to PATH 1 immediately, no screening question first' in prompt


def test_agent_prompt_requires_fresh_tool_call_for_yes_no_day_questions():
    # Reproduces a reported bug: "isnt there any doctor in this dept available on
    # tue?" got answered with a fabricated general claim ("cardiology doctors work
    # on Saturdays and Sundays") instead of a fresh get_department_availability call
    # for Tuesday — and that claim directly contradicted a Wednesday slot the same
    # conversation had already shown.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert '"isn\'t there anyone available on tue?", "is any doctor free on Tuesday?"' in prompt


def test_agent_prompt_forbids_asserting_doctor_schedule_patterns_from_memory():
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert 'NEVER assert which days a doctor "works", is "scheduled", or is generally' in prompt
    assert "you have no direct visibility into doctor shift patterns" in prompt


def test_agent_prompt_permits_general_knowledge_for_emergency_judgment():
    # Reported bug: "i got hit by a car and im bleeding very much" (a case the
    # regex gate now also catches, but the LLM-level backstop must independently
    # be unambiguous too) got the generic "tell me your main symptom" redirect
    # instead of an emergency recommendation — the model needs explicit permission
    # to reason about severity from its own general knowledge, not just retrieved
    # clinic context, and explicit confirmation this isn't a forbidden diagnosis.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "reason from your own general real-world medical/safety knowledge" in prompt
    assert "diagnoses nothing" in prompt


def test_agent_prompt_instructs_omitting_note_when_patient_named_the_department():
    # Reported bug: "book me with a cardiologist" (patient names the department
    # outright) was still getting the symptom-triage reasoning sentence ("Based on
    # what you've described, this sounds like something Cardiology should look
    # at") prepended, which is nonsensical when the patient did the routing
    # themselves. The prompt must explicitly instruct omitting `note` in that case.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "Omit `note`" in prompt
    assert "the patient already named the department or specialty themselves" in prompt


def test_agent_prompt_instructs_checking_both_departments_on_a_genuine_tie():
    # Reported gap: a symptom that plausibly fits two real departments about equally
    # well (e.g. dizziness with a racing heartbeat -> Cardiology or Neurology) was
    # silently routed to just one, with no mention of the other option at all. The
    # bot already has a mechanism for combining multiple get_department_availability
    # calls into one reply (used for explicit cross-department requests) — this rule
    # extends that same mechanism to a genuine symptom-routing tie.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "AMBIGUOUS BETWEEN MULTIPLE REAL DEPARTMENTS" in prompt
    assert "do NOT silently pick just one" in prompt
    assert "Call get_department_availability once for EACH plausible real department" in prompt
    assert "Reserve this for a genuine tie between real options" in prompt
    assert "they already told you X" in prompt


def test_current_date_is_injected_into_the_system_prompt():
    messages = llm._build_agent_messages("hello", "en", [], [])
    system_content = messages[0].content
    assert "Today's date is" in system_content
    assert llm._current_date_str() in system_content


# --- regression: a cross-department question must not stop after the first call ----


def test_get_department_availability_is_not_in_terminal_tools():
    # The core of the item-3 fix: this tool must NOT short-circuit the agent loop the
    # moment it's first called, or a cross-department sweep can never make a second
    # call.
    assert "get_department_availability" not in llm._TERMINAL_TOOLS
    assert "book_appointment" in llm._TERMINAL_TOOLS
    assert "reschedule_appointment" in llm._TERMINAL_TOOLS
    assert "cancel_appointment" in llm._TERMINAL_TOOLS


def test_cross_department_sweep_calls_the_tool_more_than_once_before_final_reply(monkeypatch):
    cardiology_tool_reply = "DOCTOR_OPTIONS::{\"department_name\": \"Cardiology\", \"doctors\": []}"
    dermatology_tool_reply = "DOCTOR_OPTIONS::{\"department_name\": \"Dermatology\", \"doctors\": []}"
    dept_tool = _FakeTool("get_department_availability")

    # invoke() results depend on which tool call is in flight, not the LLM
    # response queue — the tool itself returns whatever its .invoke() is stubbed to
    # for that call's args, but here it's simpler to have the fake tool return a
    # fixed value per call in sequence.
    tool_results = iter([cardiology_tool_reply, dermatology_tool_reply])

    def _invoke(args):
        dept_tool.calls.append(args)
        return next(tool_results)

    dept_tool.invoke = _invoke

    _patch_chat_groq(
        monkeypatch,
        {
            llm._MODEL_RETRY_SEQUENCE[0]: [
                _FakeToolCallResponse([_tool_call("get_department_availability", {"department_name": "Cardiology"}, "call-1")]),
                _FakeToolCallResponse([_tool_call("get_department_availability", {"department_name": "Dermatology"}, "call-2")]),
                _FakeFinalResponse("Here's what I found: Cardiology has no free slots right now, and neither does Dermatology."),
            ]
        },
    )

    result = llm.run_chat_agent(
        message="list every doctor across every department",
        language="en",
        context_chunks=[],
        history=[],
        tools=[dept_tool],
    )

    # The tool was called twice (once per department) before the loop accepted a
    # final plain-text reply with no further tool calls. The combined reply is
    # assembled in code from both raw results (see
    # app.services.chat_tools.combine_department_availability_results) — the fake
    # LLM's own freehand final-content response is discarded entirely, never
    # trusted as the reply once get_department_availability was called more than
    # once this turn.
    assert len(dept_tool.calls) == 2
    assert dept_tool.calls[0] == {"department_name": "Cardiology"}
    assert dept_tool.calls[1] == {"department_name": "Dermatology"}
    assert result.startswith("DEPARTMENT_LIST::")
    payload = json.loads(result[len("DEPARTMENT_LIST::"):])
    assert [d["department_name"] for d in payload["departments"]] == ["Cardiology", "Dermatology"]


def test_single_department_call_returns_the_tools_raw_result_never_the_models_paraphrase(monkeypatch):
    # Single-department question: still just one tool call. The fake model's "final"
    # response deliberately paraphrases instead of relaying verbatim (exactly the bug
    # that let a raw slot_id leak into prose) — _finalize_reply must discard that
    # paraphrase entirely and return the tool's own raw result untouched.
    card = "DOCTOR_OPTIONS::{\"department_name\": \"Cardiology\", \"doctors\": []}"
    dept_tool = _FakeTool("get_department_availability", result=card)

    _patch_chat_groq(
        monkeypatch,
        {
            llm._MODEL_RETRY_SEQUENCE[0]: [
                _FakeToolCallResponse([_tool_call("get_department_availability", {"department_name": "Cardiology"}, "call-1")]),
                _FakeFinalResponse("Cardiology has no doctors with free slots at slot_id abc-123 right now."),
            ]
        },
    )

    result = llm.run_chat_agent(
        message="who's available in Cardiology",
        language="en",
        context_chunks=[],
        history=[],
        tools=[dept_tool],
    )

    assert len(dept_tool.calls) == 1
    assert result == card
    assert "slot_id" not in result


def test_terminal_tool_short_circuits_immediately(monkeypatch):
    booking_card = "BOOKING_CONFIRMED::{\"doctor_name\": \"Dr. Jane\"}"
    book_tool = _FakeTool("book_appointment", result=booking_card)

    _patch_chat_groq(
        monkeypatch,
        {llm._MODEL_RETRY_SEQUENCE[0]: [_FakeToolCallResponse([_tool_call("book_appointment", {"slot_id": "abc"}, "call-1")])]},
    )

    result = llm.run_chat_agent(
        message="book that slot",
        language="en",
        context_chunks=[],
        history=[],
        tools=[book_tool],
    )

    # Terminal tools return their own result the moment they're called — no second
    # LLM round-trip needed (a second .invoke() on the primary model's fake would
    # raise StopIteration, proving a single call was enough).
    assert result == booking_card
    assert len(book_tool.calls) == 1


# --- token-reduction: conditional triage section (item 1 of the token-usage work) ---


def test_build_agent_messages_includes_triage_section_by_default():
    messages = llm._build_agent_messages("hi", "en", [], [])
    assert "SYMPTOM TRIAGE RULE" in messages[0].content
    assert "PATH 1" in messages[0].content


def test_build_agent_messages_omits_triage_section_when_not_symptom_related():
    messages = llm._build_agent_messages("hi", "en", [], [], include_triage=False)
    content = messages[0].content
    assert "SYMPTOM TRIAGE RULE" not in content
    assert "PATH 1" not in content
    # Everything else must still be present — this is purely a token-savings change,
    # never a behavior change for non-triage turns.
    assert "STRICT GROUNDING RULE" in content
    assert "TOOL USE RULES" in content
    assert "`note` ALSO doubles" in content


def test_omitting_triage_section_meaningfully_shrinks_the_prompt():
    with_triage = llm._build_agent_messages("hi", "en", [], [], include_triage=True)[0].content
    without_triage = llm._build_agent_messages("hi", "en", [], [], include_triage=False)[0].content
    assert len(without_triage) < len(with_triage) - 10000


def test_run_chat_agent_passes_include_triage_through_to_the_prompt(monkeypatch):
    captured = {}

    class _FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            captured["system_content"] = messages[0].content
            return _FakeToolCallResponse([])

    monkeypatch.setattr(llm, "ChatGroq", lambda **kwargs: _FakeLLM())

    llm.run_chat_agent(
        message="what are your hours",
        language="en",
        context_chunks=[],
        history=[],
        tools=[],
        include_triage=False,
    )

    assert "SYMPTOM TRIAGE RULE" not in captured["system_content"]


# --- cross-session patient memory digest --------------------------------------------


def test_build_messages_defaults_patient_memory_to_none_placeholder():
    messages = llm._build_messages("hi", "en", [], [])
    assert "PATIENT MEMORY" in messages[0].content
    assert "(none)" in messages[0].content


def test_build_messages_includes_patient_memory_text_when_given():
    messages = llm._build_messages("hi", "en", [], [], patient_memory="Patient has recurring migraines.")
    assert "Patient has recurring migraines." in messages[0].content


def test_build_agent_messages_defaults_patient_memory_to_none_placeholder():
    messages = llm._build_agent_messages("hi", "en", [], [])
    assert "PATIENT MEMORY" in messages[0].content


def test_build_agent_messages_includes_patient_memory_text_when_given():
    messages = llm._build_agent_messages("hi", "en", [], [], patient_memory="Patient has a penicillin allergy.")
    assert "Patient has a penicillin allergy." in messages[0].content


def test_agent_prompt_forbids_using_patient_memory_as_a_tool_argument_source():
    # The digest is background context only — department/doctor names must still only
    # ever come from a real tool result or Retrieved context, never from memory text
    # that could be stale or imprecise.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "never use it as a source for a tool call argument" in prompt


def test_run_chat_agent_passes_patient_memory_through_to_the_prompt(monkeypatch):
    captured = {}

    class _FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            captured["system_content"] = messages[0].content
            return _FakeToolCallResponse([])

    monkeypatch.setattr(llm, "ChatGroq", lambda **kwargs: _FakeLLM())

    llm.run_chat_agent(
        message="hi",
        language="en",
        context_chunks=[],
        history=[],
        tools=[],
        patient_memory="Patient previously described knee pain.",
    )

    assert "Patient previously described knee pain." in captured["system_content"]
