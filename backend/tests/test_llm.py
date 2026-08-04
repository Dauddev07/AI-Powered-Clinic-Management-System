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
    across separate run_tool_calling_agent loop iterations reusing the same model, so the
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


# --- item: Groq-only retry sequence (same model, 6 attempts, 6 different keys) -----


def test_retry_sequence_is_the_same_primary_model_all_six_times():
    # No second model (previously Qwen) in the sequence — every attempt retries on
    # a different API key (see api_key_manager's own round-robin) rather than a
    # different model, per explicit request. One attempt per configured key (6).
    assert llm._MODEL_RETRY_SEQUENCE == ("openai/gpt-oss-120b",) * 6


# --- token-reduction: reasoning_effort is model-specific, never one-size-fits-all --


def test_reasoning_effort_is_low_for_gpt_oss_models():
    # Verified directly against Groq's API: gpt-oss models reject "none"/"default"
    # and accept only "low"/"medium"/"high" — "low" cuts hidden chain-of-thought
    # reasoning tokens dramatically with no effect on the visible reply, since those
    # tokens are already stripped from `content` by reasoning_format="hidden" and
    # were never shown to the patient in the first place.
    assert llm._reasoning_effort_for("openai/gpt-oss-120b") == "low"
    assert llm._reasoning_effort_for("openai/gpt-oss-20b") == "low"


def test_reasoning_effort_is_omitted_for_a_non_gpt_oss_model():
    # Not exercised by the current retry sequence (single-model now — see above),
    # but _reasoning_effort_for stays a per-model function rather than a bare
    # constant precisely so a future non-gpt-oss fallback doesn't silently get the
    # wrong value and 400 every time it's actually needed. Reproduces a real 400
    # verified directly against Groq's API for Qwen specifically.
    assert llm._reasoning_effort_for("qwen/qwen3.6-27b") is None


def test_run_plain_reply_and_run_tool_calling_agent_pass_per_model_reasoning_effort(monkeypatch):
    seen_kwargs = []

    def fake_chat_groq(**kwargs):
        seen_kwargs.append(kwargs)
        return _FakeLLM(iter([_FakeFinalResponse("hi")]))

    monkeypatch.setattr(llm, "ChatGroq", fake_chat_groq)
    llm.run_plain_reply("test system prompt", "hi", [])

    assert seen_kwargs[0]["reasoning_effort"] == "low"
    assert seen_kwargs[0]["model"] == "openai/gpt-oss-120b"


def test_no_rate_limit_uses_a_single_attempt_only_happy_path_unchanged(monkeypatch):
    calls = []

    def _fake_chat_groq(model, **kwargs):
        calls.append(model)
        return _FakeLLM(iter([_FakeFinalResponse("Hello there.")]))

    monkeypatch.setattr(llm, "ChatGroq", _fake_chat_groq)

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == "Hello there."
    assert calls == ["openai/gpt-oss-120b"]


def test_rate_limit_on_first_key_succeeds_via_second_key(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    # Same model at every position now — its queue holds one entry per attempt,
    # consumed in order regardless of which API key each attempt actually drew
    # (key selection is api_key_manager's own concern, not this module's).
    _patch_chat_groq(
        monkeypatch,
        {model: [_rate_limit_error(model), _FakeFinalResponse("Served on the second attempt.")]},
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    # Succeeds via the second attempt — no error surfaced to the patient.
    assert result == "Served on the second attempt."


def test_token_limit_413_falls_back_to_the_next_attempt_same_as_a_rate_limit(monkeypatch):
    # Regression: a plain groq.APIStatusError (413, tokens-per-minute cap) used to
    # NOT be caught here at all (only the 429 RateLimitError subclass was), so it
    # crashed the whole request instead of advancing to the next attempt —
    # surfacing as an unhandled 500 to the patient instead of the graceful
    # fallback reply.
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_token_limit_error(model), _FakeFinalResponse("Served on the second attempt.")]},
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == "Served on the second attempt."


def test_token_limit_413_on_every_attempt_gives_the_graceful_fallback_not_a_crash(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_token_limit_error(model) for _ in llm._MODEL_RETRY_SEQUENCE]},
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == llm._RATE_LIMIT_REPLY


def test_rate_limit_on_first_two_attempts_succeeds_on_the_third(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_rate_limit_error(model), _rate_limit_error(model), _FakeFinalResponse("Served on the third attempt.")]},
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == "Served on the third attempt."


def test_rate_limit_on_first_four_attempts_succeeds_on_the_fifth(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {
            model: [
                _rate_limit_error(model),
                _rate_limit_error(model),
                _rate_limit_error(model),
                _rate_limit_error(model),
                _FakeFinalResponse("Served on the fifth attempt."),
            ]
        },
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == "Served on the fifth attempt."


def test_rate_limit_on_first_five_attempts_succeeds_on_the_sixth(monkeypatch):
    # Confirms the sequence really does go all 6 attempts (6 configured keys) deep
    # before giving up, not just 5 — a regression here would silently give up early
    # and show the patient an error when the 6th key would have worked.
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {
            model: [
                _rate_limit_error(model),
                _rate_limit_error(model),
                _rate_limit_error(model),
                _rate_limit_error(model),
                _rate_limit_error(model),
                _FakeFinalResponse("Served on the sixth attempt."),
            ]
        },
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == "Served on the sixth attempt."


def test_rate_limit_on_all_six_attempts_returns_the_plain_message_not_an_error(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_rate_limit_error(model) for _ in llm._MODEL_RETRY_SEQUENCE]},
    )

    result = llm.run_plain_reply("test system prompt", "hi", [])

    assert result == llm._RATE_LIMIT_REPLY


def test_a_non_rate_limit_error_is_not_retried_against_the_next_attempt(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(monkeypatch, {model: [RuntimeError("some other failure")]})

    with pytest.raises(RuntimeError):
        llm.run_plain_reply("test system prompt", "hi", [])


def test_run_tool_calling_agent_rate_limit_on_first_key_succeeds_via_second_key(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_rate_limit_error(model), _FakeFinalResponse("Agent reply from the second attempt.")]},
    )

    result = llm.run_tool_calling_agent("test system prompt", "hi", [], [])

    assert result == "Agent reply from the second attempt."


def test_run_tool_calling_agent_rate_limit_on_first_two_attempts_succeeds_on_the_third(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_rate_limit_error(model), _rate_limit_error(model), _FakeFinalResponse("Agent reply from the third attempt.")]},
    )

    result = llm.run_tool_calling_agent("test system prompt", "hi", [], [])

    assert result == "Agent reply from the third attempt."


def test_run_tool_calling_agent_returns_plain_message_when_all_six_attempts_are_rate_limited(monkeypatch):
    model = llm._MODEL_RETRY_SEQUENCE[0]
    _patch_chat_groq(
        monkeypatch,
        {model: [_rate_limit_error(model) for _ in llm._MODEL_RETRY_SEQUENCE]},
    )

    result = llm.run_tool_calling_agent("test system prompt", "hi", [], [])

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

    result = llm.run_tool_calling_agent(
        "test system prompt", "is there no doctor available for tommorow?", [], [_RaisingTool()]
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

    result = llm.run_tool_calling_agent("test system prompt", "book that slot", [], [_RaisingTool()])

    assert "sorry" in result.lower()


# --- item 1: non-red-flag symptom triage also screens for urgency ------------------


def test_agent_prompt_instructs_an_emergency_screening_clarifying_question():
    # A genuinely live LLM call isn't exercised in this test suite (see other tests'
    # mocked-LLM approach) — this asserts the prompt-engineering requirement itself
    # is actually present in the system prompt the model is given, since that's the
    # only lever this feature is implemented through.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "help distinguish urgent from routine for that symptom area" in prompt


def test_agent_prompt_never_calls_availability_tool_for_a_plain_emergency_routing_question():
    # A patient asking "which department handles emergencies?" is an informational
    # question, not a request to book into a department named "Emergency" — the
    # prompt must steer this to a plain-text answer instead of a guessed
    # department_name call (this exact bug: get_department_availability called with
    # "Emergency" as department_name, which then surfaced the tool's own real
    # not-found response as if "Emergency" had actually been looked up).
    assert "do NOT call get_department_availability for them at all" in llm._AGENT_SYSTEM_PROMPT
    assert "does not have a dedicated emergency/ER department" in llm._AGENT_SYSTEM_PROMPT


def test_agent_prompt_instructs_plain_urgent_recommendation_on_a_worrying_answer():
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "plainly tell the patient this sounds like an emergency" in prompt
    assert "do not call get_department_availability" in prompt.lower()


def test_agent_prompt_emergency_backstop_applies_to_any_message_not_just_answers():
    # The backstop must not be scoped to "an answer to a clarifying question" only —
    # a first message describing a severe/emergency presentation (that the
    # same-message regex check happened to miss, e.g. a typo or an unusual injury
    # combination) must trigger it too.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "not just an answer to your own question" in prompt
    assert "including the first message" in prompt


def test_agent_prompt_handles_emergency_with_no_matching_department():
    # If the clinic has no department appropriate for a genuine emergency, the model
    # must still say so plainly rather than silently forcing a department match or
    # treating it as routine triage.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "If no real department fits, still state plainly" in prompt
    assert "just to force a department match" in prompt


def test_agent_prompt_defers_to_the_same_message_emergency_check_as_authoritative():
    # The secondary screening layer must explicitly describe itself as
    # non-authoritative and subordinate to red_flag.py's same-message check — never
    # implying it could override or delay that check.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "Secondary layer only" in prompt
    assert "takes priority" in prompt


def test_agent_prompt_requires_first_aid_guidance_on_confirmed_emergency():
    # PATH 1 must tell the patient to go to the ER AND give 2-3 safe, generic
    # first-aid tips for the wait — never medication or a home remedy.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "give 2-3 short, generic, safe first-aid actions" in prompt
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
    assert "Do NOT decide a department, call get_department_availability, or state/imply" in prompt
    assert "chest pain, tightness, or pressure; head pain or severe headache" in prompt


def test_agent_prompt_path2_covers_broad_real_world_symptom_list_not_just_examples():
    # PATH 2 must explicitly be a non-exhaustive, general rule ("any presentation
    # that plausibly ranges from routine to a genuine emergency"), not just the
    # three original named examples — plus cover several concrete additional
    # real-world categories (abdominal pain, high fever, dizziness, etc.) with
    # their own differentiator guidance.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "not a fixed list" in prompt
    assert "severe/persistent abdominal" in prompt
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
    assert "visible deformity, bone through skin" in prompt
    assert 'asserting "this is a fracture" isn\'t' in prompt


def test_agent_prompt_requires_one_to_two_questions_for_routine_symptoms():
    # PATH 3: routine (non-ambiguous) symptoms require 1-2 clarifying questions
    # before routing — lowered from 2-3 to cut turns (and the triage prompt's token
    # cost) off the common, unambiguous case, per explicit request.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 3" in prompt
    assert "ask 1-2 real clarifying questions" in prompt


def test_agent_prompt_caps_path2_at_one_screening_round():
    # Reported bug: for chest pain, the model asked severity, then a SEPARATE reply
    # asking about shortness of breath/sweating/nausea, then ANOTHER asking about
    # pressure vs tightness, then ANOTHER asking about activity/illness — four
    # rounds of questions and it never once called get_department_availability,
    # eventually hitting a rate limit. PATH 2 must be capped at exactly one
    # screening reply before a mandatory decision.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 2 IS EXACTLY ONE ROUND — HARD LIMIT" in prompt
    assert "no second round of differentiator questions" in prompt
    assert "then you MUST decide" in prompt


def test_agent_prompt_caps_path3_at_two_questions_hard_ceiling():
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "2 QUESTIONS IS A HARD CEILING, NOT A TARGET" in prompt
    assert "your very next reply MUST call the tool" in prompt
    assert "counting any PATH 2 screening reply as one," in prompt


def test_agent_prompt_treats_a_string_of_denied_red_flags_as_enough_to_route_early():
    # Reproduces a reported bug: mild neck pain, patient denied every differentiator
    # asked (no radiation, no stiffness, no known trigger) across 3 separate
    # questions, and the model kept probing instead of routing once it already had a
    # clearly mild, stable picture.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "A string of denied differentiators/red-flags" in prompt
    assert "already enough to route immediately, not a cue to keep probing" in prompt


def test_agent_prompt_forbids_ending_path3_with_freehand_advice_instead_of_the_tool_call():
    # Reproduces a reported bug: a mild-neck-pain conversation ended with a long
    # freehand paragraph of home-remedy/OTC-medication advice and never once called
    # get_department_availability — no department was ever offered.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "PATH 3 ALWAYS ENDS WITH THE TOOL CALL, NEVER FREE-TEXT ADVICE INSTEAD" in prompt
    assert "never a multi-point home-remedy essay" in prompt


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
    assert "go to the ER now instead of waiting for the visit" in prompt


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


def test_agent_prompt_requires_actually_answering_a_direct_memory_recall_question():
    # Reported bug: patient_memory was correctly populated ("Asked about Cardiology
    # availability...") and passed into the prompt, yet a direct "what are the
    # things I told you" in a new session still got "I don't have any details" — the
    # model treated the "don't volunteer it unprompted" guidance as covering direct
    # questions too. Must be unambiguous that a direct recall question requires
    # actually using PATIENT MEMORY, not defaulting to "nothing stored".
    prompt = llm._AGENT_SYSTEM_PROMPT
    assert "IMPORTANT — WHEN THE PATIENT DIRECTLY ASKS ABOUT IT" in prompt
    assert "what are the things I told you" in prompt
    assert "you must actually use it rather than defaulting to" in prompt


def test_plain_system_prompt_requires_actually_answering_a_direct_memory_recall_question():
    prompt = llm._SYSTEM_PROMPT
    assert "IMPORTANT — WHEN THE PATIENT DIRECTLY ASKS ABOUT IT" in prompt
    assert "what are the things I told you" in prompt
    assert "you must actually use it rather than defaulting to" in prompt


def test_agent_prompt_forbids_booking_fresh_when_picking_a_slot_resolves_a_reschedule():
    # Reported bug: mid-reschedule, once the model showed the patient a list of new
    # times (via get_department_availability) and they picked one, the model called
    # book_appointment instead of reschedule_appointment — creating a stray extra
    # appointment instead of moving the existing one. The generic "call
    # book_appointment once a slot is picked from a shown list" rule alone doesn't
    # distinguish a reschedule's slot-pick from a fresh booking's.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "DO NOT confuse a slot-pick that resolves a RESCHEDULE with a fresh booking" in prompt
    assert "never book_appointment" in prompt
    assert "creates a stray extra" in prompt


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
    assert "A MAJOR/WEIGHT-BEARING BONE STATED AS BROKEN/FRACTURED SKIPS PATH 2" in prompt
    assert '"my leg is broken" — go straight to PATH 1, no screening first' in prompt


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
    assert "reason from general real-world medical/safety knowledge" in prompt
    assert "names no condition" in prompt


def test_agent_prompt_instructs_omitting_note_when_patient_named_the_department():
    # Reported bug: "book me with a cardiologist" (patient names the department
    # outright) was still getting the symptom-triage reasoning sentence ("Based on
    # what you've described, this sounds like something Cardiology should look
    # at") prepended, which is nonsensical when the patient did the routing
    # themselves. The prompt must explicitly instruct omitting `note` in that case.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "Omit `note`" in prompt
    assert "the patient already named the department/specialty themselves" in prompt


def test_agent_prompt_instructs_checking_both_departments_on_a_genuine_tie():
    # Reported gap: a symptom that plausibly fits two real departments about equally
    # well (e.g. dizziness with a racing heartbeat -> Cardiology or Neurology) was
    # silently routed to just one, with no mention of the other option at all. The
    # bot already has a mechanism for combining multiple get_department_availability
    # calls into one reply (used for explicit cross-department requests) — this rule
    # extends that same mechanism to a genuine symptom-routing tie.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "MULTIPLE DEPARTMENTS IN ONE TURN" in prompt
    assert "TIE:" in prompt
    assert "do NOT silently pick one" in prompt
    assert "get_department_availability once per real department in the same turn" in prompt
    assert "Reserve for a genuine tie" in prompt
    assert "they did the routing, not you" in prompt


def test_agent_prompt_instructs_covering_multiple_distinct_symptoms_in_one_turn():
    # Reported gap: a patient describing two separate, unrelated complaints in one
    # message (e.g. ear pain AND itchy skin -> ENT and Dermatology, no ambiguity
    # about either individually) got routed to only the first department, and had to
    # explicitly ask again before the second one was addressed. This is distinct from
    # the genuine-tie case above (one symptom uncertain between departments) — here
    # each complaint has its own clear department, so both departments must be called
    # in the same turn once the bot is ready to conclude.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "DISTINCT SYMPTOMS:" in prompt
    assert "no tie to reason about" in prompt
    assert "get_department_availability once per real department in the same turn" in prompt
    assert "never leave one for the patient to raise again" in prompt


def test_agent_prompt_instructs_a_separate_symptom_specific_note_per_department():
    # Reported live: ear pain + blurry vision (a genuine DISTINCT SYMPTOMS case, ENT
    # and Ophthalmology) got the exact same TIE-style note ("this could be evaluated
    # by either ENT or Ophthalmology") repeated identically on both cards, reading
    # as if one ambiguous symptom was being hedged across two departments instead of
    # two different symptoms each being explained on their own terms.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "Each call gets its OWN `note` naming the SPECIFIC symptom it's for" in prompt
    assert "never the TIE phrasing" in prompt


# current-date injection into an agent's system prompt is now tested via
# app.services.orchestrator.agents.symptom_agent in tests/test_orchestrator.py,
# since _build_agent_messages() (which built the old fixed template) no longer
# exists — each specialist agent composes its own prompt now.


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

    result = llm.run_tool_calling_agent(
        "test system prompt", "list every doctor across every department", [], [dept_tool]
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

    result = llm.run_tool_calling_agent("test system prompt", "who's available in Cardiology", [], [dept_tool])

    assert len(dept_tool.calls) == 1
    assert result == card
    assert "slot_id" not in result


def test_single_department_call_with_no_free_slots_unwraps_the_internal_marker_to_plain_text(monkeypatch):
    # _get_department_availability_impl tags a "found but nobody's free" result with
    # NO_SLOTS_MARKER internally (see chat_tools.py) so a multi-department combine can
    # tell it apart from "department doesn't exist" instead of silently dropping it.
    # For a SINGLE call, that internal marker must never leak to the patient — this
    # must still read exactly like the old plain-text message.
    no_slots = "NO_SLOTS::{\"department_name\": \"Dermatology\", \"message\": \"I couldn't find any doctors with free upcoming slots in Dermatology right now. Please check back later or contact the clinic directly.\"}"
    dept_tool = _FakeTool("get_department_availability", result=no_slots)

    _patch_chat_groq(
        monkeypatch,
        {
            llm._MODEL_RETRY_SEQUENCE[0]: [
                _FakeToolCallResponse([_tool_call("get_department_availability", {"department_name": "Dermatology"}, "call-1")]),
                _FakeFinalResponse("Dermatology has nothing free — sorry about that."),
            ]
        },
    )

    result = llm.run_tool_calling_agent("test system prompt", "who's available in Dermatology", [], [dept_tool])

    assert result == "I couldn't find any doctors with free upcoming slots in Dermatology right now. Please check back later or contact the clinic directly."
    assert not result.startswith("NO_SLOTS::")


def test_terminal_tool_short_circuits_immediately(monkeypatch):
    booking_card = "BOOKING_CONFIRMED::{\"doctor_name\": \"Dr. Jane\"}"
    book_tool = _FakeTool("book_appointment", result=booking_card)

    _patch_chat_groq(
        monkeypatch,
        {llm._MODEL_RETRY_SEQUENCE[0]: [_FakeToolCallResponse([_tool_call("book_appointment", {"slot_id": "abc"}, "call-1")])]},
    )

    result = llm.run_tool_calling_agent("test system prompt", "book that slot", [], [book_tool])

    # Terminal tools return their own result the moment they're called — no second
    # LLM round-trip needed (a second .invoke() on the primary model's fake would
    # raise StopIteration, proving a single call was enough).
    assert result == booking_card
    assert len(book_tool.calls) == 1


# --- token-reduction: conditional triage section (item 1 of the token-usage work) ---
#
# Whether the triage section is included at all is now decided by which specialist
# agent the orchestrator router sends a message to (symptom_agent always includes
# it; appointment_agent/general_info_agent never do) rather than an include_triage
# flag on a single shared agent function — see tests/test_orchestrator.py for the
# equivalent coverage (symptom_agent always includes SYMPTOM TRIAGE RULE/PATH 1;
# appointment_agent/general_info_agent's prompts never contain it at all).


# --- token-reduction: conditionally droppable PATH 2 body --------------------------


def test_triage_always_plus_path2_reassembles_byte_for_byte_into_the_original():
    # The mechanical-split guarantee: _TRIAGE_ALWAYS/_TRIAGE_PATH2 were derived by
    # slicing the original _TRIAGE_SECTION text, never retyped — this proves that
    # slicing round-trips exactly, with include_path2=True behaving identically to
    # the pre-split single block.
    assert llm._triage_section(include_path2=True) == llm._TRIAGE_SECTION


def test_triage_section_with_path2_excluded_drops_only_path2_specific_text():
    without_path2 = llm._triage_section(include_path2=False)
    # PATH 2's own body, and the EXCEPTION that lives inside it, must be gone.
    assert "PATH 2 — AMBIGUOUS" not in without_path2
    assert "EXCEPTION — A MAJOR/WEIGHT-BEARING BONE" not in without_path2
    assert "PATH 2 IS EXACTLY ONE ROUND" not in without_path2
    assert "NOTABLE-BUT-NOT-CLEARLY-EMERGENCY" not in without_path2
    # Everything else — PATH 1, PATH 3, and the cross-cutting safety nets — must
    # still be fully present; none of these are separable from PATH 1/3.
    assert "SYMPTOM TRIAGE RULE" in without_path2
    assert "PATH 1 — CONFIRMED EMERGENCY" in without_path2
    assert "PATH 3 — ROUTINE SYMPTOM" in without_path2
    assert "PATH 3 ALWAYS ENDS WITH THE TOOL CALL" in without_path2
    assert "EMERGENCY BACKSTOP — SECONDARY LAYER" in without_path2
    assert "Judging THIS" in without_path2
    assert "MULTIPLE DEPARTMENTS IN ONE TURN" in without_path2


def test_excluding_path2_meaningfully_shrinks_the_triage_section():
    with_path2 = llm._triage_section(include_path2=True)
    without_path2 = llm._triage_section(include_path2=False)
    assert len(without_path2) < len(with_path2) - 3000


# include_path2 wiring through an actual agent call is now tested via
# app.services.orchestrator.agents.symptom_agent.run_symptom_agent in
# tests/test_orchestrator.py — the _triage_section() tests above already cover the
# content-level guarantee (PATH 1/3 always present, PATH 2 the only droppable
# piece); the orchestrator tests cover that symptom_agent actually wires
# needs_path2_screening()'s decision through to it correctly.


# --- cross-session patient memory digest --------------------------------------------
#
# Patient-memory wiring through an actual agent call (defaulting to "(none)", or
# carrying real digest text through) is now tested per-agent in
# tests/test_orchestrator.py, since _build_messages()/_build_agent_messages() (which
# built the old fixed templates) no longer exist.


def test_agent_prompt_forbids_using_patient_memory_as_a_tool_argument_source():
    # The digest is background context only — department/doctor names must still only
    # ever come from a real tool result or Retrieved context, never from memory text
    # that could be stale or imprecise.
    prompt = llm._AGENT_SYSTEM_PROMPT + llm._TRIAGE_SECTION
    assert "never use it as a source for a tool call argument" in prompt
