"""Server-side backstop enforcing "the bot never names a condition."

The triage system prompt (app/services/llm.py) already instructs the model never to
diagnose or speculate about a specific condition — but a prompt instruction alone is
not a guarantee, so every reply that goes through the triage/agent path is scanned
here before it reaches the patient. This is a heuristic safety net, not a medical
NLU system: it catches the phrasing patterns a model slips into when it tries to
name or strongly imply a diagnosis, and swaps the reply for a safe, on-persona
redirect back to department-routing when one is found.
"""
import re

# Words/phrases this bot's OWN normal, non-diagnostic replies routinely produce right
# after a "you have"/"this is"/"it's probably" style phrase — appointment summaries
# ("you have 2 upcoming appointments"), scheduling confirmations, and plain
# pleasantries ("this is a great question"). Without excluding these, the broad
# diagnostic-phrase patterns below false-fire on completely ordinary bot replies, not
# just on an actual diagnosis attempt. Not an attempt at exhaustive medical NLU —
# just enough to stop the demonstrated false positives in this bot's own reply
# vocabulary.
_BENIGN_CONTINUATIONS = (
    "appointment", "appointments", "slot", "slots", "visit", "visits", "history",
    "question", "questions", "option", "options", "choice", "choices", "plan", "plans",
    "idea", "ideas", "point", "points", "department", "departments", "doctor", "doctors",
    "note", "message", "messages", "chat", "session", "sessions", "time", "times",
    "good", "great", "nice", "fine", "busy", "quiet", "available", "open", "closed",
    "booked", "confirmed", "cancelled", "rescheduled", "possible", "helpful", "useful",
    "correct", "right", "wrong", "true", "false", "upcoming", "reminder", "reminders",
    # Generic hedging/lead-in words the triage agent's own clarifying questions
    # routinely use before getting to the actual question ("this is a common
    # combination of symptoms...", "it's probably nothing serious...", "this could
    # be a number of things...") — none of these name a condition on their own, but
    # without excluding them the same broad pattern that catches "you might have
    # the flu" also caught these and silently replaced a perfectly fine clarifying
    # question with the generic safe-redirect text.
    "common", "typical", "normal", "usual", "routine", "general", "mild", "minor",
    "nothing", "something", "anything", "number", "few", "couple", "several", "many",
    "various", "different", "range", "variety", "combination", "reason", "reasons",
    "possibility", "possibilities", "concern", "concerns", "way", "ways",
    # Safety/urgency vocabulary the triage rules explicitly REQUIRE the model to use
    # verbatim for a real emergency escalation ("this sounds like an emergency",
    # "this could be a medical emergency, please seek immediate care") — the prompt
    # itself says this phrasing "names no condition and diagnoses nothing," but
    # without these words listed here the same broad "this sounds like X" pattern
    # caught the escalation sentence itself, silently swapping a required urgent-care
    # warning for a useless generic redirect — the worst possible failure mode for a
    # safety feature. Also covers broad, non-disease injury descriptors the prompt's
    # own examples explicitly permit as hedged reassurance/reasoning ("this sounds
    # like it may just be a mild muscle strain", "this could be a sprain or a
    # fracture, let me ask a couple more questions") — these describe an injury type,
    # not a specific disease/condition diagnosis, which is what the no-diagnosis rule
    # actually targets (an ordinary disease name like "flu" or "appendicitis" is
    # still NOT in this list, and is still caught).
    "emergency", "emergencies", "urgent", "urgently", "medical", "serious",
    "seriously", "attention", "care", "sprain", "sprains", "fracture", "fractures",
    "strain", "strains", "injury", "injuries", "issue", "issues",
)
_BENIGN_LOOKAHEAD = r"(?!.{0,20}\b(?:" + "|".join(_BENIGN_CONTINUATIONS) + r"|\d)\b)"

_DIAGNOSTIC_PHRASE_RE = re.compile(
    r"\b("
    # (?<!do )(?<!did ) excludes the question forms "do/did you have ...?" — the
    # triage agent's own clarifying questions routinely ask "do you have any other
    # symptoms such as X" to screen for differentiators, which is the opposite of
    # asserting a diagnosis. Without this, that ordinary screening question false-fired
    # here just as often as an actual "you have condition X" assertion would.
    rf"(?<!do )(?<!did )you (?:have|might have|may have|could have|likely have|probably have) {_BENIGN_LOOKAHEAD}\w+"
    r"|you'?re (?:suffering from|dealing with)\b"
    rf"|this (?:is|sounds like|looks like|could be|appears to be) {_BENIGN_LOOKAHEAD}(?:a |an )?\w+"
    rf"|it(?:'s| is) (?:probably|likely|most likely) {_BENIGN_LOOKAHEAD}(?:a |an )?\w+"
    r"|(?:sounds|looks) like (?:a |an )?\w+ (?:infection|condition|disease|disorder)"
    r"|diagnos\w*"
    r"|you (?:are|might be|may be|could be) experiencing (?:a |an )?\w+ (?:attack|infection|disease)"
    r")\b",
    re.IGNORECASE,
)

SAFE_REDIRECT_EN = (
    "I'm not able to diagnose conditions, but I can help point you to the right "
    "department. Could you tell me a bit more about your main symptom?"
)

SAFE_REDIRECT_UR = (
    "میں کسی بیماری کی تشخیص نہیں کر سکتا، لیکن میں آپ کو صحیح شعبے کی طرف رہنمائی کر سکتا ہوں۔ "
    "کیا آپ اپنی بنیادی علامت کے بارے میں تھوڑی مزید تفصیل بتا سکتے ہیں؟"
)


def violates_no_diagnosis_rule(reply: str) -> bool:
    return bool(_DIAGNOSTIC_PHRASE_RE.search(reply))


def enforce_no_diagnosis(reply: str, language: str) -> str:
    """Returns `reply` unchanged unless it trips the diagnostic-phrasing pattern, in
    which case it is replaced outright with a safe, deterministic redirect — never
    patched or rephrased, since a partial edit could still leave a named condition
    in place."""
    if violates_no_diagnosis_rule(reply):
        return SAFE_REDIRECT_UR if language == "ur" else SAFE_REDIRECT_EN
    return reply
