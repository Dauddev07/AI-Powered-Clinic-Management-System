"""Classifies an incoming chat message into one of three intents:

- knowledge_seeking: a real question about the clinic, symptoms, departments,
  doctors, booking, etc. — must go through the retrieval/grounding gate.
- conversational: greetings, thanks, small talk, off-topic statements — gets a
  direct, natural LLM reply with no KB grounding needed.
- personal_recall: a question about something the patient themselves said earlier
  in the conversation ("what is my name?", "what did I just tell you?") — has
  nothing to do with the KB, so retrieval would only ever produce the fixed
  fallback even though the real answer is already sitting in conversation history.
  Skips retrieval exactly like conversational does, for the same reason: there is
  no clinic/medical claim to ground, just history already in the prompt.

This exists purely to stop plain small talk and self-referential recall questions
from hitting the same fixed fallback message as a genuine out-of-scope question — it
must never become a way for a real clinical/factual question (or arbitrary trivia
like "what's the capital of France?") to skip grounding. Every path here defaults to
"knowledge_seeking" whenever there's genuine ambiguity or a failure, so the existing
anti-hallucination behavior for real questions is never put at risk by this
classifier being wrong or unavailable.
"""
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.api_keys import api_key_manager
from app.core.config import settings
from app.services.chat_markers import (
    BOOKING_MARKER,
    DEPARTMENT_LIST_MARKER,
    DOCTOR_DISAMBIGUATION_MARKER,
    DOCTOR_OPTIONS_MARKER,
)

logger = logging.getLogger(__name__)

KNOWLEDGE_SEEKING = "knowledge_seeking"
CONVERSATIONAL = "conversational"
PERSONAL_RECALL = "personal_recall"

# Exact/near-exact small talk — matched only after stripping trailing punctuation, so
# this stays a fast, cheap path for the overwhelming majority of greetings/thanks/
# acknowledgments without ever needing the LLM fallback below.
_CONVERSATIONAL_PHRASES = frozenset({
    "hi", "hello", "hey", "hiya", "yo", "hi there", "hello there",
    "thanks", "thank you", "thanks a lot", "thank you so much", "many thanks",
    "bye", "goodbye", "see you", "see ya", "take care",
    "ok", "okay", "sure", "cool", "great", "nice", "awesome", "perfect", "fine",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "what's up", "whats up",
    "lol", "haha", "no problem", "np", "alright", "got it", "understood",
    "sounds good", "you're welcome", "youre welcome", "welcome",
})

# Self-referential recall questions — the patient asking about something *they*
# said, not asking the clinic/assistant a new question. Deliberately narrow and
# specific (unlike the broad, over-inclusive knowledge-seeking keyword list below):
# a false positive here would let a real question skip grounding, so this only
# matches clearly self-referential phrasing, not just any question with "I" in it.
#
# The "recall verb" group + flexible "what ... did/have ... i/we ... VERB" pattern
# below replaced a growing list of individually-enumerated fixed phrases after this
# same gap kept recurring for new phrasings the fixed list never anticipated
# ("what have i described to u?", then "what i have discussed with u", then "what
# info did i tell u" — each one a slightly different word order or an extra filler
# word the old list didn't happen to include). Rather than add another one-off
# phrase every time, this matches ANY "what ... (did|have) ... (i|we) ... <recall
# verb>" shape regardless of word order (auxiliary-first "what have I discussed" or
# subject-first "what I have discussed") or filler words in between ("what all did
# I tell you", "what info did I tell u") — bounded gaps keep it from drifting into
# an unrelated sentence, and requiring a past-tense-shaped auxiliary (did/have)
# keeps it from matching a prospective question like "what should I tell the
# doctor" (no did/have there at all, so it can't match).
_RECALL_VERBS = r"(?:tell|told|say|said|share|shared|mention|mentioned|describe|described|discuss|discussed|talk|talked)"

_PERSONAL_RECALL_RE = re.compile(
    r"\b("
    r"what(?:'s|s| is) my name"
    r"|do you remember (?:what|that) i"
    r"|what do you (?:know|remember) about me"
    r"|what did you (?:say|tell) (?:i|me)"
    r"|remind me what i (?:said|told|shared|described|discussed|talked about)"
    r"|what are the things i (?:told|said|shared|mentioned|described|discussed)"
    rf"|what\b.{{0,25}}\b(?:did\b.{{0,10}}\b(?:i|we)|have\b.{{0,10}}\b(?:i|we)|(?:i|we)\b.{{0,10}}\bhave)\b.{{0,20}}\b{_RECALL_VERBS}"
    r")\b",
    re.IGNORECASE,
)

# A leading question word is a strong knowledge-seeking signal on its own, cheaply
# ruling out most real questions before any keyword lookup is needed.
_QUESTION_WORD_RE = re.compile(
    r"^\s*(what|when|where|who|why|how|which|can|could|would|is|are|will|do|does|did|should|may|might)\b",
    re.IGNORECASE,
)

# Clinic-logistics vocabulary — hours, location, fees, booking mechanics. Kept
# separate from _SYMPTOM_KEYWORDS below (rather than one flat list) so
# is_symptom_message() can ask a narrower question than "is this knowledge-seeking
# at all": whether the message is specifically about a symptom, not just any
# clinic-related statement.
_LOGISTICS_KEYWORDS = frozenset({
    "clinic", "hospital", "doctor", "doctors", "department", "departments",
    "appointment", "appointments", "book", "booking", "schedule", "slot", "slots",
    "hour", "hours", "timing", "timings", "open", "close", "closing",
    "location", "address", "direction", "directions", "parking",
    "fee", "fees", "price", "cost", "charge", "charges", "insurance",
    "policy", "policies", "contact", "phone", "email", "holiday", "holidays", "weekend",
})

# Common symptom/complaint vocabulary — used both by the broad knowledge-seeking
# heuristic below and by is_symptom_message() to decide chat.py's retrieval routing
# (real department-list context instead of medical_kb retrieval, which no longer
# exists). English only — this system is English-specific, Roman Urdu is out of
# scope (same as app.services.language, which only detects English vs. real Urdu
# script, never Roman Urdu).
_SYMPTOM_KEYWORDS = frozenset({
    "pain", "ache", "aches", "aching", "fever", "cough", "symptom", "symptoms",
    "headache", "cold", "flu", "medicine", "medication", "prescription", "treatment",
    "diagnosis", "sick", "ill", "illness", "hurt", "hurts", "injury", "injured",
    # "diziness" (missing the second "z") reported live: this typo matched no
    # keyword at all, so the message fell through to GENERAL_INFO/the plain
    # freehand-LLM agent instead of symptom_agent's PATH 1/2/3 triage flow, which
    # asks 1-2 clarifying questions before ever naming a department — the plain
    # agent has no such guarantee at all and just answered with generic advice.
    "dizzy", "dizziness", "diziness", "nausea", "vomit", "vomiting", "rash", "allergy", "allergic",
    # Fractures/sprains/soft-tissue trauma — the gap that let "my hand got broken"
    # slip through undetected entirely.
    "broken", "break", "fracture", "fractured", "sprain", "sprained", "twisted",
    "dislocated", "dislocation",
    # Swelling/bruising/wounds
    "swollen", "swelling", "bruise", "bruised", "bleeding", "bled", "cut", "wound",
    "wounded", "gash", "blister", "blisters",
    # Sensation/weakness
    "numb", "numbness", "tingling", "weak", "weakness", "fatigue", "faint", "fainted",
    "stiff", "stiffness",
    # Respiratory/circulatory
    "breathless", "wheeze", "wheezing", "palpitations", "chills", "sweating",
    # Digestive
    "diarrhea", "diarrhoea", "constipation", "cramp", "cramps", "bloating",
    # Skin/infection
    "infection", "infected", "itchy", "itching", "sting", "stung", "bite", "bitten",
    "burn", "burned", "burning",
    # Common named complaints
    "migraine", "sore", "toothache", "earache", "backache", "stomachache", "nosebleed",
    "lump", "bump", "discharge",
    # Self-diagnosis claims ("i have a brain tumor", "i think i have cancer") — the
    # patient naming a suspected condition outright rather than describing raw
    # symptoms. Reported live: "i have brain tumor" had no keyword hit at all, so it
    # fell through to GENERAL_INFO/the plain-KB agent instead of the symptom agent's
    # concise department-card path, producing a long free-text breakdown (doctor
    # schedule + booking mechanics + reschedule policy) instead of a short redirect.
    "tumor", "tumour", "cancer", "diabetes", "diabetic",
    # Reported live: "i am also having skin related issues as well" — bare "skin"
    # matched no keyword here at all, so is_symptom_message returned False and
    # router.py's Rule 1.8 (a message that itself plainly states a symptom must
    # override stale booking continuity) never fired, silently misrouting a
    # genuinely new symptom to appointment_agent (no screening logic at all)
    # instead of symptom_agent. This exposed a much larger, systemic gap: this
    # list and symptom_hints.SYMPTOM_DEPARTMENT_HINTS (the department-hinting
    # table) had drifted apart — 119 words that table already treats as real
    # symptom vocabulary were never added here. Synced in one pass rather than
    # patching one word at a time as each is separately reported. Deliberately
    # EXCLUDES a handful of symptom_hints.py words that only mean something
    # combined with another word there (co-occurrence-gated), not standalone:
    # "weight" (bare "weight" is not itself symptom-indicative — see
    # _LIMB_JOINT_INJURY_SIGNAL_WORDS' own docstring), the low-mood persistence
    # words ("days", "weeks", "always", "lately", "chronic", "chronically",
    # "constantly", "ongoing", "persistent", "persistently" — pure duration
    # modifiers with zero symptom meaning alone), and "hearing"/"vision"/"mental"
    # (too ambiguous standalone in everyday English — "I'm hearing good things",
    # "my vision for this project", "that's mental!" — a false positive here has
    # a real cost now that Rule 1.8 depends on it, unlike this function's
    # original lower-stakes KB-routing use).
    "abdomen", "abdominal", "acne", "ankle", "anxiety", "arm", "arms", "asthma",
    "back", "boil", "boils", "bone", "brain", "breathing", "bruising", "calf",
    "cardiac", "chest", "chew", "chewing", "crying", "dental", "depressed",
    "depression", "digestive", "ear", "eczema", "elbow", "elbows", "eye", "eyes",
    "fell", "groin", "hand", "hands", "headaches", "heart", "hip", "hips",
    "hives", "hoarse", "hoarseness", "hopeless", "hopelessness", "hurting",
    "hypertension", "insomnia", "jaw", "joint", "joints", "kidney", "kidneys",
    "knee", "leg", "legs", "lightheaded", "lightheadedness", "lung", "malaise",
    "menstrual", "menstruation", "migraines", "mole", "moles", "nasal", "neck",
    "necks", "nose",
    "panic", "paralysis", "paralyzed", "pelvic", "period", "periods", "pimple",
    "pimples", "pregnancy", "pregnant", "redness", "respiratory", "sad",
    "sadness", "scrotal", "scrotum", "seizure", "seizures", "shoulder",
    "shoulders", "sinus", "sinusitis", "skin", "spine", "stomach", "stress",
    "stressed", "sugar", "teeth", "teeths", "testicle", "testicles",
    "testicular", "testies", "thigh", "thirst", "thirsty", "throat", "tooth",
    "tremor", "tremors", "urinary", "urinate", "urinating", "urination",
    "urine", "vaginal", "vertigo", "wart", "warts", "wrist", "wrists",
})

# Deliberately broad and erring toward false positives (routing more things to
# knowledge_seeking than strictly necessary) per this module's fail-safe design.
_KNOWLEDGE_KEYWORDS = _LOGISTICS_KEYWORDS | _SYMPTOM_KEYWORDS


# PATH 2's own named-symptom list (llm.py's _TRIAGE_SECTION) — mirrored here only
# to decide whether PATH 2's body is worth sending, deliberately broad/over-inclusive
# since a false positive here just means PATH 2 stays included (the safe direction),
# while a false negative would drop real screening guidance for a symptom that
# needed it.
_PATH2_SYMPTOM_KEYWORDS = frozenset({
    "chest", "dizziness", "diziness", "dizzy", "lightheaded", "lightheadedness", "faint",
    "fainting", "fainted", "vomiting", "diarrhea", "diarrhoea", "sprain", "sprained",
    "fracture", "fractured", "burn", "burned", "burning", "palpitations", "bite",
    "bitten", "pregnant", "pregnancy",
})

_PATH2_SYMPTOM_PHRASES = (
    "chest pain", "chest tightness", "chest pressure", "head pain", "severe headache",
    "broken bone", "suspected fracture", "abdominal pain", "stomach pain",
    "high fever", "persistent fever", "back pain", "deep cut", "vision change",
    "vision loss", "vision changes", "ear pain", "tooth pain", "racing heart",
    "irregular heart", "racing heartbeat", "irregular heartbeat",
)

# Reported live: "i am having pain in stomach" did NOT trigger PATH 2 screening and
# routed straight to General Medicine with slots shown — _PATH2_SYMPTOM_PHRASES only
# matches the fixed word order "stomach pain", not "pain in stomach", which is
# exactly how a lot of patients actually phrase it. A fixed-phrase substring check is
# inherently order-dependent; this is the order-independent backstop: any message
# that mentions "pain" as its own word AND names one of these body parts, in either
# order and with words in between, still needs PATH 2's screening question.
_PATH2_PAIN_BODY_PARTS = frozenset({
    "stomach", "abdominal", "abdomen", "chest", "head", "back", "ear", "tooth", "teeth", "neck",
})

# The one exception that skips PATH 2's screening question entirely (see
# _TRIAGE_SECTION's EXCEPTION paragraph, which physically lives inside PATH 2's
# body) — a message matching this must keep PATH 2 included, since the EXCEPTION
# text itself is what tells the model to route straight to PATH 1 for this case.
_MAJOR_BONE_FRACTURE_RE = re.compile(
    r"\b(leg|hip|thigh|pelvis|spine)\b.{0,20}\b(broken|broke|fracture[d]?)\b"
    r"|\b(broken|broke|fracture[d]?)\b.{0,20}\b(leg|hip|thigh|pelvis|spine)\b",
    re.IGNORECASE,
)


def needs_path2_screening(message: str, history=None) -> bool:
    """True when PATH 2's body (screening question, named-symptom list, the
    major-bone EXCEPTION, the one-round limit) should be included this turn — see
    app.services.llm.run_chat_agent's include_path2. Deliberately biased toward
    True: only returns False for a confident, clean routine-symptom match with no
    plausible screening need at all — every other case, including genuine
    ambiguity, defaults to keeping PATH 2 included."""
    lowered = message.lower()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    if words & _PATH2_SYMPTOM_KEYWORDS:
        return True
    if any(phrase in lowered for phrase in _PATH2_SYMPTOM_PHRASES):
        return True
    if "pain" in words and words & _PATH2_PAIN_BODY_PARTS:
        return True
    if _preceding_assistant_turn_looks_like_a_question(history):
        return True
    if _MAJOR_BONE_FRACTURE_RE.search(message):
        return True
    return False


# Reported live: "what are the available depts" / "show me available depts" wasn't
# recognized at all — it contains "available", one of _BOOKING_ACTION_KEYWORDS below,
# so the router sent it to appointment_agent (which has no tool that can list every
# department — get_department_availability requires one specific department_name),
# leaving nothing that could actually answer it. This is a request for the full
# department list, not availability for one named department or a booking action —
# a distinct case that needs its own deterministic detection and short-circuit (see
# app.services.orchestrator.router and app.services.orchestrator.agents.
# general_info_agent), the same "answer from real DB rows in code, never trust the
# model to compose it" principle already used for personal recall above.
#
# Requires the PLURAL "departments"/"depts" specifically — "what department is Dr.
# Smith in" (singular, about one specific doctor) must NOT match this, since that's
# an entirely different question this pattern has no business answering. The
# "all/every" alternative allows the singular too ("every department" is correct,
# unambiguous English for the full list despite the singular noun).
# NOTE: "dept" is NOT a prefix of "department" ("dep-a-rtment" vs "dep-t") — they're
# matched as a genuine alternation, not concatenated as one shared stem.
_DEPT_WORD = r"(?:dept|department)s"
_DEPT_WORD_ANY = r"(?:dept|department)s?"

_DEPARTMENT_LIST_RE = re.compile(
    rf"\b("
    rf"(?:what|which)\s+(?:are\s+)?(?:the\s+)?(?:available\s+)?{_DEPT_WORD}\b"
    rf"|show\s+(?:me\s+)?(?:the\s+)?(?:available\s+)?{_DEPT_WORD}\b"
    rf"|list\s+(?:of\s+)?(?:the\s+)?(?:available\s+)?{_DEPT_WORD}\b"
    rf"|(?:all|every)\s+{_DEPT_WORD_ANY}\b"
    rf"|available\s+{_DEPT_WORD}\b"
    rf")",
    re.IGNORECASE,
)


def is_department_list_request(message: str) -> bool:
    """True when the patient is asking for the full list of departments the clinic
    has, not asking about one specific named department/doctor. See the module-level
    comment above _DEPARTMENT_LIST_RE for why plural is required."""
    return bool(_DEPARTMENT_LIST_RE.search(message.strip()))


# Reported live: a patient described fever + body aches (routed correctly to
# General Medicine), then asked "isn't neurologist a better idea?" and "what do you
# think based on my symptoms" — these got shown Neurology's availability directly (no
# reasoning at all) and then a free-text reply that HALLUCINATED a symptom ("ear
# pain") never actually mentioned, since nothing recognized these as the patient
# asking for a grounded recommendation rather than a plain booking action. Detects
# that shape of message so app.services.orchestrator.router can route it to
# symptom_agent's deterministic recommendation short-circuit (real symptom-to-
# department mapping, never model-invented reasoning) regardless of what card was
# shown last turn.
_RECOMMENDATION_RE = re.compile(
    r"("
    r"what do (?:you|u) (?:think|recommend|suggest)"
    r"|what would (?:you|u) (?:think|recommend|suggest)"
    r"|(?:your|ur) recommendation"
    r"|which (?:department|dept|doctor|specialist|one) (?:is|would be|should i)"
    r"|what (?:department|dept) should i"
    r"|(?:isn'?t|is|wouldn'?t|would)\b.{0,25}\bbetter\b"
    # Reported live: "i think cardiologist can be a best fit for it?" used "best
    # fit" rather than "better", which the pattern above didn't cover at all.
    r"|\b(?:best|good|better) fit\b"
    r"|based on my symptoms"
    # Reported live: "find me a suitable dept" (after already describing a real
    # symptom) matched none of the above, so it fell through to the router's
    # generic fallback and landed on general_info_agent — which has no tool bound
    # at all to query real department availability, only KB retrieval — so it
    # just free-texted a department name with no real card, no doctors, no slots.
    # This phrasing is asking for the SAME grounded recommendation as "what do you
    # think" above, just worded as a request rather than a question.
    r"|find (?:me )?(?:a|an|the)?\s*(?:suitable|appropriate|right|good)?\s*(?:department|dept)\b"
    r")",
    re.IGNORECASE,
)


def is_department_recommendation_request(message: str) -> bool:
    """True when the patient is asking for a recommendation/opinion on which
    department fits their (already-described) symptoms — e.g. "what do you think",
    "isn't X a better idea", "which department should I see" — as opposed to a plain
    booking action ("book me with Dr. X") or an initial symptom description."""
    return bool(_RECOMMENDATION_RE.search(message.strip()))


# Reported live: after a Cardiology recommendation, "so what does dermatologist
# looks to?" then "so what symptoms does dermatologist treats?" both landed on
# symptom_agent (the preceding assistant turn was a question, so router.py's screening-
# continuity rule kept routing here), which has no way to answer a general "what does
# this specialty treat" question — it just produced its generic "I'm not able to
# diagnose... tell me more about your symptom" dead end, ignoring the actual question
# asked. This is a general informational question about a specialty's SCOPE, not a
# symptom description or a recommendation request, and belongs with general_info_agent
# instead (real KB/DB-grounded answer, same as any other clinic-info question).
_DEPARTMENT_SCOPE_RE = re.compile(
    r"\bwhat\s+(?:kind\s+of\s+|type\s+of\s+)?(?:symptoms?|conditions?|things|issues)?\s*"
    r"(?:does|do|can)\s+(?:a\s+|an\s+|the\s+)?[a-z]+(?:\s+[a-z]+)?\s+"
    r"(?:treats?|handles?|deals?\s+with|looks?\s+at|manages?|addresses?|covers?|sees?|specializes?\s+in)\b",
    re.IGNORECASE,
)


def is_department_scope_question(message: str) -> bool:
    """True when the patient is asking what a specialty/department TREATS in
    general ("what symptoms does a dermatologist treat", "what does cardiology
    handle") — an informational question about scope, not a description of their
    own symptoms and not a recommendation request."""
    return bool(_DEPARTMENT_SCOPE_RE.search(message.strip()))


def is_symptom_message(message: str) -> bool:
    """True when the message mentions a symptom/complaint — used by chat.py to route
    a turn to real department-list context (see app.services.department_availability)
    instead of KB retrieval, since medical_kb no longer exists. Deliberately a simple
    keyword check, not the full classify_message_intent() pipeline: a false positive
    here only means a symptom-shaped turn gets department names instead of hospital_info
    chunks, which the triage agent needs anyway; a false negative just falls through to
    ordinary hospital_info retrieval, so neither failure mode is unsafe."""
    words = re.findall(r"[a-z0-9]+", message.lower())
    return any(word in _SYMPTOM_KEYWORDS for word in words)


def _preceding_assistant_turn_looks_like_a_question(history) -> bool:
    """True when the last assistant turn was a triage/tool-driven prompt awaiting a
    reply — a DOCTOR_OPTIONS:: card (awaiting a slot pick) or a plain reply ending in
    "?" (a clarifying question). Used only to stop a bare "yes"/"no"/"ok"/"sure" reply
    from being misread as small talk when it's actually the patient answering that
    prompt mid-triage."""
    if not history:
        return False
    last = history[-1]
    if getattr(last, "role", None) != "assistant":
        return False
    content = (getattr(last, "content", "") or "").strip()
    if not content:
        return False
    if content.startswith(DOCTOR_OPTIONS_MARKER):
        return True
    return content.endswith("?") or content.endswith("؟")


# The specific "cancel" / "reschedule" action-verb vocabularies — pulled out as
# their own named constants (rather than only living inline inside
# _BOOKING_ACTION_KEYWORDS below) because app.services.orchestrator.agents.
# appointment_agent needs these exact same two word sets for its own, much
# narrower purpose (_detect_action_intent: deciding WHICH action a message
# names, not just whether it's booking-related at all). They used to be defined
# independently in both files with identical membership — real duplication that
# could silently drift apart if one were ever updated without the other. One
# shared definition here means there's only one place to update either list.
_CANCEL_ACTION_WORDS = frozenset({"cancel", "cancellation", "cancelled", "canceling", "cancelling"})
_RESCHEDULE_ACTION_WORDS = frozenset({"reschedule", "rescheduling", "postpone", "postponing"})

# Booking-action intent vocabulary for needs_booking_action_tools() below —
# deliberately broad/over-inclusive, same fail-safe philosophy as
# _KNOWLEDGE_KEYWORDS: binding book_appointment/reschedule_appointment/
# cancel_appointment unnecessarily costs a few hundred prompt tokens; omitting one
# that's actually needed breaks the booking flow outright. The asymmetry means this
# should err toward matching too much, never too little.
_BOOKING_ACTION_KEYWORDS = frozenset({
    "book", "booking", "appointment", "appointments", "schedule",
    "confirm", "confirmed", "slot", "slots",
    # Reported gap: "is there any cardiologist available on Friday?" named no booking
    # action word at all, so it fell through to general_info_agent (KB-only, no
    # get_department_availability tool) and got answered from static Retrieved
    # context prose about typical hours instead of real, current slots. "available"/
    # "availability" is exactly how patients ask this without ever saying "book" or
    # "slot" — must route to a tool-bound agent same as those do.
    "available", "availability",
}) | _CANCEL_ACTION_WORDS | _RESCHEDULE_ACTION_WORDS

# Phrasal cancel/reschedule intent that a single-word keyword check would miss —
# "I can't make it anymore" carries the same intent as "cancel" without using that
# word at all.
_BOOKING_ACTION_PHRASES = (
    "can't make it", "cant make it", "won't be able to make it", "wont be able to make it",
    "no longer need", "don't need it anymore", "dont need it anymore", "never mind",
)


def needs_booking_action_tools(message: str, history=None) -> bool:
    """True when book_appointment/reschedule_appointment/cancel_appointment should be
    bound for this agent turn (see app.services.chat_tools.build_tools). Deliberately
    biased to return True whenever genuinely unsure — see the keyword-set comment
    above for why that asymmetry matters. Only returns False when NONE of: the
    message itself names a booking action, the conversation is already mid-question
    (could be a slot pick or booking confirmation), or availability/booking has
    already been shown earlier this session (a reschedule/cancel of THAT could come
    up at any later point)."""
    lowered = message.lower()
    words = set(re.findall(r"[a-z0-9']+", lowered))
    if words & _BOOKING_ACTION_KEYWORDS:
        return True
    if any(phrase in lowered for phrase in _BOOKING_ACTION_PHRASES):
        return True
    if _preceding_assistant_turn_looks_like_a_question(history):
        return True
    if history:
        for row in history:
            if getattr(row, "role", None) != "assistant":
                continue
            content = getattr(row, "content", "") or ""
            if content.startswith(
                (DOCTOR_OPTIONS_MARKER, DEPARTMENT_LIST_MARKER, BOOKING_MARKER, DOCTOR_DISAMBIGUATION_MARKER)
            ):
                return True
    return False


def _heuristic_classify(message: str, history=None) -> str | None:
    """Fast keyword/pattern pre-filter. Returns a definitive classification for
    clear-cut cases, or None when the message is ambiguous enough to need the LLM
    fallback (e.g. an off-topic statement with no question mark or keyword).

    `history` (most-recent-last, same shape as app.models.conversation_memory rows)
    is optional and only consulted for the short-reply override below — every other
    branch is unaffected by it and behaves exactly as before when it's omitted.
    """
    stripped = message.strip()
    if not stripped:
        return CONVERSATIONAL

    # Checked before the "?" shortcut below: a personal-recall question ("what is my
    # name?") is still a question, but must not fall into the generic
    # knowledge-seeking bucket just because it has a question mark.
    if _PERSONAL_RECALL_RE.search(stripped):
        return PERSONAL_RECALL

    lowered = stripped.lower().rstrip("!.")
    words = re.findall(r"[a-z0-9]+", lowered)

    # A bare, short acknowledgment ("yes", "no", "ok", "sure") is genuinely
    # ambiguous on its own — it's small talk in isolation, but it's exactly what a
    # patient sends answering a clarifying triage question or picking a doctor-option
    # card. Checked before the conversational-phrase/short-message shortcuts below
    # (which would otherwise claim "ok"/"sure" outright) so a reply mid-triage always
    # reaches the agent instead of getting a generic chat response with no
    # diagnosis_guard check applied to it.
    # Capped at 6 words, not 3 — confirming a doctor's name routinely runs longer
    # than a bare "yes"/"ok" once the name itself is included (e.g. "yes Dr. Babar
    # Ali" is already 4 words: yes/dr/babar/ali). A 3-word cap silently missed
    # exactly that case, sent it to the LLM fallback below, which misclassified it
    # as small talk — routing a patient who'd just confirmed a doctor's name to the
    # tool-less conversational reply path instead of the agent that could actually
    # look up availability for them.
    if 1 <= len(words) <= 6 and _preceding_assistant_turn_looks_like_a_question(history):
        return KNOWLEDGE_SEEKING

    # "؟" is the Arabic/Urdu-script question mark — a real question in Urdu script
    # usually ends with this instead of ASCII "?".
    if "?" in stripped or "؟" in stripped:
        return KNOWLEDGE_SEEKING

    if lowered in _CONVERSATIONAL_PHRASES:
        return CONVERSATIONAL

    if _QUESTION_WORD_RE.match(stripped):
        return KNOWLEDGE_SEEKING

    if any(word in _KNOWLEDGE_KEYWORDS for word in words):
        return KNOWLEDGE_SEEKING

    # A short *Latin-script* message with none of the above signals is almost always
    # small talk ("ok thanks", "sounds good", "haha nice"). Requiring `words` to
    # actually cover the message (not just be short) matters for non-Latin script
    # (e.g. Urdu script): word-regex extraction yields an empty list there regardless
    # of content, which is a tokenization gap, not evidence of small talk — that case
    # must fall through to the LLM classifier below, not get defaulted here.
    non_word_chars = re.sub(r"[a-z0-9\s,'!.]", "", lowered)
    if words and len(words) <= 3 and not non_word_chars:
        return CONVERSATIONAL

    return None


_CLASSIFY_SYSTEM_PROMPT = """Classify the patient's message as exactly one of:

KNOWLEDGE_SEEKING: a question or statement that could be answered by clinic \
information (hours, location, fees, booking, departments, doctors), that mentions a \
medical symptom or health concern (even loosely, indirectly, or in passing), or any \
other factual question not about the conversation itself (e.g. general trivia).

PERSONAL_RECALL: the patient is asking the assistant to recall something the patient \
themselves said earlier in this same conversation (their name, a preference, an \
earlier answer) — NOT asking a new question about the clinic or the world.

CONVERSATIONAL: a greeting, thanks, small talk, an off-topic personal statement, or \
an acknowledgment with no informational request about the clinic, health, or the \
conversation history.

When in doubt, or if the message has ANY clinical or factual angle at all, answer \
KNOWLEDGE_SEEKING — it is always safer to over-classify as KNOWLEDGE_SEEKING than to \
miss a real question. Only answer PERSONAL_RECALL when the message is clearly and \
specifically asking what the PATIENT said before, not asking the assistant a new \
factual or clinical question.

Respond with exactly one word: KNOWLEDGE_SEEKING, PERSONAL_RECALL, or CONVERSATIONAL."""


def _llm_classify(message: str) -> str:
    if not settings.LLM_API_KEY:
        return KNOWLEDGE_SEEKING

    try:
        llm = ChatGroq(model=settings.GROQ_HELPER_MODEL, api_key=api_key_manager.next_key(), temperature=0.0)
        raw = llm.invoke(
            [SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT), HumanMessage(content=message)]
        ).content.strip().upper()
    except Exception:
        logger.exception("Message intent classification failed, defaulting to knowledge_seeking")
        return KNOWLEDGE_SEEKING

    if "PERSONAL_RECALL" in raw and "KNOWLEDGE_SEEKING" not in raw:
        return PERSONAL_RECALL
    if "CONVERSATIONAL" in raw and "KNOWLEDGE_SEEKING" not in raw:
        return CONVERSATIONAL
    return KNOWLEDGE_SEEKING


def classify_message_intent(message: str, history=None) -> str:
    heuristic_result = _heuristic_classify(message, history)
    if heuristic_result is not None:
        return heuristic_result
    return _llm_classify(message)
