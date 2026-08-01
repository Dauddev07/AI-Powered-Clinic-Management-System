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
from app.services.chat_markers import DOCTOR_OPTIONS_MARKER

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
_PERSONAL_RECALL_RE = re.compile(
    r"\b("
    r"what(?:'s|s| is) my name"
    r"|what did i (?:tell|say|mention)"
    r"|what did i just (?:tell|say|mention)"
    r"|do you remember (?:what|that) i"
    r"|what do you (?:know|remember) about me"
    r"|what did you (?:say|tell) (?:i|me)"
    r"|remind me what i (?:said|told)"
    r"|what have i (?:told|said)"
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

# Common symptom/complaint vocabulary (including Roman Urdu, since a symptom
# mention shouldn't slip through undetected just because it's not in English) —
# used both by the broad knowledge-seeking heuristic below and by
# is_symptom_message() to decide chat.py's retrieval routing (real department-list
# context instead of medical_kb retrieval, which no longer exists).
_SYMPTOM_KEYWORDS = frozenset({
    "pain", "ache", "aches", "aching", "fever", "cough", "symptom", "symptoms",
    "headache", "cold", "flu", "medicine", "medication", "prescription", "treatment",
    "diagnosis", "sick", "ill", "illness", "hurt", "hurts", "injury", "injured",
    "dizzy", "dizziness", "nausea", "vomit", "vomiting", "rash", "allergy", "allergic",
    "bukhar", "dard", "khansi", "tabiyat", "kamzori", "zukam",
})

# Deliberately broad and erring toward false positives (routing more things to
# knowledge_seeking than strictly necessary) per this module's fail-safe design.
_KNOWLEDGE_KEYWORDS = _LOGISTICS_KEYWORDS | _SYMPTOM_KEYWORDS


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
        llm = ChatGroq(model=settings.GROQ_MODEL, api_key=api_key_manager.next_key(), temperature=0.0)
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
