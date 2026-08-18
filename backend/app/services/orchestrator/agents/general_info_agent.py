"""General-info specialist agent (app.services.orchestrator architecture).

Equivalent to the original single-pipeline get_chat_reply(): retrieves from the
hospital_info KB (hours, policies, doctor/department facts — the correct collection
for clinic facts, never symptoms), no tools bound, reuses _SYSTEM_PROMPT as-is,
unchanged from the original single-pipeline system — including its own
CONVERSATIONAL EXCEPTION, which already handles small talk fine without any
special-casing needed here (the router sends both small talk and personal-recall
questions to this agent; see app.services.orchestrator.router).

DEPARTMENT LIST SHORT-CIRCUIT: "what are the available depts" / "show me available
departments" is answered deterministically from the clinic's real, active department
names, bypassing the LLM+KB retrieval entirely — this agent has no reliable KB
document guaranteed to list every current department (departments are admin-managed
data that can change any time, not static KB content), so answering from the same
real, live query app.services.department_availability.list_active_department_names
already provides elsewhere (symptom_agent's own context line) is both simpler and
guaranteed accurate. Reported live: this phrasing wasn't recognized at all — see
app.services.orchestrator.router's own Rule 1.5 for why (it collides with a booking
keyword and used to be misrouted to appointment_agent, which has no way to answer
it).
"""
from sqlalchemy.orm import Session

from app.core.tenancy import ClinicContext
from app.models.conversation_memory import ConversationMemory
from app.rag.retrieval import retrieve
from app.services import llm
from app.services.department_availability import list_active_department_names
from app.services.message_classifier import (
    CONVERSATIONAL,
    PERSONAL_RECALL,
    classify_message_intent,
    is_department_list_explanation_request,
    is_department_list_followup_request,
    is_department_list_request,
)
from app.services.orchestrator.symptom_hints import department_symptom_labels
from app.services.query_rewrite import rewrite_query

# Marker substring unique to this agent's own department-list replies below — lets
# _is_department_list_reply recognize "that was a department list" without a false
# match against some other assistant turn that happens to mention "departments" in
# passing (e.g. quoting a KB chunk).
_DEPARTMENT_LIST_REPLY_SIGNATURE = "departments available at this clinic"


def _is_department_list_reply(history: list[ConversationMemory]) -> bool:
    """True when the immediately preceding turn was this agent's own department-list
    short-circuit reply — see is_department_list_followup_request's docstring for why
    a bare "name them" is only ever trusted as a continuation of that specific reply."""
    if not history:
        return False
    last = history[-1]
    return last.role == "assistant" and _DEPARTMENT_LIST_REPLY_SIGNATURE in last.content


def run_general_info_agent(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
) -> str:
    if is_department_list_request(message) or (
        is_department_list_followup_request(message) and _is_department_list_reply(history)
    ):
        department_names = list_active_department_names(db, ctx.clinic_id)
        if not department_names:
            return (
                "There are no departments configured at this clinic right now."
                if language != "ur"
                else "اس وقت اس کلینک میں کوئی شعبہ دستیاب نہیں ہے۔"
            )
        names_text = ", ".join(department_names)
        # Reported live: "show me list of depts... and also explanation of each dept
        # that which symptoms does they treat in detail" got ONLY the bare list back
        # — the second half of the same message was silently dropped, since this
        # short-circuit used to always return immediately regardless of what else
        # the message asked. Answered from the same real, vetted routing table this
        # module uses elsewhere (see department_symptom_labels' own docstring),
        # never an LLM freehanding descriptions of departments it wasn't shown.
        if not is_department_list_explanation_request(message):
            return (
                f"Here are the departments available at this clinic: {names_text}."
                if language != "ur"
                else f"اس کلینک میں دستیاب شعبے یہ ہیں: {names_text}۔"
            )
        labels_by_department = department_symptom_labels(department_names)
        lines = [
            f"- {name}: {', '.join(labels_by_department[name])}"
            if labels_by_department[name]
            else f"- {name}: general consultations in this specialty"
            for name in department_names
        ]
        body = "\n".join(lines)
        intro = (
            "Here are the departments available at this clinic, along with the kinds "
            "of symptoms each typically treats:"
            if language != "ur"
            else "اس کلینک میں دستیاب شعبے، اور ہر شعبہ عام طور پر جن علامات کا علاج کرتا ہے:"
        )
        return f"{intro}\n\n{body}"

    # Small talk and personal-recall questions never need KB grounding — the
    # system prompt's own CONVERSATIONAL EXCEPTION already knows how to reply
    # naturally when context is "(none)", but retrieval's similarity gate is a
    # coincidence of embedding geometry, not a real "is this even a question"
    # check: retrieve() only inspects the SINGLE best chunk's raw cosine score
    # before deciding to return "(none)" — a short, low-content message like
    # "hello bye" can occasionally score above RETRIEVAL_SIMILARITY_THRESHOLD
    # against some unrelated KB chunk purely by chance, and once that one score
    # clears the bar, the ensemble step returns its top 5 chunks unconditionally
    # regardless of whether any of them are actually relevant. Reported live:
    # exactly this happened for "hello bye" — real (irrelevant) doctor/
    # department chunks leaked into the prompt instead of "(none)".
    # classify_message_intent's own heuristic (word-list/shape based, not
    # embedding-based) is a deterministic check for "is this even a knowledge
    # question" that doesn't depend on chance embedding geometry, so it's
    # checked first and skips retrieval entirely for these two intents — the
    # same behavior the pre-orchestrator single-pipeline system already
    # guaranteed via this exact classifier before the orchestrator rewrite.
    intent = classify_message_intent(message, history)
    if intent in (CONVERSATIONAL, PERSONAL_RECALL):
        context_chunks = []
    else:
        # Same raw-first, rewrite-as-rescue retrieval pattern as the original
        # single-pipeline chat.py — a clean standalone question must never be
        # touched by rewriting, which only kicks in when the raw message
        # genuinely fails.
        result = retrieve(db, ctx.clinic_id, message)
        if not result.matched:
            retrieval_query = rewrite_query(message, history)
            if retrieval_query != message:
                result = retrieve(db, ctx.clinic_id, retrieval_query)
        context_chunks = result.chunks if result.matched else []

    system_prompt = llm._SYSTEM_PROMPT.format(
        language_name=llm._LANGUAGE_NAMES.get(language, "English"),
        context="\n\n".join(context_chunks) if context_chunks else "(none)",
    )
    return llm.run_plain_reply(system_prompt, message, history)
