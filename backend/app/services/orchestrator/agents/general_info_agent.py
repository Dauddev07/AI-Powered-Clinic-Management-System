"""General-info specialist agent (app.services.orchestrator architecture).

Equivalent to the original single-pipeline get_chat_reply(): retrieves from the
hospital_info KB (hours, policies, doctor/department facts — the correct collection
for clinic facts, never symptoms), no tools bound, reuses _SYSTEM_PROMPT as-is,
unchanged from the original single-pipeline system — including its own
CONVERSATIONAL EXCEPTION, which already handles small talk fine without any
special-casing needed here (the router sends both small talk and personal-recall
questions to this agent; see app.services.orchestrator.router).

PERSONAL RECALL SHORT-CIRCUIT: a direct "what's my name" / "what are the things I
told you" question is answered deterministically from `patient_memory` in code (see
run_general_info_agent below), bypassing the LLM entirely for this one case — live
testing showed the model unreliably ignoring PATIENT MEMORY for exactly this
question even when it was correctly present in the prompt with explicit
instructions to use it (reported bug: "why 2nd chat dont know details"). This
mirrors the project's established pattern of composing anything correctness-critical
in code rather than trusting the model to relay it right every time.

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
from app.services.message_classifier import is_department_list_request, is_personal_recall_message
from app.services.query_rewrite import rewrite_query


def run_general_info_agent(
    db: Session,
    ctx: ClinicContext,
    message: str,
    language: str,
    history: list[ConversationMemory],
    patient_memory: str = "",
) -> str:
    if is_department_list_request(message):
        department_names = list_active_department_names(db, ctx.clinic_id)
        if department_names:
            names_text = ", ".join(department_names)
            return (
                f"Here are the departments available at this clinic: {names_text}."
                if language != "ur"
                else f"اس کلینک میں دستیاب شعبے یہ ہیں: {names_text}۔"
            )
        return (
            "There are no departments configured at this clinic right now."
            if language != "ur"
            else "اس وقت اس کلینک میں کوئی شعبہ دستیاب نہیں ہے۔"
        )

    stripped_memory = patient_memory.strip() if patient_memory else ""
    if stripped_memory and is_personal_recall_message(message):
        return (
            "Here's what I have from our earlier conversations: " + stripped_memory
            if language != "ur"
            else "ہماری پچھلی گفتگو سے میرے پاس یہ معلومات ہیں: " + stripped_memory
        )

    # Same raw-first, rewrite-as-rescue retrieval pattern as the original
    # single-pipeline chat.py — a clean standalone question must never be touched
    # by rewriting, which only kicks in when the raw message genuinely fails.
    result = retrieve(db, ctx.clinic_id, message)
    if not result.matched:
        retrieval_query = rewrite_query(message, history)
        if retrieval_query != message:
            result = retrieve(db, ctx.clinic_id, retrieval_query)
    context_chunks = result.chunks if result.matched else []

    system_prompt = llm._SYSTEM_PROMPT.format(
        language_name=llm._LANGUAGE_NAMES.get(language, "English"),
        context="\n\n".join(context_chunks) if context_chunks else "(none)",
        patient_memory=patient_memory.strip() if patient_memory and patient_memory.strip() else "(none)",
    )
    return llm.run_plain_reply(system_prompt, message, history)
