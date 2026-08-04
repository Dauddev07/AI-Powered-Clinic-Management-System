"""Rewrites a conversationally-phrased follow-up into a standalone retrieval query.

A question like "not on the info but overall in total how many departments does this
clinic have" scores below RETRIEVAL_SIMILARITY_THRESHOLD even when the underlying
question ("how many departments") was answered correctly a turn earlier — the extra
conversational wording dilutes the embedding. This module makes one small, focused LLM
call to strip that wording down to a clean standalone question before retrieval, using
only the last couple of turns of history for context (not the full conversation).

Never blocks the chat turn: any failure (no LLM configured, API error, degenerate
output) falls back to the raw message unchanged.
"""
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.api_keys import api_key_manager
from app.core.config import settings
from app.models.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

# Only the last couple of turns are relevant to disambiguating a follow-up question —
# more than that just adds noise (and cost) to a call that must stay fast.
REWRITE_HISTORY_TURNS = 2

_SYSTEM_PROMPT = """Rewrite the patient's latest message as a single, standalone \
question suitable for searching a knowledge base, using the recent conversation only \
to resolve references (e.g. "it", "that", "overall") the latest message makes.

Rules:
- Output ONLY the rewritten question, nothing else — no preamble, no quotes.
- Keep it in the same language as the patient's latest message.
- If the latest message is already standalone, return it unchanged.
- Never answer the question yourself, only rewrite it.
"""

# A rewrite outside this length band relative to the original is treated as
# degenerate (the LLM answered instead of rewriting, or returned near-nothing) and
# discarded in favor of the raw message.
_MIN_REWRITE_LENGTH = 3
_MAX_REWRITE_LENGTH_MULTIPLIER = 6


def _recent_turns_text(history: list[ConversationMemory]) -> str:
    recent = history[-(REWRITE_HISTORY_TURNS * 2):]
    lines = [f"{row.role}: {row.content}" for row in recent]
    return "\n".join(lines) if lines else "(no prior turns)"


def _is_degenerate(rewritten: str, raw_message: str) -> bool:
    if len(rewritten) < _MIN_REWRITE_LENGTH:
        return True
    if len(rewritten) > len(raw_message) * _MAX_REWRITE_LENGTH_MULTIPLIER + 100:
        return True
    return False


def rewrite_query(message: str, history: list[ConversationMemory]) -> str:
    if not settings.LLM_API_KEY:
        return message

    try:
        llm = ChatGroq(model=settings.GROQ_HELPER_MODEL, api_key=api_key_manager.next_key(), temperature=0.0)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Recent conversation:\n{_recent_turns_text(history)}\n\nLatest message: {message}"
            ),
        ]
        rewritten = llm.invoke(messages).content.strip()
    except Exception:
        logger.exception("Query rewrite failed, falling back to raw message")
        return message

    if _is_degenerate(rewritten, message):
        return message
    return rewritten
