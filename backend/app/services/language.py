"""Minimal English/Urdu detection for chat replies.

Urdu is written in Perso-Arabic script (Unicode block U+0600-U+06FF), which never
overlaps with Latin text — so a single codepoint in that range is a reliable signal,
with no need for a language-detection model or library. Romanized ("Roman Urdu") input
is out of scope: the SOW asks for English vs Urdu, not transliteration detection.
"""
import re

_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿ]")

Language = str  # "en" | "ur"


def detect_language(text: str) -> Language:
    return "ur" if _ARABIC_SCRIPT_RE.search(text) else "en"
