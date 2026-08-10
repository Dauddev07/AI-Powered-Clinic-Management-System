"""Shared text preprocessing for RAG.

CRITICAL: this exact function must run on both (a) knowledge-base chunks before they're
embedded and stored, and (b) incoming chatbot queries before they're embedded for
retrieval. If the two ever diverge, stored and queried text stop matching and retrieval
quality degrades silently. Do not fork or duplicate this logic elsewhere.

Stopwords are retained by design (per SOW) — this is NOT a bag-of-words/BM25-style
aggressive normalization pipeline, just enough cleanup for consistent embeddings.
"""
import re

import nltk
from nltk.stem import WordNetLemmatizer

# Render's filesystem is ephemeral, so the wordnet corpus isn't present on a fresh
# deploy the way it might be on a dev machine that's downloaded it before — fetch it
# on first import if missing, rather than requiring a separate provisioning step.
# Downloaded into /tmp explicitly: nltk's default search paths (site-packages-adjacent,
# /usr/share/...) aren't guaranteed writable in a deployed container, and nltk.download()
# swallows write/network failures into a return value instead of raising, so a silent
# failure there would otherwise surface later as a confusing LookupError deep inside
# WordNetLemmatizer instead of a clear error here.
_NLTK_DATA_DIR = "/tmp/nltk_data"
nltk.data.path.insert(0, _NLTK_DATA_DIR)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    if not nltk.download("wordnet", download_dir=_NLTK_DATA_DIR, quiet=True):
        raise RuntimeError(
            "Failed to download the NLTK wordnet corpus into "
            f"{_NLTK_DATA_DIR} — check outbound network access and disk space."
        )

_lemmatizer = WordNetLemmatizer()
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _lemmatize_word(word: str) -> str:
    # No POS tagger dependency: try verb lemmatization first, then noun (WordNet's
    # default pos), which catches the common cases (running->run, mice->mouse)
    # without the extra weight/downloads a full POS tagger would need.
    lemma = _lemmatizer.lemmatize(word, pos="v")
    if lemma == word:
        lemma = _lemmatizer.lemmatize(word, pos="n")
    return lemma


def _normalize_and_lemmatize(text: str) -> str:
    # Whitespace normalisation.
    text = re.sub(r"\s+", " ", text).strip()

    # Lemmatisation (stopwords retained — every token is kept, just normalized).
    words = _WORD_RE.findall(text.lower())
    lemmatized = [_lemmatize_word(w) for w in words]

    return " ".join(lemmatized)


def clean_text(text: str) -> str:
    """Ingestion-side preprocessing: full pipeline including regex cleanup for
    PDF/DOCX extraction artifacts (stray page numbers, header/footer noise, URLs)."""
    if not text:
        return ""

    # Regex cleanup: strip control/non-printable characters and URLs.
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    return _normalize_and_lemmatize(text)


def preprocess_query(text: str) -> str:
    """Query-side preprocessing: whitespace normalisation + lemmatisation only.

    Deliberately skips the regex-cleanup step from clean_text() — that step strips
    PDF/DOCX extraction artifacts a live chat message never has. Applying it would be
    harmless-but-pointless at best; the two sides of the similarity comparison only
    need to agree on normalisation and lemmatisation, not on extraction cleanup.
    """
    if not text:
        return ""
    return _normalize_and_lemmatize(text)
