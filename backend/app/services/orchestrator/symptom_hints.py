"""Shared symptom-vocabulary -> department-name-substring hint table.

Originally lived only inside symptom_agent.py, used to backstop the model's own
department routing there. Promoted to its own module so appointment_agent can reuse
the exact same table and matching logic — reported live: a patient described fever +
body aches (routed correctly to General Medicine), then asked to book with a
department their own symptoms didn't support ("isn't neurologist a better idea?" /
"i think ent would handle this better") — appointment_agent has no symptom awareness
at all today, so it just showed availability for whatever department the patient
named next, and a follow-up "what do you think based on my symptoms" got a free-text
reply that HALLUCINATED a symptom ("ear pain") the patient never actually mentioned,
since the model had no real, grounded symptom data to reason from. Both gaps need the
same real, deterministic symptom-to-department mapping this module provides — never
letting either agent's LLM freehand which department a symptom belongs to.
"""
import re

from app.models.conversation_memory import ConversationMemory

SYMPTOM_DEPARTMENT_HINTS: tuple[tuple[frozenset[str], tuple[str, ...], str], ...] = (
    (frozenset({"skin", "itchy", "itching", "rash", "allergic", "allergy", "eczema", "hives"}), ("derma",), "skin symptoms"),
    (frozenset({"ear", "earache", "hearing"}), ("ent", "otolaryn", "ear"), "ear pain"),
    (frozenset({"eye", "eyes", "vision"}), ("ophthal", "eye"), "eye symptoms"),
    (frozenset({"tooth", "teeth", "toothache", "dental"}), ("dent",), "tooth pain"),
    (
        frozenset({"bone", "fracture", "fractured", "sprain", "sprained", "joint", "dislocated", "dislocation"}),
        ("ortho",),
        "bone/joint injury",
    ),
    (frozenset({"chest", "heart", "palpitations", "cardiac"}), ("cardio",), "chest pain"),
    (
        frozenset({"stomach", "abdominal", "abdomen", "digestive", "diarrhea", "diarrhoea", "vomit", "vomiting"}),
        ("gastro", "internal medicine", "general medicine"),
        "stomach symptoms",
    ),
    (
        frozenset({
            "anxiety", "depression", "depressed", "mental", "stress", "stressed",
            "sad", "sadness", "hopeless", "hopelessness", "crying",
        }),
        ("psych",),
        "low mood",
    ),
    (
        frozenset({"cough", "breathless", "wheeze", "wheezing", "lung", "respiratory"}),
        ("pulmon", "respiratory"),
        "respiratory symptoms",
    ),
    (
        frozenset({
            "sugar", "diabetes", "diabetic", "thirst", "thirsty", "urination", "urinate",
            "urinating",
        }),
        ("endocrin", "internal medicine", "general medicine"),
        "blood sugar symptoms",
    ),
    (frozenset({"brain", "tumor", "tumour", "cancer", "seizure", "seizures"}), ("neuro", "oncol"), "neurological symptoms"),
    (
        frozenset({"fever", "chills", "sweating", "ache", "aches", "aching", "body ache", "malaise", "fatigue"}),
        ("general medicine", "internal medicine", "family medicine"),
        "fever/body aches",
    ),
)


def departments_hinted_by_patient_symptom_words(
    message: str, history: list[ConversationMemory], department_names: list[str], already_covered: set[str]
) -> dict[str, str]:
    """Returns {department_name: symptom_label} for every real, active,
    not-yet-covered department a symptom category the PATIENT mentioned (this
    message and/or their own earlier turns, never the assistant's) points to.
    Pass message="" to scan only prior history (e.g. when checking what the
    patient has described SO FAR in the session, independent of a current
    message that isn't itself symptom-shaped, like a booking request or a
    "what do you recommend" question)."""
    patient_texts = [message] + [
        getattr(row, "content", "") or "" for row in history if getattr(row, "role", None) == "user"
    ]
    words = set(re.findall(r"[a-z0-9]+", " ".join(patient_texts).lower()))
    hinted_substrings: dict[str, str] = {}
    for keywords, hints, label in SYMPTOM_DEPARTMENT_HINTS:
        if words & keywords:
            for hint in hints:
                hinted_substrings.setdefault(hint, label)
    if not hinted_substrings:
        return {}
    # Anchored to the START of a word only (not "anywhere inside one") — e.g. the
    # "ear" keyword's hint substring "ent" must match "ENT" itself, never mid-word
    # inside an unrelated name like "Dentistry" (d-ENT-istry).
    missing: dict[str, str] = {}
    for name in department_names:
        if name in already_covered:
            continue
        lowered_name = name.lower()
        for hint, label in hinted_substrings.items():
            if re.search(rf"\b{re.escape(hint)}", lowered_name):
                missing[name] = label
                break
    return missing
