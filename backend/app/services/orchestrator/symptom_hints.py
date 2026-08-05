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
    (
        frozenset({
            "skin", "itchy", "itching", "rash", "allergic", "allergy", "eczema", "hives",
            # Common named skin complaints — same "specific symptom word missing from
            # an otherwise-covered category" gap as the head/leg additions below.
            "mole", "moles", "acne", "pimple", "pimples", "boil", "boils", "wart", "warts",
        }),
        ("derma",),
        "skin symptoms",
    ),
    (
        frozenset({"ear", "earache", "hearing"}),
        ("ent", "otolaryn", "ear"),
        "ear pain",
    ),
    (
        # Throat/nose complaints are ENT territory, not Pulmonology — kept as their
        # own entry (distinct label) rather than folded into "ear pain" above or the
        # cough/respiratory entry below, so the synthesized note names the right
        # symptom either way.
        frozenset({"throat", "nose", "sinus", "sinusitis", "nasal", "hoarse", "hoarseness"}),
        ("ent", "otolaryn"),
        "throat/nose symptoms",
    ),
    (frozenset({"eye", "eyes", "vision"}), ("ophthal", "eye"), "eye symptoms"),
    # Reported live: "pain in my teeths" (a common typo — extra "s" on the already-
    # plural "teeth") matched no keyword at all, so Dentistry was silently missing
    # from `hinted` even though the LLM itself had already correctly shown a
    # Dentistry card for it — the appointment_agent mismatch check later saw the
    # patient's own symptoms as NOT supporting Dentistry and wrongly warned them
    # off it when they asked about dentist availability.
    (frozenset({"tooth", "teeth", "teeths", "toothache", "dental"}), ("dent",), "tooth pain"),
    (
        frozenset({"bone", "fracture", "fractured", "sprain", "sprained", "joint", "dislocated", "dislocation"}),
        ("ortho",),
        "bone/joint injury",
    ),
    # Reported live: "pain in my teeth, head and legs as well" only ever produced a
    # Dentistry card — neither "head"/"headache" nor "leg"/"legs" matched ANY
    # keyword in this table at all, so the existing cross-check in symptom_agent.py
    # (which already adds an extra department card for every real symptom the
    # patient's own words point to, once the model has returned its first card) had
    # nothing to find for either one. Not a missing mechanism, a missing keyword.
    (frozenset({"leg", "legs", "knee", "ankle", "thigh", "calf"}), ("ortho",), "leg pain"),
    # Same gap, other common joints/limbs not covered by "bone/joint injury" above
    # (that entry needs an injury WORD like "sprain"/"fracture"; plain "my shoulder
    # hurts" names no injury at all, same shape as plain "leg" above).
    (
        frozenset({
            "back", "backache", "spine", "shoulder", "shoulders", "arm", "arms",
            "hip", "hips", "wrist", "wrists", "elbow", "elbows",
        }),
        ("ortho",),
        "joint/muscle pain",
    ),
    (
        frozenset({"head", "headache", "headaches", "migraine", "migraines"}),
        ("general medicine", "internal medicine", "family medicine"),
        "head pain",
    ),
    (
        frozenset({"chest", "heart", "palpitations", "cardiac", "hypertension"}),
        ("cardio",),
        "chest pain",
    ),
    (
        frozenset({"stomach", "abdominal", "abdomen", "digestive", "diarrhea", "diarrhoea", "vomit", "vomiting"}),
        ("gastro", "internal medicine", "general medicine"),
        "stomach symptoms",
    ),
    (
        # Kidney/urinary complaints kept separate from the blood-sugar entry below —
        # sharing "urination"/"urinate" there is specifically about frequency/thirst
        # (the diabetes triad), not pain or blood in urine, so this needs its own
        # label to synthesize an accurate note.
        frozenset({"kidney", "kidneys", "urinary", "urine"}),
        ("internal medicine", "general medicine"),
        "urinary symptoms",
    ),
    (
        frozenset({
            "anxiety", "depression", "depressed", "mental", "stress", "stressed",
            "sad", "sadness", "hopeless", "hopelessness", "crying", "panic", "insomnia",
        }),
        ("psych",),
        "low mood",
    ),
    (
        # "cough" deliberately NOT included on its own — reported live: a plain
        # "mild cough with fever" got an extra, unwanted Pulmonology card next to
        # the General Medicine one. Clinically a mild cough+fever is ordinary
        # General Medicine territory; it only points to Pulmonology once there's
        # an actual breathing-distress symptom alongside it (difficulty
        # breathing, wheezing) — those are what this entry hints on. Deliberately
        # NOT including "swelling"/"swollen" here even though a cough+swelling
        # combination can mean Pulmonology: those words alone are far too generic
        # (ankle swelling, jaw swelling) and matching is per-word/per-category,
        # not co-occurrence-aware, so adding them would falsely hint Pulmonology
        # on unrelated injuries anywhere else in the body.
        frozenset({
            "breathless", "breathing", "wheeze", "wheezing", "lung", "respiratory",
            "asthma",
        }),
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
        # Same neuro category, a second entry rather than folded into the one above:
        # numbness/weakness/tremor point at Neurology specifically, not Oncology —
        # the "oncol" hint above is deliberately not included here.
        frozenset({"numbness", "numb", "weakness", "tremor", "tremors", "paralysis", "paralyzed"}),
        ("neuro",),
        "neurological symptoms",
    ),
    (
        # Reported live: "mild dizziness and pain in my jaw" — the LLM asked about
        # the dizziness once, got "mild", then the final recommendation only ever
        # named Dentistry (for the jaw) and silently dropped dizziness entirely,
        # since no keyword in this table covered it at all — same "whole symptom
        # missing" gap as the pre-fix Gynecology/urinary/back entries above.
        # Vertigo/dizziness is worked up by ENT (inner-ear/balance) and Neurology,
        # not Dentistry, so both are hinted here.
        frozenset({"dizziness", "dizzy", "vertigo", "lightheaded", "lightheadedness"}),
        ("ent", "otolaryn", "neuro"),
        "dizziness",
    ),
    (
        # Pregnancy/menstrual/pelvic symptoms — Gynecology had NO entry in this table
        # at all before, meaning it could never be hinted regardless of what the
        # patient described (same "whole category missing" gap, one level up from a
        # missing keyword). Pediatrics is intentionally still not covered here: it's
        # routed by patient AGE, not by symptom vocabulary, so no symptom keyword
        # would ever correctly imply it.
        frozenset({
            "pregnant", "pregnancy", "period", "periods", "menstrual", "menstruation",
            "vaginal", "pelvic",
        }),
        ("gynec",),
        "gynecological symptoms",
    ),
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
