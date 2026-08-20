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
    (
        frozenset({"eye", "eyes", "vision", "blur", "blurry", "blurred", "blury"}),
        ("ophthal", "eye"),
        "eye symptoms",
    ),
    # Reported live: "pain in my teeths" (a common typo — extra "s" on the already-
    # plural "teeth") matched no keyword at all, so Dentistry was silently missing
    # from `hinted` even though the LLM itself had already correctly shown a
    # Dentistry card for it — the appointment_agent mismatch check later saw the
    # patient's own symptoms as NOT supporting Dentistry and wrongly warned them
    # off it when they asked about dentist availability.
    # Reported live (2nd report): "jaw" was never a keyword here at all. The LLM's
    # own first-turn reply correctly said "a dentist can evaluate" mild jaw pain,
    # but when the patient then suggested "i think cardiologist can be best fit for
    # it?", that recommendation-request phrasing routes to a deterministic shortcut
    # that scans PRIOR history for real symptom words and answers from THAT, never
    # from what the patient guessed — with "jaw" untracked, the scan came back
    # empty, fell through to the LLM with nothing to push back with, and the model
    # just went along with the patient's own suggestion (Cardiology) instead of
    # reinforcing what it had already correctly concluded. Same "missing keyword,
    # not missing mechanism" shape as the "teeths" typo above.
    # Reported live (3rd report): "issue in chewing any solid thing" — "chewing"/
    # "chew" were never keywords here either, same shape as "jaw" above.
    (frozenset({"tooth", "teeth", "teeths", "toothache", "dental", "jaw", "chewing", "chew"}), ("dent",), "tooth pain"),
    (
        # "joint"/"joints" kept HERE (unconditional Orthopedics), unlike bare
        # "leg"/"arm" below — reported live, moving it into the conditional
        # limb/joint bucket was wrong: "joint pain" is already a specific, named
        # complaint (unlike generic "leg hurts", which could mean almost
        # anything), so it doesn't need an injury signal to warrant an
        # orthopedic look. Also covers the previously-missing plural "joints".
        frozenset({
            "bone", "fracture", "fractured", "sprain", "sprained", "dislocated",
            "dislocation", "joint", "joints",
        }),
        ("ortho",),
        "bone/joint injury",
    ),
    # NOTE: "limb/joint pain" (leg/arm/shoulder/hand/etc.) is deliberately NOT a
    # table entry here — it needs an if/else, not an OR-only keyword match. See
    # _LIMB_JOINT_WORDS + _LIMB_JOINT_INJURY_SIGNAL_WORDS and the branch in
    # departments_hinted_by_patient_symptom_words below for why and how.
    # NOTE: "neck"/"necks" is ALSO deliberately NOT here (unlike "joint" above),
    # for the same reason — see _NECK_NEUROLOGICAL_SIGNAL_WORDS and the branch
    # below for why and how.
    # NOTE: bare "head" is deliberately NOT in this entry — see
    # _GENERIC_PAIN_WORDS and the co-occurrence branch below for why and how.
    (
        frozenset({"headache", "headaches", "migraine", "migraines"}),
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
        #
        # Reported live: this used to ALSO hint General/internal medicine as a
        # fallback whenever no real Urology department existed at the clinic —
        # instructed to stop: a clinic with no urology specialist shouldn't have a
        # patient's urinary complaint silently rerouted to a department that isn't
        # actually equipped for it. Only a genuine Urology match ("urolog") is
        # hinted here now; when no real Urology department exists, this correctly
        # hints nothing at all, and _ORPHAN_SYMPTOM_CATEGORIES below is what turns
        # that into an honest apology instead of a wrong-department card.
        frozenset({"kidney", "kidneys", "urinary", "urine"}),
        ("urolog",),
        "urinary symptoms",
    ),
    (
        # Reported live: "pain in my chest and in testies as well" only ever got a
        # single General Medicine card with no reasoning attached — testicular/groin
        # pain had NO entry anywhere in this table. A General/internal/family
        # medicine fallback was added for it at the time, on the reasoning that this
        # clinic has no dedicated Urology department — instructed to stop: same as
        # the urinary entry above, a symptom with no real matching specialty here
        # should get an honest apology, never rerouted to a department that isn't
        # actually equipped for it. Only a genuine Urology match is hinted now.
        frozenset({
            "testicle", "testicles", "testicular", "testies", "groin", "scrotum", "scrotal",
        }),
        ("urolog",),
        "groin/testicular symptoms",
    ),
    (
        # Reported live: "i am feeling very sad today and dizzy as well" fired a
        # full Psychiatry card off the single word "sad" — no follow-up, no
        # screening, unlike physical symptoms which always go through PATH-2
        # duration/severity screening before a card appears. Split into two tiers:
        # words that already NAME a clinical condition (anxiety, depression, panic,
        # insomnia, "mental") represent a real presenting complaint on their own and
        # still fire unconditionally. Passing mood adjectives that could just be a
        # throwaway remark ("sad", "stressed", "crying", "hopeless") need an actual
        # persistence/duration signal alongside them — see
        # _LOW_MOOD_PASSING_WORDS/_LOW_MOOD_PERSISTENCE_WORDS below, same
        # "bare word needs corroboration" shape as the limb/joint and cough entries.
        frozenset({"anxiety", "depression", "depressed", "mental", "panic", "insomnia"}),
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
            # "breething" — same typo shape as "diziness" elsewhere in this
            # module: a common misspelling of "breathing" that matched no
            # keyword here at all.
            "breathless", "breathing", "breething", "wheeze", "wheezing", "lung", "respiratory",
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
        # Vertigo/dizziness is worked up by ENT (inner-ear/balance) first and
        # foremost — that's the by-far-most-common cause, so ENT alone is hinted
        # here. Reported live (2nd report): dizziness unconditionally hinting
        # Neurology TOO meant "sad and dizzy, mild, with nausea" produced a
        # Neurology card with zero actual neuro signs present. Neurology already
        # has its own dedicated trigger above (numbness/weakness/tremor/paralysis)
        # — a real neuro red flag still adds it correctly through that entry;
        # dizziness alone no longer needs to duplicate that.
        # Reported live (3rd report): "lightheaded"/"lightheadedness" were pulled
        # OUT of this unconditional set — clinically, bare lightheadedness (no
        # spinning sensation, no ear involvement) is ordinary General Medicine
        # territory (dehydration, low blood sugar, standing up too fast, etc.), not
        # an ENT balance-disorder complaint. It only points to ENT once it's
        # co-occurring with an actual vertigo/ear signal — see
        # _LIGHTHEADED_ENT_SIGNAL_WORDS and the co-occurrence branch below for the
        # if/else (same "bare word needs corroboration" shape as the limb/joint,
        # cough, and low-mood entries — a table entry can only express OR, never
        # "A unless B, then C instead").
        frozenset({"dizziness", "dizzy", "vertigo"}),
        ("ent", "otolaryn"),
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
        frozenset({
            "fever", "chills", "sweating", "sweat", "sweats", "sweaty",
            "ache", "aches", "aching", "body ache", "malaise", "fatigue",
        }),
        ("general medicine", "internal medicine", "family medicine"),
        "fever/body aches",
    ),
)

# Handles limb/joint pain (leg, arm, shoulder, hand, ...): a genuine if/else, not
# an OR-only table entry. Reported live (1st report): "pain in my teeth, head and
# legs as well" produced only a Dentistry card — "leg" matched no keyword at all,
# silently dropping it. Reported live (2nd report): once "leg" unconditionally
# hinted Orthopedics, a patient with mild, symptom-free limb pain who explicitly
# denied anything else got an unwanted Orthopedics card too. Reported live (3rd
# report): "leg pain after i fell" should go to Orthopedics ALONE, not both — so
# bare anatomy words route to General Medicine, but once an actual injury/red-flag
# word is ALSO present, that routes to Orthopedics INSTEAD of General Medicine, not
# alongside it (see the if/else below, not a table entry, since the table can only
# express OR per entry, never "A unless B, then C instead"). Deliberately excludes
# "difficulty"/"moving"/"move" from the injury-signal words — reported live, a
# patient who explicitly denied any issue ("i dont have any difficulty moving my
# legs and arms") would still match those words via plain keyword presence,
# negation-unaware, same known tradeoff as every other word-based check here.
# Reported live (4th report): "swelling on hand" ALONE was still routing to
# Orthopedics, but plain swelling isn't specific to injury at all (infection,
# allergic reaction, edema can all cause it) — "swelling"/"swollen" were removed
# from this set for that reason. Orthopedics now requires an actual injury/trauma
# word; plain swelling with nothing else falls back to General Medicine, same as
# any other bare limb symptom. Reported live (5th report): "swelling as well and
# i cant put weight on it" still only got General Medicine — "weight" (as in
# "can't put weight on it" / "can't bear weight") was never in this set at all,
# even though difficulty bearing weight is one of the strongest, most
# unambiguous orthopedic red flags there is. Added here; only fires alongside a
# limb/joint word already (same co-occurrence gate as every other word in this
# set), so the false-positive surface stays narrow.
# Reported live (6th report): "back pain while bending" / "my back hurts when i
# bend" still only got General Medicine — pain specifically triggered or
# worsened by bending/movement is a classic mechanical/structural back
# complaint (disc, muscular-skeletal), not a vague ache, so it belongs with the
# other movement-triggered red flags here rather than the bare-backache default.
# Reported live (7th report): "i am having difficulty in walking properly" had
# no coverage anywhere in this module — a bare mobility/gait complaint is the
# same "vague ache, safe primary-care default" shape as bare leg/arm pain: it
# could be orthopedic (joint/leg) or neurological (balance/nerve), so it
# defaults to General Medicine here, same as any other bare limb-family word,
# and still escalates to Orthopedics with a real injury/weight-bearing signal
# below, or to Neurology independently via the existing numbness/weakness
# trigger if that's also present.
_LIMB_JOINT_WORDS = frozenset({
    "leg", "legs", "knee", "ankle", "thigh", "calf",
    "back", "backache", "spine", "shoulder", "shoulders", "arm", "arms",
    "hand", "hands", "hip", "hips", "wrist", "wrists", "elbow", "elbows",
    "walk", "walking", "gait", "limp", "limping",
    # Same bare mobility complaint as "difficulty in walking" above, just standing
    # rather than walking — same default (General Medicine, escalates via the
    # existing injury/weight-bearing or numbness/weakness triggers below).
    "stand", "standing",
})
# "neck"/"necks" deliberately NOT in this set — it gets its own dedicated
# co-occurrence branch below (_NECK_NEUROLOGICAL_SIGNAL_WORDS) rather than
# folding into this General-Medicine-by-default bucket, since a real injury
# signal isn't the thing that should redirect it: numbness/weakness alongside
# neck pain means Neurology, not Orthopedics, regardless of any injury word.
_LIMB_JOINT_INJURY_SIGNAL_WORDS = frozenset({
    "redness", "bruising", "bruised", "twisted", "injury", "injured", "fell",
    "stiff", "stiffness", "weight", "bend", "bending", "bent",
    # Reported live (8th report): "leg pain after moving my leg" and "swelling
    # on my leg" should both route to Orthopedics ALONE, not the bare-limb
    # General Medicine default — pain specifically triggered by movement and
    # visible swelling are both concrete orthopedic red flags, not vague
    # aches. This reverses the earlier (4th report) removal of swelling from
    # this set: that removal was about swelling with NOTHING else present
    # being too nonspecific (infection/allergy/edema), but explicit
    # instruction now is that swelling on a limb, alongside limb pain, should
    # route to Orthopedics rather than General Medicine.
    "swelling", "swollen", "moving", "moved", "move",
})

# Companion to the "low mood" table entry above: passing mood adjectives ("sad",
# "stressed", "crying", "hopeless") only hint Psychiatry when an actual
# persistence/duration signal is also present — a single "feeling sad today"
# mentioned alongside an unrelated physical complaint shouldn't produce a full
# specialist card with zero screening, same "bare word needs corroboration" shape
# as the limb/joint and cough entries. Named clinical terms (anxiety, depression,
# panic, insomnia) are NOT in this set — those stay in the unconditional table
# entry above since they already represent a real presenting complaint on their
# own, not just a passing remark.
_LOW_MOOD_PASSING_WORDS = frozenset({
    "sad", "sadness", "stress", "stressed", "hopeless", "hopelessness", "crying",
})
_LOW_MOOD_PERSISTENCE_WORDS = frozenset({
    "days", "weeks", "months", "lately", "always", "constantly", "chronic",
    "chronically", "persistent", "persistently", "ongoing",
})

# Reported live: "i am feeling numb... in the head... having weakness" got a
# spurious General Medicine card for "head pain" — but the patient never
# described a headache at all; "head" was their answer to "where exactly are you
# feeling the numbness?", a body LOCATION for a completely different symptom
# (numbness/weakness, already correctly hinting Neurology on its own), not a pain
# complaint. Bare "head" alone is too generic a word to safely mean "headache" —
# same shape as bare "leg"/"arm" needing an injury word before meaning something
# specific. "head" only hints General Medicine when it co-occurs with an actual
# pain word; "headache"/"migraine" already inherently mean pain and stay
# unconditional in the table entry above.
_GENERIC_PAIN_WORDS = frozenset({"pain", "ache", "aches", "aching", "hurt", "hurts", "hurting", "sore"})

# Reported live: "pain in my neck and i am feeling numb" got TWO cards —
# Neurology (correctly, from the numbness/weakness table entry above) AND a
# second Orthopedics card for the same neck pain. Same complaint, not two —
# numbness/weakness alongside neck pain is a neurological presentation (possible
# nerve involvement), not a separate orthopedic one, so Orthopedics must not
# also fire here. Mirrors the same words as the numbness/weakness table entry
# above (kept as a separate constant, not a shared import, since the two lists
# serve different roles: one is a positive trigger, this one is a suppressor).
_NECK_NEUROLOGICAL_SIGNAL_WORDS = frozenset({
    "numbness", "numb", "weakness", "tremor", "tremors", "paralysis", "paralyzed",
})

# Companion to "lightheaded" being pulled out of the dizziness/vertigo table entry
# above: bare lightheadedness routes to General Medicine, but co-occurring with an
# actual ear/balance signal (vertigo, dizziness, spinning, or an ear symptom) means
# the room-is-spinning kind of lightheadedness, which is ENT's territory. "spining"
# (a common misspelling of "spinning") is included alongside the correct spelling —
# same "typo variant, not a missing concept" shape as "teeths" above.
_LIGHTHEADED_ENT_SIGNAL_WORDS = frozenset({
    "dizzy", "dizziness", "vertigo", "spinning", "spining", "spin",
    "ear", "earache", "hearing",
})


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
    joined = " ".join(patient_texts).lower()
    # Reported live: "im very light headed" never matched the "lightheaded" keyword
    # below — tokenizing splits it into separate "light"/"headed" words, same
    # "space where the keyword is one word" gap as every other fix in this table.
    # Normalizing the space/hyphen variants to the single-token spelling before
    # tokenizing catches all three ways a patient might type it.
    joined = re.sub(r"light[\s-]+headed(ness)?", r"lightheaded\1", joined)
    words = set(re.findall(r"[a-z0-9]+", joined))
    hinted_substrings: dict[str, str] = {}
    for keywords, hints, label in SYMPTOM_DEPARTMENT_HINTS:
        if words & keywords:
            for hint in hints:
                hinted_substrings.setdefault(hint, label)
    if words & _LIMB_JOINT_WORDS:
        if words & _LIMB_JOINT_INJURY_SIGNAL_WORDS:
            hinted_substrings.setdefault("ortho", "limb/joint injury symptoms")
        else:
            # Reported live: a patient described hand swelling + chest pain; the
            # LLM's OWN tool call had already (correctly) shown Orthopedics for the
            # hand, but this bare-word branch independently recomputed "hand" as
            # General Medicine and added a redundant second card for the same body
            # part under a different department name — the two branches don't know
            # about each other, only about department NAMES already covered.
            # Orthopedics and General Medicine are two possible outcomes for the
            # SAME limb/joint family (see the if/else above); if Orthopedics is
            # already covered by anything else (the LLM's own reasoning, an earlier
            # turn, an injury-signal hint elsewhere in the same message), the
            # generic fallback for this exact family is redundant and skipped.
            # Deliberately one-directional: General Medicine already being covered
            # (e.g. for an unrelated fever) must NOT suppress a genuine Orthopedics
            # addition above — that's still a real, more specific need.
            already_covered_by_ortho = any(
                re.search(r"\bortho", name.lower()) for name in already_covered
            )
            if not already_covered_by_ortho:
                for hint in ("general medicine", "internal medicine", "family medicine"):
                    hinted_substrings.setdefault(hint, "limb/joint pain")
    if words & _LOW_MOOD_PASSING_WORDS and words & _LOW_MOOD_PERSISTENCE_WORDS:
        hinted_substrings.setdefault("psych", "low mood")
    if "head" in words and words & _GENERIC_PAIN_WORDS:
        for hint in ("general medicine", "internal medicine", "family medicine"):
            hinted_substrings.setdefault(hint, "head pain")
    if words & {"neck", "necks"} and not (words & _NECK_NEUROLOGICAL_SIGNAL_WORDS):
        hinted_substrings.setdefault("ortho", "neck/joint pain")
    if words & {"lightheaded", "lightheadedness"}:
        if words & _LIGHTHEADED_ENT_SIGNAL_WORDS:
            hinted_substrings.setdefault("ent", "lightheadedness with vertigo/ear symptoms")
        else:
            for hint in ("general medicine", "internal medicine", "family medicine"):
                hinted_substrings.setdefault(hint, "lightheadedness")
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


# Symptom categories with NO reliable specific-specialty department at a typical
# clinic using this system (no Urology, no Endocrinology/growth clinic) — kept
# entirely separate from SYMPTOM_DEPARTMENT_HINTS above rather than given a
# General/internal/family medicine fallback there. Instructed live: silently
# rerouting a symptom like this to General Medicine just because SOME department
# exists is misleading — General Medicine isn't actually equipped for it, so the
# patient should get an honest, gentle apology instead of a wrong-department card.
# Each entry's `real_hint` is the genuine specialty substring to check for first
# (e.g. "urolog") — if a clinic DOES have that real department, this category is
# never "orphaned" there and departments_hinted_by_patient_symptom_words above
# already routes it correctly; `real_hint=None` means no clinic in this system
# has ever had a matching specialty for it (growth/height), so it's always
# unsupported.
_ORPHAN_SYMPTOM_CATEGORIES: tuple[tuple[frozenset[str], str, str | None], ...] = (
    (frozenset({"height", "growth", "growing", "stunted"}), "growth/height concerns", None),
    (frozenset({"kidney", "kidneys", "urinary", "urine"}), "urinary symptoms", "urolog"),
    (
        frozenset({"testicle", "testicles", "testicular", "testies", "groin", "scrotum", "scrotal"}),
        "groin/testicular symptoms",
        "urolog",
    ),
)


def unsupported_symptom_labels(
    message: str, history: list[ConversationMemory], department_names: list[str]
) -> list[str]:
    """Returns the human-readable label(s) for every ORPHAN symptom category (see
    _ORPHAN_SYMPTOM_CATEGORIES above) the PATIENT has described (this message
    and/or their own earlier turns) that this clinic genuinely has no matching
    real department for. Used to answer with a gentle, honest apology instead of
    ever silently rerouting to a department that isn't actually equipped for it."""
    patient_texts = [message] + [
        getattr(row, "content", "") or "" for row in history if getattr(row, "role", None) == "user"
    ]
    words = set(re.findall(r"[a-z0-9]+", " ".join(patient_texts).lower()))
    lowered_department_names = [name.lower() for name in department_names]
    labels = []
    for keywords, label, real_hint in _ORPHAN_SYMPTOM_CATEGORIES:
        if not (words & keywords):
            continue
        if real_hint is not None and any(
            re.search(rf"\b{re.escape(real_hint)}", name) for name in lowered_department_names
        ):
            continue
        labels.append(label)
    return labels


# Every (hint substrings, label) pairing this module uses to route a PATIENT's own
# described symptom to a department, reused here for the reverse question — "what
# symptoms does department X treat" — so that answer is built from the exact same
# vetted table, never left for an LLM to freehand and risk inventing a department
# that doesn't exist or omitting/misdescribing a real one. Mirrors every
# hinted_substrings.setdefault(...) call inside departments_hinted_by_patient_symptom_words
# above (the flat table entries, then each conditional co-occurrence branch) —
# kept as a flat list of (hints, label) tuples so a new routing rule added to
# either place doesn't silently drift out of sync with the other without a
# reviewer noticing the parallel structure.
_ALL_HINT_LABEL_PAIRS: tuple[tuple[tuple[str, ...], str], ...] = tuple(
    (hints, label) for _keywords, hints, label in SYMPTOM_DEPARTMENT_HINTS
) + (
    (("ortho",), "limb/joint injury symptoms"),
    (("general medicine", "internal medicine", "family medicine"), "limb/joint pain"),
    (("psych",), "low mood"),
    (("general medicine", "internal medicine", "family medicine"), "head pain"),
    (("ortho",), "neck/joint pain"),
    (("general medicine", "internal medicine", "family medicine"), "lightheadedness"),
    (("ent", "otolaryn"), "lightheadedness with vertigo/ear symptoms"),
)


def department_symptom_labels(department_names: list[str]) -> dict[str, list[str]]:
    """Returns {department_name: [symptom_label, ...]} — every symptom category this
    module's own routing table maps to each real, active department, in the table's
    own declaration order with duplicates removed. Used to answer "what symptoms
    does each department treat" from the same real, vetted data this module already
    uses to route a patient's own symptoms, rather than letting an LLM compose that
    list freehand and risk inventing or omitting a department."""
    labels: dict[str, list[str]] = {name: [] for name in department_names}
    for hints, label in _ALL_HINT_LABEL_PAIRS:
        for name in department_names:
            lowered_name = name.lower()
            if label in labels[name]:
                continue
            if any(re.search(rf"\b{re.escape(hint)}", lowered_name) for hint in hints):
                labels[name].append(label)
    return labels
