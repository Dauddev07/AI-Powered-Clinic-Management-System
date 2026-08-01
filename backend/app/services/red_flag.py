"""Server-side emergency (red-flag) detection.

Deliberately NOT delegated to the LLM: the system prompt also instructs the model to
defer to emergency services, but a prompt instruction alone is not a guarantee (it can
be missed, ignored, or reasoned around) — this regex gate runs before any LLM call and
short-circuits triage/booking/KB retrieval entirely when it fires, exactly the same way
for every message regardless of what the model would have done with it.

Patterns are deliberately broad (erring toward false positives) since the cost of an
unnecessary urgent-routing message is far lower than missing a genuine emergency.

SEMANTIC LAYER: the regex patterns above only ever match a message that actually
contains one of their literal words/phrasings — a patient who describes the exact same
emergency in different words entirely (no shared vocabulary with any pattern) slips
through. detect_red_flag() therefore ALSO runs a semantic similarity check: the
incoming message and a curated bank of example emergency descriptions (_EXEMPLARS
below, covering the same categories as the regex patterns) are embedded with this
app's existing local sentence-transformers model (app.rag.embeddings — already loaded
once at process start for KB retrieval, not a second model), and a message is treated
as a red flag if its cosine similarity to the closest exemplar clears
_SEMANTIC_SIMILARITY_THRESHOLD. This is a best-effort supplementary layer, not a
replacement for the regex gate or the LLM's own PATH 1/2 judgment in
app.services.llm — dense sentence embeddings do not cleanly separate every possible
phrasing, so some genuine emergencies described in a very unusual way may still miss
both layers, same as any heuristic system. The threshold (0.5) was picked empirically
against a real paraphrase/benign test set (see tests/test_red_flag.py) rather than
guessed, since ungrounded thresholds are exactly the kind of thing that silently drifts
wrong.
"""
import re

import numpy as np

from app.rag.embeddings import embed_texts

# Each pattern matches loosely-phrased real-world wording, not just clinical terms —
# a patient describing an emergency rarely uses textbook language.
_RED_FLAG_PATTERNS = [
    # Cardiac: an explicit self-identified "heart attack" still auto-fires — that's
    # already the patient's own worst-case read of it. Plain "chest pain"/"chest
    # tightness"/"chest pressure" deliberately do NOT auto-fire here anymore: those
    # range anywhere from a pulled muscle to a real cardiac emergency, so product
    # policy is to have the chat agent itself ask a same-turn severity screening
    # question (severe/bearable/mild + a second differentiator) before deciding —
    # see the AGENT_SYSTEM_PROMPT's PATH 2 in app/services/llm.py — rather than
    # blanket-firing the canned redirect on the word "chest pain" alone with no
    # chance to ask anything first.
    r"\bheart\s*attack\b",
    # Breathing difficulty — explicitly includes "shortness of breath" (previously missed)
    r"\bshortness of breath\b",
    r"\bshort of breath\b",
    r"\b(can'?t|cannot|can not|difficulty|struggling to|trouble)\s*breath",
    r"\bnot breathing\b",
    r"\bgasping\b",
    r"\bsuffocat",
    # Stroke signs (FAST: face, arms, speech, time)
    r"\b(face|mouth)\b.{0,15}\b(drooping|dropping|numb|sagging|droopy)\b",
    r"\bslurred speech\b",
    r"\bspeech\b.{0,15}\bslurred\b",
    r"\bsudden(?:ly)?\s*(numb|numbness|weakness)\b.{0,20}\b(one side|face|arm|leg)\b",
    r"\b(one side|left side|right side)\b.{0,20}\b(numb|weak|paraly)",
    r"\bstroke\b",
    r"\bcan'?t speak\b",
    r"\bsudden confusion\b",
    # Severe bleeding — both word orders ("severe bleeding" and "bleeding severely"),
    # everyday (non-clinical) severity phrasing ("bleeding very much", "bleeding a
    # ton", "bleeding so much"), and a FUZZY match for "severe(ly)" (\bsever\w{0,4}ly\b)
    # rather than an enumerated list of specific misspellings — a patient describing a
    # real injury in a hurry can type "severly", "severaly", "severley", etc., and
    # trying to enumerate every possible typo one at a time is a losing battle; this
    # matches any "sever" + up to 4 extra characters + "ly" instead.
    # Negative lookbehinds guard against a patient explicitly denying severity
    # ("not bleeding severely", "isn't bleeding badly", "no heavy bleeding") — without
    # them the phrase itself still matched regardless of the negation word right in
    # front of it, auto-firing on a patient's own reassurance that it's NOT severe.
    r"(?<!not )(?<!n't )(?<!no )\b(sever\w{0,4}ly|severe|heavy|heavily|uncontrolled|won'?t stop|not stopping|"
    r"a lot|badly|profusely|excessively)\s*bleed",
    r"(?<!not )(?<!n't )(?<!no )\bbleed\w*\s*(sever\w{0,4}ly|severe|heavy|heavily|a lot|a ton|so much|very much|"
    r"won'?t stop|badly|profusely|excessively)\b",
    r"\bblood.{0,15}(everywhere|won'?t stop)\b",
    # Vehicle accidents / high-energy trauma — the mechanism itself (being struck by
    # or involved in a vehicle collision) is a red flag on its own, regardless of how
    # the patient phrases the resulting injury.
    r"\b(hit|struck|run over|ran over)\b.{0,15}\b(by|with)\b.{0,10}\b(a |the )?(car|truck|bike|motorcycle|"
    r"bus|vehicle)\b",
    r"\b(car|truck|bike|motorcycle|bus|vehicle)\b.{0,15}\b(hit|struck|ran over)\b",
    r"\b(car|truck|bike|motorcycle|bus|vehicle|road)\s*(accident|crash|collision)\b",
    r"\bhit and run\b",
    r"\bcollided\b",
    # Foreign object / penetrating injury to ANY body part — not just the eye. A nail,
    # knife, screw, splinter, or similar object stuck/lodged/embedded/piercing the
    # head, skull, neck, chest, abdomen, back, or any other body part is exactly as
    # much an emergency as one in the eye (arguably more so for the head/chest/
    # abdomen), so this is not eye-specific.
    r"\b(nail|glass|metal|splinter|screw|knife|shard|needle|object|something (?:sharp|pointy))\b"
    r".{0,20}\b(stuck|lodged|embedded|piercing|penetrat\w*|went (?:in|into)|sticking (?:in|into|out))\b"
    r".{0,20}\b(head|skull|brain|neck|throat|chest|abdomen|stomach|back|face|eyes?|ears?|body)\b",
    r"\b(head|skull|brain|neck|throat|chest|abdomen|stomach|back|face|eyes?|ears?)\b.{0,20}"
    r"\b(stuck|lodged|embedded|piercing|penetrat\w*)\b",
    r"\b(stuck|lodged|embedded|piercing|penetrat\w*)\b.{0,20}\b(in|into)\b.{0,10}"
    r"\b(head|skull|brain|neck|throat|chest|abdomen|stomach|back|face|eyes?|ears?)\b",
    r"\bforeign (object|body)\b.{0,15}\b(eyes?|head|skull|neck|chest|abdomen|body)\b",
    r"\b(something|object|nail|knife|glass|metal)\b.{0,10}\bin (my|his|her|the) "
    r"(eyes?|head|skull|neck|chest|abdomen|stomach|back|throat)\b",
    # Chemical exposure to the eyes or skin (splashed/sprayed/got in) — distinct from
    # the ingestion/poisoning patterns above, since exposure rather than swallowing is
    # its own recognized emergency (chemical burns, blindness risk).
    r"\b(acid|chemical|bleach)\b.{0,15}\b(in|on)\b.{0,10}\b(my|his|her|the)\b.{0,10}\b(eyes?|skin|face)\b",
    r"\b(splashed|sprayed|got)\b.{0,15}\b(acid|chemical|bleach)\b",
    # Loss of consciousness / seizure
    r"\bunconscious\b",
    r"\bunresponsive\b",
    r"\bpassed out\b",
    r"\bfainted\b",
    r"\bseizure\b",
    r"\bconvuls",
    # Severe allergic reaction
    r"\banaphylax",
    r"\bthroat.{0,15}(closing|swelling|swollen)\b",
    r"\bface.{0,15}swelling\b",
    # Suicidal / self-harm crisis
    r"\bsuicid",
    r"\bkill(ing)? myself\b",
    r"\bwant to die\b",
    r"\bhurt(ing)? myself\b",
    # Severe trauma / limb loss — previously missed entirely (a message like "it is
    # detached" or "I am missing a leg" fell through to the generic triage agent
    # instead of the emergency redirect, since no pattern here covered amputation or
    # severed/missing limbs).
    r"\bamputat\w*\b",
    r"\b(severed|detached)\b.{0,25}\b(leg|arm|hand|foot|finger|toe|limb)\b",
    r"\b(leg|arm|hand|foot|finger|toe|limb)\b.{0,25}\b(severed|detached|cut off|chopped off|torn off|ripped off)\b",
    r"\bmissing\b.{0,10}\b(a|an|my|one)\b.{0,10}\b(leg|arm|hand|foot|finger|toe|limb)\b",
    r"\bcompound fracture\b",
    r"\bbone\b.{0,15}\bsticking out\b",
    r"\bcrush(ed|ing)\b.{0,20}\b(leg|arm|hand|foot|limb|chest|head)\b",
    r"\b(leg|arm|hand|foot|limb|chest|head)\b.{0,20}\bcrush(ed|ing)\b",
    r"\bimpaled\b",
    # Choking / airway obstruction
    r"\bchoking\b",
    r"\bchoked on\b",
    r"\b(something|food|it)\b.{0,15}\bstuck in (my|his|her|the) throat\b",
    r"\bcan'?t swallow\b",
    r"\bturning blue\b",
    r"\b(lips|skin|face)\b.{0,10}\b(turning|is|are|went)\b.{0,10}\bblue\b",
    # Poisoning / overdose / toxic ingestion
    r"\b(swallowed|drank|drunk|ingested)\b.{0,20}\b(poison|bleach|chemical|acid|pills|medicine|detergent)\b",
    r"\b(took|took too many|overdosed on)\b.{0,15}\bpills\b",
    r"\boverdos\w*\b",
    r"\bpoison(ed|ing)?\b",
    # Severe burns
    r"\b(severe|bad|badly|serious|third[- ]degree|large)\s*burn",
    r"\bburned?\b.{0,15}\b(badly|severely|all over|a lot)\b",
    r"\b(badly|severely)\b.{0,15}\bburned?\b",
    r"\bburning\b.{0,15}\ball over\b",
    r"\bcaught fire\b",
    r"\bon fire\b",
    # Electrocution
    r"\belectrocut\w*\b",
    r"\belectric(al)? shock\b",
    r"\bshocked by\b.{0,15}\b(wire|socket|outlet|electricity)\b",
    # Drowning / near-drowning
    r"\bdrown(ed|ing)?\b",
    r"\bnear[- ]drowning\b",
    r"\bpulled\b.{0,15}\bout of (the )?(water|pool|river|sea)\b.{0,20}\bnot breathing\b",
    # Gunshot / stab wounds / weapon injuries
    r"\bgun\s*shot\b",
    r"\bshot\b.{0,15}\b(with a gun|by a gun|gun)\b",
    r"\b(been |got )?shot\b.{0,10}\b(in|on)\b.{0,10}\b(chest|head|arm|leg|stomach|back)\b",
    r"\bstabbed\b",
    r"\bstab wound\b",
    r"\bknife wound\b",
    # Fall from height
    r"\bfell\b.{0,15}\bfrom\b.{0,10}\b(the )?(roof|ladder|stairs|balcony|building|height|tree|window)\b",
    r"\bfall(?:ing)?\b.{0,15}\bfrom\b.{0,10}\b(a |the )?(roof|ladder|height|building|window)\b",
    # Venomous bite / sting — deliberately NOT a bare "bite"/"bitten" pattern (that
    # would also catch an everyday dog/cat bite or insect bite, which is PATH 2
    # territory, not an automatic red flag) — scoped to snake/scorpion specifically.
    r"\bsnake\s*bite\b",
    r"\bbit(?:ten)? by a snake\b",
    # Reverse word order ("a snake bit me") — the semantic layer used to catch this
    # via a "bitten by a venomous snake" exemplar, but that exemplar also false-fired
    # on an everyday dog bite (species isn't something general embeddings reliably
    # separate) and was removed; this regex closes the same word-order gap directly
    # instead, without reintroducing that false positive.
    r"\bsnake\b.{0,15}\bbit",
    # Named venomous species, either word order — the same false-positive problem
    # ruled out a generic semantic "venomous snake/scorpion" exemplar, so specific
    # common venomous species names are covered directly by regex instead.
    r"\b(cobra|viper|rattlesnake|python|krait|mamba|copperhead|water\s*moccasin|"
    r"coral\s*snake)\b.{0,15}\bbit",
    r"\bbit(?:ten)?\b.{0,15}\bby\b.{0,10}\b(a |the )?(cobra|viper|rattlesnake|python|"
    r"krait|mamba|copperhead|water\s*moccasin|coral\s*snake)\b",
    r"\bscorpion sting\b",
    r"\bstung by a scorpion\b",
    # Sudden severe testicular/scrotal pain — a recognized time-critical emergency
    # (torsion) regardless of how mild the rest of the message sounds.
    r"\bsudden\b.{0,15}\b(testicular|testicle|scrotal|scrotum)\b.{0,15}\bpain\b",
    r"\b(testicular|testicle|scrotal|scrotum)\b.{0,20}\b(severe|sudden|sharp)\b.{0,10}\bpain\b",
    r"\b(severe|sudden|sharp)\b.{0,10}\bpain\b.{0,20}\b(testicular|testicle|scrotal|scrotum)\b",
    # Head injury combined with a recognized danger sign — a bump on the head alone
    # is common and not auto-fired (see PATH 2 in llm.py for ambiguous head pain),
    # but a head injury plus vomiting, confusion, or memory loss is a well-known
    # red-flag combination worth auto-firing on its own.
    r"\bhit\b.{0,10}\b(my|his|her|their)\b.{0,5}\bhead\b.{0,40}\b(vomit|confus\w*|can'?t remember|"
    r"blacked out|lost consciousness)",
    r"\bhead injury\b.{0,40}\b(vomit|confus\w*|can'?t remember|blacked out)",
    # Spinal injury signs after a fall, accident, or trauma — inability to feel or
    # move a limb is a recognized time-critical warning sign regardless of how the
    # rest of the message is phrased.
    r"\bcan'?t (feel|move)\b.{0,15}\b(my|his|her|their)\b.{0,10}\b(legs?|arms?|hands?|feet)\b",
    r"\b(paraly\w*|numb)\b.{0,20}\bafter\b.{0,15}\b(fall|accident|crash)\b",
    # Heat stroke / severe hypothermia
    r"\bheat\s*stroke\b",
    r"\b(severe|extreme)\s*(dehydration|hypothermia)\b",
    r"\bhypothermi\w*\b",
    r"\bshivering uncontrollably\b",
    r"\bcollapsed\b.{0,15}\b(from|in|due to)\b.{0,10}\bheat\b",
    # Severe asthma attack / rescue inhaler not working
    r"\basthma attack\b.{0,30}\b(not working|not helping|can'?t breathe)\b",
    r"\binhaler\b.{0,20}\b(not working|not helping|isn'?t working)\b",
]

_RED_FLAG_RE = re.compile("|".join(_RED_FLAG_PATTERNS), re.IGNORECASE)

# Natural-language exemplars of a DELIBERATELY LIMITED subset of the regex categories
# above — several variants per category so a paraphrase only needs to land near ANY
# one of them, not one fixed canonical phrasing. This subset is smaller than the full
# regex category list on purpose: categories whose regex patterns exist precisely
# BECAUSE severity can't be inferred from topic alone (chest pain, bleeding, burns, a
# broken bone) are intentionally excluded here. Empirically (see the calibration notes
# in tests/test_red_flag.py), a general-purpose sentence embedding places e.g. "chest
# pressure and tightness" and "something stuck in my throat" close enough together
# that including both chest-pain AND choking exemplars collapses the gap between
# genuine emergencies and the deliberately-non-auto-firing PATH 2 cases (see
# app.services.llm's PATH 2) — there is no threshold that cleanly separates both at
# once. So this list only covers categories that are unambiguous once identified at
# all (no "mild version" that should route to PATH 2 instead): stroke signs,
# poisoning, electrocution, drowning, weapon injuries, falls from height, venomous
# bites, testicular torsion, unconsciousness, suicidal ideation, amputation, vehicle
# collisions, and eye foreign objects. Chest pain, bleeding, burns, choking, and
# fractures rely on the regex patterns (and the LLM's own PATH 1/2 judgment) alone,
# not this semantic layer. Anaphylaxis is ALSO excluded, for a different reason: an
# exemplar built around "allergic reaction" scored a benign personal-recall question
# ("what allergy did I mention earlier?") at 0.6 similarity purely off the shared word
# "allergy" — a plain mention of the word is common and unremarkable (medical history,
# profile info, small talk), so this category isn't safe to include without a much
# larger negative-calibration set than is practical here. Anaphylaxis still has solid
# regex coverage on its own ("anaphylax", throat/face swelling).
#
# Two exemplars were tried and REMOVED after real false positives: "bitten by a
# venomous snake or scorpion" scored 0.561 against a plain "i got bitten by a dog"
# (species isn't something these embeddings reliably separate), and "a finger or
# limb completely cut off" scored 0.55+ against ordinary kitchen cuts ("i got a cut
# on my hand", "i cut my finger cutting an apple") — severity isn't reliably
# separated from topic here either, same failure mode as chest pain/bleeding/burns
# above. Both categories already have solid, precise regex coverage on their own
# (\bsnake\s*bite\b / "bit(ten) by a snake"; the severed/detached/"cut off" + body-
# part pattern), so dropping them from the semantic bank loses no real true-positive
# coverage, only the false positives. "amputated or severed limb" stays — it covers
# phrasing regex doesn't (bare "amputated", no "cut off"/"severed" verb) and tested
# clean (<0.4) against the same benign-cut/bite messages.
_EXEMPLARS: tuple[str, ...] = (
    "signs of a stroke like face drooping or slurred speech",
    "sudden weakness or numbness on one side of the body",
    "swallowed poison or overdosed on pills",
    "accidentally drank a toxic chemical or cleaning product",
    "electrocuted by electricity or a live wire",
    "drowning in water and not breathing",
    "a near drowning incident",
    "gunshot wound",
    "stabbed with a knife",
    "fell from a height like a roof, ladder, or balcony",
    "sudden severe testicular pain",
    "unconscious and not responding",
    "collapsed and will not wake up",
    "suicidal thoughts or wanting to end one's life",
    "thinking about killing myself or self harm",
    "amputated or severed limb",
    "hit by a car or involved in a vehicle collision",
    "an object like glass or metal embedded in the eye",
)

# Picked empirically against this exact _EXEMPLARS list (see the calibration test
# class in tests/test_red_flag.py): every negative/PATH-2 message in the existing test
# suite scores <= 0.48 against the closest exemplar, and the true-emergency paraphrases
# this layer is meant to catch score >= 0.51 — 0.5 sits cleanly in that gap. This is
# NOT a universal constant: if _EXEMPLARS changes, re-run the calibration rather than
# assuming 0.5 still holds — it was derived from data, not chosen a priori.
_SEMANTIC_SIMILARITY_THRESHOLD = 0.5

_exemplar_vectors = np.array(embed_texts(list(_EXEMPLARS)))
_exemplar_vectors = _exemplar_vectors / np.linalg.norm(_exemplar_vectors, axis=1, keepdims=True)


def _semantic_red_flag(message: str) -> bool:
    if not message or not message.strip():
        return False
    vector = np.array(embed_texts([message])[0])
    norm = np.linalg.norm(vector)
    if norm == 0:
        return False
    vector = vector / norm
    similarity = float(np.max(_exemplar_vectors @ vector))
    return similarity >= _SEMANTIC_SIMILARITY_THRESHOLD


RED_FLAG_MESSAGE_EN = (
    "This may be a medical emergency. Please call your local emergency number or go to "
    "the nearest emergency room right away. This assistant cannot handle emergencies.\n\n"
    "While you're on your way:\n"
    "- Stay as calm and still as possible, and avoid moving any injured area more than necessary.\n"
    "- Do not eat, drink, or take any medication unless a medical professional tells you to.\n"
    "- If there is visible bleeding, apply firm, steady pressure with a clean cloth until help arrives.\n"
    "- If possible, don't go alone — have someone accompany you or call an ambulance rather than driving yourself."
)

RED_FLAG_MESSAGE_UR = (
    "یہ ایک طبی ایمرجنسی ہو سکتی ہے۔ براہ کرم فوری طور پر ایمرجنسی سروس کو کال کریں یا "
    "قریب ترین ایمرجنسی روم جائیں۔ یہ اسسٹنٹ ایمرجنسی صورتحال میں مدد نہیں کر سکتا۔\n\n"
    "راستے میں یہ کریں:\n"
    "- جتنا ممکن ہو پرسکون اور ساکت رہیں، اور زخمی حصے کو غیر ضروری حرکت سے بچائیں۔\n"
    "- جب تک کوئی طبی ماہر نہ کہے، کچھ کھائیں پئیں یا کوئی دوا نہ لیں۔\n"
    "- اگر خون بہہ رہا ہو تو صاف کپڑے سے مسلسل، مضبوط دباؤ ڈالیں جب تک مدد نہ پہنچے۔\n"
    "- اکیلے جانے کے بجائے کسی کو ساتھ لے جائیں یا خود گاڑی چلانے کے بجائے ایمبولینس کو کال کریں۔"
)


def detect_red_flag(message: str) -> bool:
    """True if EITHER layer fires: the regex gate (exact/loose wording matches) OR the
    semantic similarity check (paraphrases with no shared vocabulary with any regex
    pattern). Either one alone is sufficient — this is a union, not a requirement that
    both agree."""
    return bool(_RED_FLAG_RE.search(message)) or _semantic_red_flag(message)


def red_flag_message(language: str) -> str:
    return RED_FLAG_MESSAGE_UR if language == "ur" else RED_FLAG_MESSAGE_EN
