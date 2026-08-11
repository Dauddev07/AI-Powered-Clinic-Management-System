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
    # Breathing difficulty — explicitly includes "shortness of breath" (previously missed).
    # Negative lookbehinds guard against a patient explicitly denying it ("no shortness
    # of breath", "not short of breath", "no trouble breathing") — without them the
    # phrase itself still matched regardless of the negation word right in front of it,
    # auto-firing on a patient's own reassurance that they're NOT having trouble
    # breathing. Same technique as the bleeding-severity patterns below.
    r"(?<!not )(?<!n't )(?<!no )\bshortness of breath\b",
    r"(?<!not )(?<!n't )(?<!no )\bshort of breath\b",
    # Reported live: "i am having difficulty to breath now" (a foreign object lodged
    # in the nose, patient actively struggling to breathe) fell through entirely —
    # \s*breath required the trigger word to sit directly against "breath" with only
    # whitespace between them, so the extra "to" in this very common non-native-English
    # phrasing ("difficulty to breathe", also "trouble to breathe") broke the match.
    # The optional (?:\s+to)? closes exactly that gap without loosening the pattern
    # into a generic character-gap that would risk matching unrelated "difficulty ...
    # breathing" mentions many words apart.
    r"(?<!not )(?<!n't )(?<!no )\b(can'?t|cannot|can not|difficulty|struggling to|trouble)(?:\s+to)?\s*breath",
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
    #
    # Deliberately excludes the nose and throat, unlike every other body part here —
    # product decision: something merely resting in the nose or throat (a seed, bead,
    # piece of food) is common and not itself an emergency; it only becomes one
    # alongside an actual distress signal (can't breathe, can't swallow, choking,
    # turning blue — all matched by their own separate patterns below/above
    # regardless of whether an object is mentioned at all). A bare "X stuck in my
    # nose/throat" with no distress signal is meant to fall through to normal
    # department routing (ENT, via app.services.orchestrator.symptom_hints), not
    # auto-fire here. The eye is NOT given this same carve-out — any foreign object
    # in the eye specifically stays an unconditional red flag below.
    r"\b(nail|glass|metal|splinter|screw|knife|shard|needle|object|something (?:sharp|pointy))\b"
    r".{0,20}\b(stuck|lodged|embedded|piercing|penetrat\w*|went (?:in|into)|sticking (?:in|into|out))\b"
    r".{0,20}\b(head|skull|brain|neck|chest|abdomen|stomach|back|face|eyes?|ears?|body)\b",
    r"\b(head|skull|brain|neck|chest|abdomen|stomach|back|face|eyes?|ears?)\b.{0,20}"
    r"\b(stuck|lodged|embedded|piercing|penetrat\w*)\b",
    r"\b(stuck|lodged|embedded|piercing|penetrat\w*)\b.{0,20}\b(in|into)\b.{0,10}"
    r"\b(head|skull|brain|neck|chest|abdomen|stomach|back|face|eyes?|ears?)\b",
    r"\bforeign (object|body)\b.{0,15}\b(eyes?|head|skull|neck|chest|abdomen|body)\b",
    r"\b(something|object|nail|knife|glass|metal)\b.{0,10}\bin (my|his|her|the) "
    r"(eyes?|head|skull|neck|chest|abdomen|stomach|back)\b",
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
    # Reported live: "my leg is broken" fell through entirely — every pattern in this
    # section required either "compound fracture" or "sticking out" specifically, but
    # a patient's own plain "it's broken"/"I broke my X"/"fractured" is exactly as
    # unambiguous a severe-trauma claim and is far more common everyday phrasing than
    # either of those two clinical terms.
    r"\bbroken\b.{0,15}\b(leg|arm|hand|foot|ankle|wrist|hip|bone|finger|toe|rib|collarbone|jaw|nose)\b",
    r"\b(leg|arm|hand|foot|ankle|wrist|hip|finger|toe|rib|collarbone|jaw|nose)\b.{0,10}\bis broken\b",
    r"\bbroke\b.{0,10}\b(my|his|her|their)\b.{0,5}"
    r"\b(leg|arm|hand|foot|ankle|wrist|hip|finger|toe|rib|collarbone|jaw|nose)\b",
    r"\bfractured\b.{0,15}\b(leg|arm|hand|foot|ankle|wrist|hip|bone|finger|toe|rib|collarbone|jaw|nose)\b",
    r"\b(leg|arm|hand|foot|ankle|wrist|hip|finger|toe|rib|collarbone|jaw|nose)\b.{0,10}\bis fractured\b",
    r"\bcrush(ed|ing)\b.{0,20}\b(leg|arm|hand|foot|limb|chest|head)\b",
    r"\b(leg|arm|hand|foot|limb|chest|head)\b.{0,20}\bcrush(ed|ing)\b",
    r"\bimpaled\b",
    # Choking / airway obstruction. Deliberately no bare "X stuck in my throat"
    # pattern here — product decision (see the foreign-object section above): that
    # alone, with no distress signal, is meant to fall through to ENT department
    # routing rather than auto-firing. These patterns are exactly the distress
    # signals that make a throat/nose object into a genuine emergency instead.
    r"\bchoking\b",
    r"\bchoked on\b",
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
    # (torsion) regardless of how mild the rest of the message sounds. The semantic
    # layer no longer backstops this category at all (see _EXEMPLARS' docstring in
    # this file for why it was removed), so this regex coverage is now the ONLY
    # automatic detection for it — every natural word order needs its own pattern,
    # not just the ones a first pass happened to cover.
    r"\bsudden\b.{0,15}\b(testicular|testicle|scrotal|scrotum)\b.{0,15}\bpain\b",
    r"\b(testicular|testicle|scrotal|scrotum)\b.{0,20}\b(severe|sudden|sharp)\b.{0,10}\bpain\b",
    r"\b(severe|sudden|sharp)\b.{0,10}\bpain\b.{0,20}\b(testicular|testicle|scrotal|scrotum)\b",
    # The plain adjective-noun-noun order ("severe testicle pain", mirroring the
    # already-covered "severe chest pain" construction elsewhere in this file) was
    # missing entirely — none of the three patterns above match "severe" ...
    # "testicle" ... "pain" in that exact order.
    r"\b(severe|sudden|sharp)\b.{0,10}\b(testicular|testicle|scrotal|scrotum)\b.{0,10}\bpain\b",
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
# bites, unconsciousness, suicidal ideation, amputation, vehicle collisions, and eye
# foreign objects. Chest pain, bleeding, burns, choking, and fractures rely on the
# regex patterns (and the LLM's own PATH 1/2 judgment) alone, not this semantic layer.
# Anaphylaxis is ALSO excluded, for a different reason: an exemplar built around
# "allergic reaction" scored a benign personal-recall question ("what allergy did I
# mention earlier?") at 0.6 similarity purely off the shared word "allergy" — a plain
# mention of the word is common and unremarkable (medical history, profile info, small
# talk), so this category isn't safe to include without a much larger negative-
# calibration set than is practical here. Anaphylaxis still has solid regex coverage
# on its own ("anaphylax", throat/face swelling).
#
# Testicular torsion was ALSO removed after real false positives, for the SAME reason
# as chest pain/bleeding/burns above (not a distinct new failure mode): "i am having
# pain in my testies" (no severity/suddenness/swelling stated at all) scored 0.61
# against the exemplar — even the margin layer (see _SEMANTIC_MARGIN below) couldn't
# save it, since the message is genuinely topically on-target, just missing any
# urgency signal, and three separate reworded candidates (adding "swelling", "within
# minutes", "unlike ordinary discomfort") all still scored plain mentions at ~0.51-0.52
# while barely improving true-positive separation. This confirms testicular pain
# belongs in the same "severity can't be inferred from topic alone" bucket as chest
# pain/bleeding/burns, not the "unambiguous once identified" bucket this list is
# for — it should never have been included here in the first place. The regex
# patterns above (requiring "sudden"/"severe"/"sharp" alongside "testicular"/
# "testicle"/"scrotal"/"scrotum") still auto-fire correctly for anyone using that
# vocabulary; a genuine torsion described in neither clinical nor severity language
# (e.g. "excruciating pain in one of my balls that started suddenly") now relies on
# the LLM's own PATH 1/2 judgment alone, same as chest pain/bleeding/burns already do.
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
    # Reported live: "i am having pain in my leg and in my hands as well" — ordinary
    # multi-limb PAIN, nothing stroke-related at all — scored 0.5012 against the
    # original, looser wording of this exemplar ("sudden weakness or numbness on one
    # side of the body"), just barely over threshold, purely off shared limb
    # vocabulary. That score sat almost exactly on top of a genuine true positive
    # already in this bank ("she went over the balcony railing", 0.5003), so no single
    # threshold could separate the two — the exemplar itself needed to be more
    # specific. Tying it explicitly to "one side" + a named body part combination
    # (facial drooping alongside limb weakness — the actual FAST stroke pattern, not
    # just "numbness" in isolation) drops the false positive to ~0.41 while every
    # existing true-positive paraphrase in tests/test_red_flag.py still scores well
    # above 0.5 against it.
    "sudden facial drooping with weakness or numbness down one side of the body",
    "swallowed poison or overdosed on pills",
    "accidentally drank a toxic chemical or cleaning product",
    "electrocuted by electricity or a live wire",
    "drowning in water and not breathing",
    "a near drowning incident",
    "gunshot wound",
    "stabbed with a knife",
    "fell from a height like a roof, ladder, or balcony",
    "unconscious and not responding",
    "collapsed and will not wake up",
    # Reported live: "Psychiatry in this dept" (a patient simply naming the
    # department to book, right after being told "Psychiatry" wasn't recognized as
    # typed) scored 0.5071 against the original, looser wording of this exemplar
    # ("suicidal thoughts or wanting to end one's life") — purely off shared
    # mental-health-adjacent vocabulary, nothing about self-harm at all. Anchoring
    # the exemplar to an explicit first-person feeling ("feeling suicidal", not
    # just topically "suicidal thoughts") drops ordinary Psychiatry-booking
    # messages to <=0.47 while real suicidal-ideation paraphrases with no shared
    # regex vocabulary ("I feel like ending it all", "thoughts of not wanting to
    # be alive") still score >=0.65 against it.
    "feeling suicidal and thinking about ending my life",
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

# MARGIN LAYER — added after three separate real false positives (short, vague, or
# topically-adjacent messages each individually scoring just over
# _SEMANTIC_SIMILARITY_THRESHOLD against one specific _EXEMPLARS entry: "the pain is
# mild and moderate" vs. testicular pain, "my hands and legs hurt" vs. stroke,
# "Psychiatry in this dept" vs. suicidal ideation) required three separate rounds of
# hand-tuning individual exemplar wording. Rewording each exemplar after the fact
# works but doesn't generalize — it only fixes the one reported phrasing, not the
# underlying shape of the problem: an absolute similarity-to-emergency-exemplars
# threshold alone can't tell "this message is actually about an emergency" apart from
# "this message is short/vague/topically-adjacent and therefore drifts a bit toward
# EVERY exemplar bank, including the emergency one."
#
# The structural fix is RELATIVE, not absolute: also embed a bank of ordinary/benign
# clinic messages (_BENIGN_EXEMPLARS below — routine booking, mild/hedged symptom
# descriptions, small talk, personal info) and only fire when a message is closer to
# the emergency bank than to the benign bank by a real margin, not just past a fixed
# floor. A benign message that happens to drift toward one emergency exemplar
# (embeddings are noisy for short text) almost always drifts AT LEAST as far toward
# the benign bank too — the margin cancels that shared noise out. A genuine emergency
# paraphrase does not have this problem: it's close to the emergency bank and FAR from
# the benign one, so its margin stays wide even though its absolute similarity might
# be similar to a false positive's.
#
# Calibrated against the full existing true-positive/negative test suite (see the
# calibration test class in tests/test_red_flag.py): every semantic-only true positive
# has margin >= 0.136 (the tightest: the testicular-pain paraphrase against a "mild
# ache" benign exemplar); every negative/PATH-2/false-positive message in the test
# suite — including all three historical false positives above — has margin <= 0.012.
# 0.08 sits in the middle of that gap with real headroom on both sides. Like the
# absolute threshold, this is NOT a universal constant: if either exemplar bank
# changes, re-run the calibration.
_SEMANTIC_MARGIN = 0.08

# Representative ordinary/benign clinic messages — deliberately drawn from the SAME
# shapes that have caused real false positives (vague/hedged symptom descriptions,
# plain department-booking phrasing, personal info, small talk) so the margin layer
# above actually cancels the noise those shapes produce, rather than being a generic
# unrelated negative set that happens not to help.
_BENIGN_EXEMPLARS: tuple[str, ...] = (
    "I want to book an appointment in a department",
    "can I see a doctor for a routine checkup",
    "show me available appointment slots",
    "can you recommend a department for my symptoms",
    "I'd like to reschedule my appointment",
    "my symptoms are mild and manageable",
    "the pain is not severe, just a little uncomfortable",
    "it's been going on for a few days, nothing too serious",
    "I have a minor ache that comes and goes",
    "hi, my name is",
    "what is my name and what have I told you before",
    "what are your clinic hours",
    "thanks, that's helpful",
)

_exemplar_vectors = np.array(embed_texts(list(_EXEMPLARS)))
_exemplar_vectors = _exemplar_vectors / np.linalg.norm(_exemplar_vectors, axis=1, keepdims=True)

_benign_vectors = np.array(embed_texts(list(_BENIGN_EXEMPLARS)))
_benign_vectors = _benign_vectors / np.linalg.norm(_benign_vectors, axis=1, keepdims=True)


def _semantic_red_flag(message: str) -> bool:
    if not message or not message.strip():
        return False
    vector = np.array(embed_texts([message])[0])
    norm = np.linalg.norm(vector)
    if norm == 0:
        return False
    vector = vector / norm
    emergency_similarity = float(np.max(_exemplar_vectors @ vector))
    benign_similarity = float(np.max(_benign_vectors @ vector))
    return (
        emergency_similarity >= _SEMANTIC_SIMILARITY_THRESHOLD
        and emergency_similarity - benign_similarity >= _SEMANTIC_MARGIN
    )


_SEVERITY_QUESTION_RE = re.compile(r"\bsever(e|ity)\b", re.IGNORECASE)
_MILD_RE = re.compile(r"\bmild\b", re.IGNORECASE)
_SEVERE_WORD_RE = re.compile(r"\bsevere\b", re.IGNORECASE)
# Negation window: up to 3 words between a negation word and "severe" — covers
# "not that severe", "isn't really severe", "no, not severe", not just "not severe"
# with nothing between them. A fixed-width lookbehind (the style used elsewhere in
# this file, e.g. the bleeding-severity patterns) can't stretch across an
# intervening word like "that"/"really", so this uses a bounded forward window
# instead.
_NEGATED_SEVERE_RE = re.compile(r"\b(not|n't|no)\b(?:\s+\S+){0,3}\s+severe\b", re.IGNORECASE)


def detect_severity_escalation(message: str, last_assistant_message: str | None) -> bool:
    """Deterministic backstop for PATH 2's own "a severe answer -> PATH 1 immediately"
    rule (see llm.py's _TRIAGE_SECTION). Reported live TWICE: "its very severe and
    since 10 days" and, after the prompt was already strengthened once to say extra
    reply detail must never override an explicit "severe" answer, a bare "its severe"
    on its own STILL routed to PATH 3 instead of PATH 1 for a headache. A prompt
    instruction alone is not a guarantee — same principle as detect_red_flag() above
    — so this is a plain server-side check for this exact conversational pattern
    instead of trusting the model to comply a third time.

    Fires only when BOTH hold: the assistant's own immediately-preceding turn reads
    as a severity-screening question (contains both "severe"/"severity" and "mild" —
    the scale's two named endpoints, e.g. "is it mild, moderate, or severe?"), and
    the patient's reply affirms severe (not negated, and not just "moderate"/"mild").
    Deliberately narrow: an unrelated "severe" mention with no immediately-preceding
    severity question does not fire here — this is a backstop for this one reported
    failure mode, not a general severity detector.
    """
    if not last_assistant_message:
        return False
    if not (_SEVERITY_QUESTION_RE.search(last_assistant_message) and _MILD_RE.search(last_assistant_message)):
        return False
    if not _SEVERE_WORD_RE.search(message):
        return False
    return not _NEGATED_SEVERE_RE.search(message)


RED_FLAG_MESSAGE_EN = (
    "This may be a medical emergency. Please call 1122 or go to "
    "the nearest emergency room right away. This assistant cannot handle emergencies.\n\n"
    "While you're on your way:\n"
    "- Stay as calm and still as possible, and avoid moving any injured area more than necessary.\n"
    "- Do not eat, drink, or take any medication unless a medical professional tells you to.\n"
    "- If there is visible bleeding, apply firm, steady pressure with a clean cloth until help arrives.\n"
    "- If possible, don't go alone — have someone accompany you or call an ambulance rather than driving yourself."
)


def detect_red_flag(message: str) -> bool:
    """True if EITHER layer fires: the regex gate (exact/loose wording matches) OR the
    semantic similarity check (paraphrases with no shared vocabulary with any regex
    pattern). Either one alone is sufficient — this is a union, not a requirement that
    both agree."""
    return bool(_RED_FLAG_RE.search(message)) or _semantic_red_flag(message)


def red_flag_message() -> str:
    """Always English — Urdu removed per product decision; a red-flagged reply must
    never be translated at request time (an LLM asked to translate a life-safety
    message risks softening it), and a second hardcoded-Urdu variant isn't maintained
    alongside the English one."""
    return RED_FLAG_MESSAGE_EN
