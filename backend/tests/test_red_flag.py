import pytest

from app.services.red_flag import (
    RED_FLAG_MESSAGE_EN,
    detect_red_flag,
    red_flag_message,
)

# --- chest pain / cardiac ------------------------------------------------------------


def test_explicit_heart_attack_still_fires():
    # An explicit self-identified "heart attack" is already the patient's own
    # worst-case read of it, so this still auto-fires straight to the emergency
    # redirect with no screening question.
    assert detect_red_flag("I think I'm having a heart attack")


@pytest.mark.parametrize(
    "message",
    [
        "I have severe chest pain",
        "chest pressure and tightness",
        "there is a squeezing pain in my chest",
    ],
)
def test_plain_chest_pain_no_longer_auto_fires(message):
    # Product decision: plain chest pain/pressure/tightness (without an explicit
    # "heart attack") deliberately does NOT auto-fire anymore, even with "severe"
    # stated — it ranges from a pulled muscle to a real cardiac emergency, so the
    # chat agent itself screens it via PATH 2 and decides PATH 1 vs. PATH 3 from
    # the answer (see app.services.llm._TRIAGE_SECTION). A blanket server-side
    # bare-"severe" rule briefly lived here (see detect_red_flag's own docstring
    # for why it was added and then removed) — the actual bug it was compensating
    # for turned out to be in symptom_agent's own reply-recovery logic discarding
    # a valid PATH 1 reply, now fixed at that source instead.
    assert not detect_red_flag(message)


# --- breathing difficulty, including "shortness of breath" specifically -------------


@pytest.mark.parametrize(
    "message",
    [
        "I have shortness of breath",
        "I am short of breath",
        "I can't breathe properly",
        "she is gasping for air",
        "he says he is suffocating",
    ],
)
def test_breathing_difficulty_patterns_fire_including_shortness_of_breath(message):
    assert detect_red_flag(message)


def test_shortness_of_breath_specifically_fires():
    # Called out explicitly in the SOW as a phrase the bot previously missed.
    assert detect_red_flag("I have shortness of breath and feel dizzy")


@pytest.mark.parametrize(
    "message",
    [
        "no shortness of breath",
        "not short of breath",
        "there's no shortness of breath",
        "no trouble breathing",
        "it's not shortness of breath, just mild",
    ],
)
def test_explicitly_denied_breathing_difficulty_does_not_false_fire(message):
    # Reported live: a patient's own reassurance ("no shortness of breath") still
    # tripped the emergency short-circuit before the LLM ever saw the message, since
    # the phrase matched regardless of the negation word right in front of it — same
    # failure mode as the bleeding-severity patterns below, fixed the same way.
    assert not detect_red_flag(message)


# --- stroke signs ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "my face is drooping on one side",
        "his speech is slurred",
        "her speech is slurred all of a sudden",
        "my left side feels weak",
        "left side is numb",
        "sudden numbness in my face",
        "I think this is a stroke",
        "he suddenly can't speak",
    ],
)
def test_stroke_sign_patterns_fire(message):
    assert detect_red_flag(message)


# --- severe bleeding --------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "I am bleeding heavily and it won't stop",
        "severe bleeding from a cut",
        "there is blood everywhere and it won't stop",
    ],
)
def test_severe_bleeding_patterns_fire(message):
    assert detect_red_flag(message)


# --- unconsciousness / seizure ------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "he passed out and is unresponsive",
        "she had a seizure",
        "my dad is unconscious",
        "he fainted at the store",
        "she is convulsing",
    ],
)
def test_unconsciousness_and_seizure_patterns_fire(message):
    assert detect_red_flag(message)


# --- anaphylaxis / severe allergic reaction -----------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "my throat is closing up",
        "severe allergic reaction, face is swelling",
        "I think this is anaphylaxis",
    ],
)
def test_anaphylaxis_patterns_fire(message):
    assert detect_red_flag(message)


# --- suicidal / self-harm crisis ----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "I want to kill myself",
        "I don't want to live anymore, I want to die",
        "I keep thinking about suicide",
        "I've been hurting myself",
    ],
)
def test_suicidal_ideation_patterns_fire(message):
    assert detect_red_flag(message)


# --- severe trauma / limb loss -------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "my leg is detached",
        "his arm is severed",
        "I am missing a leg",
        "she is missing an arm",
        "my finger got cut off",
        "his hand was torn off in the accident",
        "the doctor had to amputate his foot",
        "he needs an amputation",
        "compound fracture in the leg",
        "the bone is sticking out of my arm",
        "his leg got crushed in the accident",
        "she was impaled by a metal rod",
    ],
)
def test_severe_trauma_and_limb_loss_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "my head got detached from my body",
        "my head is detached",
        "i got decapitated",
        "my head was cut off",
        "his head got chopped off",
        "my body was cut in half",
        "i was cut in half",
        "he was split in half",
        "my neck is severed",
        "my neck snapped",
        "my spine got severed",
        "my spine is broken in half",
        "my head is separated from my neck",
    ],
)
def test_decapitation_and_body_bisection_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "i cut the apple into two pieces",
        "my shirt was torn in half",
    ],
)
def test_bare_cut_in_half_without_a_body_reference_does_not_regex_fire(message):
    # Scoping check for the patterns above — a bare "cut/torn in half" or "cut
    # into two pieces" with no body/torso/waist word and no personal pronoun
    # right before the verb must not match on food/object phrasing. Checked
    # against the regex layer directly (not detect_red_flag) since the separate
    # semantic layer has its own known, pre-existing, unrelated false-positive
    # drift on some everyday cut/slice phrasing — see this file's other
    # documented pre-existing semantic-layer failures.
    from app.services.red_flag import _RED_FLAG_RE

    assert not _RED_FLAG_RE.search(message)


@pytest.mark.parametrize(
    "message",
    [
        "my leg is broken",
        "his arm is broken",
        "I think my wrist is broken",
        "I broke my leg",
        "she broke her arm falling down the stairs",
        "my ankle is fractured",
        "he fractured his hip",
        # Reported live: these connector variants ("got broken", "just broke") still
        # fell through the original patterns, which only matched the exact phrase
        # "is broken"/"is fractured" in reverse order.
        "my arm got broken",
        "my leg got broken",
        "my ankle just broke",
        "my wrist feels broken",
    ],
)
def test_plain_broken_bone_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "my leg is not broken",
        "good news my arm isn't broken thankfully",
        "my ankle is not fractured",
        # Reported live: "i doesnt broke my leg" — an apostrophe-dropped contraction
        # ("doesnt", not "doesn't") — still fired, since "doesnt" contains neither
        # "not" nor "n't" as a literal substring. This is an extremely common way to
        # type these contractions, not an edge case.
        "i doesnt broke my leg",
        "i dont think my arm is broken",
        "i havent broken anything",
    ],
)
def test_explicitly_denied_broken_bone_does_not_false_fire(message):
    assert not detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "its not severe, just annoying",
        "doesnt seem too severe",
        "i dont have severe pain",
        "isnt that severe honestly",
    ],
)
def test_apostrophe_dropped_negation_does_not_false_fire_on_severity(message):
    assert not detect_red_flag(message)


# --- semantic-layer veto: a confirmed word-level denial isn't overridden ------------


@pytest.mark.parametrize(
    "message",
    [
        # Reported live: the word-level check correctly recognizes these as denials
        # (negation guard blocks the underlying pattern), but the semantic layer used
        # to still fire anyway — it only measures similarity-to-exemplar and has no
        # concept of negation, so "a car ... ran over me" reads as close to "hit by a
        # car" regardless of the "doesnt" in front of it. See _is_confirmed_denial's
        # own docstring in red_flag.py for why this veto is scoped narrowly (only a
        # message the word-level layer specifically identified as a denial of one of
        # its OWN categories) rather than firing on any negation word anywhere.
        "a car doesnt ran over me",
        "a car did not run over me",
        "i wasnt hit by a car",
    ],
)
def test_confirmed_denial_vetoes_the_semantic_layer(message):
    assert not detect_red_flag(message)


def test_confirmed_denial_veto_does_not_suppress_an_unrelated_real_emergency():
    # The veto must never suppress a genuine, different emergency just because the
    # same message also happens to contain an unrelated negation word — "wasnt hit
    # by a car" is a denial, but "he is unconscious" right after it is a real,
    # separate red flag that must still fire.
    assert detect_red_flag("i wasnt hit by a car but he is unconscious")


# --- ambiguous-category severity is the LLM's own job, not a blanket server rule ----


@pytest.mark.parametrize(
    "message",
    [
        "its severe",
        "its very severe and since 10 days",
        "i am having severe headache",
        "i have severe chest pain",
        "severe abdominal pain since yesterday",
        "my back pain is severe",
        "its mild",
        "moderate, comes and goes",
        "not severe, just annoying",
        "it's not that severe",
        "my headache isn't really severe",
    ],
)
def test_ambiguous_category_severity_never_auto_fires_here(message):
    # A blanket "any bare 'severe' mention is an emergency" rule briefly lived
    # here (see detect_red_flag's own docstring for the full story) after
    # reports that a "severe" PATH 2 answer wasn't reliably escalating to PATH 1.
    # The actual bug turned out to be in symptom_agent's own reply-recovery
    # logic silently discarding a VALID PATH 1 reply because its mandated
    # numbered first-aid-steps format looked like an "advice dump" — the model
    # had been deciding correctly the whole time. With that fixed at its real
    # source, ambiguous/PATH-2 category severity (headache, chest pain,
    # abdominal pain, back pain, etc.) — "severe" or otherwise — is back to
    # being screened and decided by the LLM itself via PATH 2/PATH 1 in
    # app.services.llm, not this server-side gate. None of these fire here,
    # regardless of stated severity.
    assert not detect_red_flag(message)


# --- bleeding: reversed word order and common misspellings --------------------------


@pytest.mark.parametrize(
    "message",
    [
        "im bleeding severly",
        "im bleeding severely",
        "bleeding badly from the cut",
        "bleeding a lot from my arm",
        "bleeding profusely",
    ],
)
def test_bleeding_severity_word_after_bleeding_still_fires(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "i got my arm cut and bleeding severaly",
        "bleeding severaly",
        "im bleeding severly",
    ],
)
def test_bleeding_severity_fuzzy_typo_matching_fires(message):
    # "severe(ly)" is fuzzy-matched (\bsever\w{0,4}ly\b) rather than an enumerated
    # typo list — reproduces the exact reported case ("severaly", a real typo a
    # patient actually sent) rather than only the earlier-known "severly".
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "my nose is bleeding a little",
        "i have a small cut, barely bleeding",
        "bleeding gums after brushing",
    ],
)
def test_mild_bleeding_phrasing_does_not_false_fire(message):
    assert not detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "the cut is not deep and its not bleeding severaly",
        "it's not bleeding severely, just a scratch",
        "not bleeding badly at all",
        "no heavy bleeding, just a little",
    ],
)
def test_explicitly_denied_severe_bleeding_does_not_false_fire(message):
    # A patient's own reassurance that it is NOT severe/heavy/bad was previously
    # matched anyway, since the regex only checked for the severity phrase itself
    # and ignored a negation word sitting right in front of it.
    assert not detect_red_flag(message)


# --- eye injury / foreign object in the eye ------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i got a nail stuck in my eyes",
        "there is glass stuck in my eye",
        "something sharp is in my eye",
        "a piece of metal is lodged in my eye",
        "foreign object in my eye",
    ],
)
def test_eye_foreign_object_patterns_fire(message):
    assert detect_red_flag(message)


def test_combined_severe_symptoms_in_one_message_fires():
    # Reproduces the reported case directly: a first message describing multiple
    # severe symptoms together (broken leg, severe bleeding with a typo, an object
    # stuck in the eye) must be caught by the same-message check, not fall through
    # to the generic triage agent.
    assert detect_red_flag(
        "i broke my leg and im bleeding severly and i got a nail stuck in my eyes "
        "which is hurting very badly"
    )


# --- foreign object in the nose/throat: emergency only WITH a distress signal -------
#
# Product decision: something merely resting in the nose or throat (a seed, bead,
# piece of food) is common and, on its own, not an emergency — it should fall
# through to normal department routing (ENT). It only becomes an emergency
# alongside an actual distress signal: can't breathe, can't swallow, choking, or
# turning blue. The eye is deliberately NOT given this same carve-out — see the
# eye-specific section further down, where any foreign object in the eye stays an
# unconditional red flag regardless of other symptoms.


def test_nasal_foreign_object_with_breathing_difficulty_fires():
    # Reported live: this exact message fell through both layers entirely. Two
    # separate gaps combined to miss it — "nose" wasn't in the foreign-object
    # body-part list at all (only eyes/head/neck/chest/etc.), and "difficulty to
    # breath" (a common non-native-English construction) didn't match the breathing
    # pattern, which required "difficulty" to sit directly against "breath" with only
    # whitespace between them.
    assert detect_red_flag(
        "seed of date got stuck in my nose and i am having difficulty to breath now"
    )


@pytest.mark.parametrize(
    "message",
    [
        "difficulty to breath now",
        "having trouble to breathe",
        "he is struggling to breath",
    ],
)
def test_breathing_difficulty_with_connector_word_fires(message):
    # The "X to breath(e)" construction specifically — previously missed because the
    # pattern required the trigger word directly against "breath".
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "something stuck in my throat and i can't swallow",
        "food stuck in my throat, i am choking",
        "a bead is stuck in my nose and i am having trouble breathing",
        "something is lodged in my throat and my lips are turning blue",
    ],
)
def test_nose_or_throat_foreign_object_WITH_distress_signal_fires(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "something got stuck in my nose",
        "a seed is stuck in my nose",
        "a bead got stuck in my nose",
        "she pushed a button into her nose",
        "my son inserted a small toy into his nostril",
        "there's something lodged in my nose",
        "something is stuck in my throat",
        "a small piece of food got stuck in my throat",
        "there's an object lodged in my nose",
    ],
)
def test_nose_or_throat_foreign_object_ALONE_does_not_auto_fire(message):
    # No distress signal stated — meant to route to ENT via ordinary department
    # matching (see app.services.orchestrator.symptom_hints, which already hints ENT
    # for "nose"/"throat"), not auto-fire the emergency redirect here.
    assert not detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "my nose is blocked",
        "I have a stuffy nose",
        "he has a cold and a blocked nose",
        "her nose is runny and congested",
    ],
)
def test_ordinary_blocked_or_stuffy_nose_does_not_false_fire(message):
    # A blocked/stuffy nose from an ordinary cold is one of the most common ENT
    # complaints there is, nowhere close to an emergency, and shares no vocabulary
    # with the foreign-object-insertion wording above.
    assert not detect_red_flag(message)


# --- foreign object in the eye stays an unconditional red flag ----------------------
#
# Unlike the nose/throat carve-out above, the eye keeps the original behavior: ANY
# foreign object in the eye is a red flag on its own, with or without additional
# symptoms — eye trauma is time-critical (risk of permanent vision loss) regardless
# of how the patient describes the rest of it.


@pytest.mark.parametrize(
    "message",
    [
        # Object + explicitly stated additional symptoms — still fires (the extra
        # symptoms don't change anything; the object alone is already sufficient).
        "a piece of glass got in my eye and now it's red and watering a lot",
        "something went into my eye and my vision is blurry and it hurts badly",
        "there is metal stuck in my eye and I can't open it now",
    ],
)
def test_eye_foreign_object_with_additional_symptoms_fires(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        # Ordinary eye symptoms, no foreign object mentioned at all — routes to
        # Ophthalmology via normal symptom/department matching, not an emergency.
        "my eyes have been itchy and red for a few days",
        "I have blurry vision on and off",
        "my eye feels dry and irritated",
        "I've had watery eyes since yesterday, nothing hit it",
    ],
)
def test_ordinary_eye_symptoms_without_foreign_object_do_not_fire(message):
    assert not detect_red_flag(message)


# --- vehicle accidents / high-energy trauma mechanism --------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i got hit by a car and im bleeding very much",
        "i was hit by a truck",
        "a motorcycle hit me",
        "i was in a car accident",
        "there was a road accident",
        "i had a bike crash",
        "we collided with another vehicle",
        "it was a hit and run",
        "i got run over by a car",
    ],
)
def test_vehicle_accident_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "bleeding very much",
        "bleeding so much from the cut",
        "bleeding a ton",
    ],
)
def test_everyday_bleeding_severity_phrasing_fires(message):
    assert detect_red_flag(message)


def test_reported_vehicle_accident_case_fires():
    # Exact reported case: a first message describing being hit by a car and
    # bleeding heavily (in everyday, non-clinical phrasing) must be caught by the
    # same-message check rather than routed to normal department triage.
    assert detect_red_flag("i got hit by a car and im bleeding very much")


# --- choking / airway obstruction ----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i am choking",
        "my son choked on a grape",
        "something is stuck in my throat and i cant swallow",
        "my lips are turning blue",
        "his skin is turning blue",
    ],
)
def test_choking_and_airway_patterns_fire(message):
    assert detect_red_flag(message)


# --- poisoning / overdose --------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i swallowed bleach",
        "i think i overdosed on pills",
        "my kid drank some chemical",
        "i took too many pills",
        "he was poisoned",
    ],
)
def test_poisoning_and_overdose_patterns_fire(message):
    assert detect_red_flag(message)


# --- severe burns / fire ---------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "my child got badly burned",
        "i have a severe burn on my arm",
        "i caught fire while cooking",
        "there was a third-degree burn",
    ],
)
def test_severe_burn_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "i have a minor sunburn",
        "i got a small burn on my finger",
    ],
)
def test_mild_burn_phrasing_does_not_false_fire(message):
    assert not detect_red_flag(message)


# --- electrocution / drowning -----------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i got electrocuted by a wire",
        "i had an electric shock",
        "my friend almost drowned in the pool",
        "he was drowning",
    ],
)
def test_electrocution_and_drowning_patterns_fire(message):
    assert detect_red_flag(message)


# --- weapon injuries / falls from height -------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "i got shot in the leg",
        "there was a gunshot wound",
        "i was stabbed",
        "there is a stab wound",
        "i fell from the roof",
        "he fell from a ladder",
    ],
)
def test_weapon_injury_and_fall_from_height_patterns_fire(message):
    assert detect_red_flag(message)


# --- venomous bites / testicular torsion / head injury + vomiting -----------------------


@pytest.mark.parametrize(
    "message",
    [
        "i got bitten by a snake",
        "there was a snake bite",
        "i was stung by a scorpion",
        "i have sudden testicular pain",
        "sudden sharp pain in my testicle",
        "i hit my head and now i am vomiting",
    ],
)
def test_bite_torsion_and_head_injury_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    ["severe testicle pain", "sharp scrotal pain", "severe testicular pain"],
)
def test_testicular_pain_plain_adjective_noun_noun_order_fires(message):
    # Regression: the semantic layer used to backstop this category, but was
    # removed entirely after real false positives (see _EXEMPLARS' docstring in
    # red_flag.py) — regex is now the ONLY detection for testicular pain, and this
    # exact word order ("severe testicle pain", mirroring the already-covered
    # "severe chest pain" construction) was missing from all three original
    # patterns, none of which matched severity-adjective directly followed by the
    # testicle word directly followed by "pain".
    assert detect_red_flag(message)


def test_snake_bite_reverse_word_order_fires():
    # "a snake bit me" (subject-verb-object) rather than the passive "bitten by a
    # snake" phrasing the other snake-bite patterns cover.
    assert detect_red_flag("a snake bit me on the leg")


@pytest.mark.parametrize(
    "message",
    [
        "a cobra bit my brother in the garden",
        "bitten by a rattlesnake while hiking",
        "i think a viper bit my hand",
    ],
)
def test_named_venomous_snake_species_bite_fires(message):
    # Named-species coverage moved from the semantic layer to regex directly (see
    # test_everyday_bites_and_minor_cuts_do_not_false_fire's comment) — a generic
    # "venomous snake" exemplar isn't safe to keep, so specific species names are
    # matched explicitly instead.
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "i got bitten by a dog",
        "my dog bit me on the arm",
        "i got a cut on my hand",
        "i cut a cut on my finger when i was cutting apple",
        "i have a small paper cut",
        "my cat scratched me",
    ],
)
def test_everyday_bites_and_minor_cuts_do_not_false_fire(message):
    # Regression for a reported bug: the semantic layer's "bitten by a venomous
    # snake or scorpion" and "a finger or limb completely cut off" exemplars scored
    # above threshold against an ordinary dog bite / kitchen cut — species and
    # severity aren't reliably separated from topic by these embeddings, so both
    # exemplars were removed (their genuine true-positive cases stay covered by
    # regex instead — see test_snake_bite_reverse_word_order_fires and the
    # severed/cut-off body-part regex).
    assert not detect_red_flag(message)


# --- foreign object / penetrating injury to any body part (not just the eye) --------


@pytest.mark.parametrize(
    "message",
    [
        "i got nail stuck in my head an its very painful",
        "a nail is stuck in my head",
        "my head has something stuck in it",
        "there is a knife stuck in his chest",
        "a screw went into my back",
        "something is embedded in his neck",
    ],
)
def test_foreign_object_in_any_body_part_fires(message):
    assert detect_red_flag(message)


# --- chemical exposure to eyes/skin --------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "acid got in my eyes",
        "bleach splashed on my face",
        "chemical sprayed in my eyes",
    ],
)
def test_chemical_exposure_patterns_fire(message):
    assert detect_red_flag(message)


# --- spinal injury signs / heat stroke / hypothermia / severe asthma ------------------


@pytest.mark.parametrize(
    "message",
    [
        "i cant feel my legs after the accident",
        "i cant move my arms after the fall",
        "he is paralyzed after the crash",
        "i think i have heat stroke",
        "he collapsed from the heat",
        "severe hypothermia",
        "shivering uncontrollably in the cold",
        "asthma attack and my inhaler is not working",
    ],
)
def test_spinal_heat_and_asthma_patterns_fire(message):
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        "my leg fell asleep and i cant feel it right now",
        "my inhaler is almost empty, can i get a refill",
        "it is really hot today",
        "i feel a bit dehydrated, should i drink more water",
    ],
)
def test_spinal_heat_and_asthma_negative_phrasing_does_not_false_fire(message):
    assert not detect_red_flag(message)


# --- head injury + confusion/memory loss (broadened beyond just vomiting) -----------


@pytest.mark.parametrize(
    "message",
    [
        "i hit my head and now i cant remember anything",
        "head injury and hes confused",
    ],
)
def test_head_injury_with_confusion_or_memory_loss_fires(message):
    assert detect_red_flag(message)


# --- negative cases: must NOT false-fire on unrelated phrasing ----------------------


@pytest.mark.parametrize(
    "message",
    [
        "I have a mild headache",
        "I want to book an appointment with cardiology",
        "what are your clinic hours",
        "I feel a bit tired today",
        "my stomach hurts a little",
        "I have a slight cough",
        "sounds like a great plan",
        "I'm dying to see that new movie",
        "can I reschedule my appointment",
        "who's available in dermatology",
        "I have a bit of a rash on my arm",
        "my knee has been aching for two days",
    ],
)
def test_unrelated_phrasing_does_not_false_fire(message):
    assert not detect_red_flag(message)


# --- semantic similarity layer: paraphrases with NO shared vocabulary with any regex
# pattern must still fire, since detect_red_flag() is a union of the regex gate and
# the embedding-similarity check (see red_flag.py's module docstring and _EXEMPLARS
# for how/why this list was calibrated). ------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # Stroke — no "stroke"/"slurred"/"drooping"/"numb" keyword at all.
        "the left side of my face suddenly stopped working and my words are slurring",
        # Poisoning — no "poison"/"overdose"/"swallowed" keyword.
        "my kid ate a bunch of pills from the cabinet",
        # Electrocution — no "electrocut"/"shock" keyword.
        "i touched a live wire and got zapped hard",
        # Gunshot — no "shot"/"gun" keyword at all.
        "someone put a bullet in his stomach",
        # Fall from height — no "fell"/"roof"/"ladder" keyword.
        "she went over the balcony railing from the third floor",
        # Unconsciousness — no "unconscious"/"unresponsive"/"passed out" keyword.
        "he just collapsed and wont wake up no matter what we do",
        # Suicidal ideation — no "suicid"/"kill myself"/"want to die"/"hurt myself"
        # keyword at all.
        "I feel like ending it all",
        "I have been having thoughts of not wanting to be alive",
    ],
)
def test_semantic_similarity_catches_paraphrases_with_no_shared_regex_vocabulary(message):
    # Each of these is confirmed to NOT match the regex gate on its own (see the
    # calibration script in the PR/commit that added this) — only the embedding
    # similarity layer can catch them.
    from app.services.red_flag import _RED_FLAG_RE

    assert not _RED_FLAG_RE.search(message), f"test setup error: regex already matches {message!r}"
    assert detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        # These deliberately stay negative even with the semantic layer active — see
        # _EXEMPLARS' docstring for why chest pain/bleeding/burns/choking/fractures/
        # testicular pain are excluded from the semantic bank specifically.
        "I have severe chest pain",
        "chest pressure and tightness",
        "there is a squeezing pain in my chest",
        "i have a small cut, barely bleeding",
        "i have a minor sunburn",
        "i got a small burn on my finger",
        "i have a bit of a rash on my arm",
        "my knee has been aching for two days",
        # Regression: testicular torsion was REMOVED from the semantic exemplar bank
        # entirely after this scored 0.61 with no severity/suddenness/swelling
        # stated at all — even the margin layer couldn't separate it from a genuine
        # emergency paraphrase, same "severity can't be inferred from topic alone"
        # failure as chest pain/bleeding/burns above. See that exemplar's own
        # removal comment in red_flag.py for the full calibration details.
        "i am having pain in my testies",
        "i have pain in my testicles",
        "my testicle hurts a bit",
        # Regression: a benign personal-recall question about a previously-mentioned
        # allergy scored 0.6 against an early "allergic reaction" exemplar purely off
        # the shared word "allergy" — anaphylaxis was dropped from _EXEMPLARS as a
        # result (see its docstring). Kept here so it can't silently regress if
        # anaphylaxis-style wording is ever added back to the exemplar bank.
        "What allergy did I mention earlier?",
        # Regression: ordinary multi-limb pain (nothing one-sided, nothing about
        # weakness/numbness/facial drooping) scored just over threshold against the
        # original, looser stroke exemplar wording — see that exemplar's own comment
        # in red_flag.py for the calibration details.
        "i am having pain in my leg and in my hands as well",
        "i am having pain in my leg and also my hands",
        "my hands and legs hurt",
        # Regression: a routine PATH-2 screening follow-up with no body part named at
        # all scored just over threshold against the original, looser testicular-pain
        # exemplar wording — see that exemplar's own comment in red_flag.py for the
        # calibration details.
        "its been from past 10 days\nand the pain is mild and moderate",
        "i am having pain in my hand and also have some skin related issues",
        # Regression: simply naming/booking the Psychiatry department (no self-harm
        # content at all) scored just over threshold against the original, looser
        # suicidal-ideation exemplar wording — see that exemplar's own comment in
        # red_flag.py for the calibration details.
        "Psychiatry in this dept",
        "I want to book an appointment in Psychiatry",
        "book me with psychiatry department",
        "I need a psychiatrist appointment",
        "can I see a psychiatrist",
        "show me psychiatry availability",
    ],
)
def test_semantic_similarity_does_not_false_fire_on_path2_or_benign_messages(message):
    assert not detect_red_flag(message)


@pytest.mark.parametrize(
    "message",
    [
        # Regression for the MARGIN LAYER itself (see _SEMANTIC_MARGIN's docstring
        # in red_flag.py): each of these is a hedged/mild variant of a real danger
        # sign — the same "short + vague drifts toward every exemplar bank" shape
        # that caused three separate real false positives before the margin layer
        # was added. These previously scored just under the OLD absolute-only
        # threshold's safety margin and are kept here so a future exemplar-only
        # tweak (without the margin layer's help) can't silently regress them.
        "i have slight numbness in my hand and it feels a bit weak",
        "ive had mild numbness in one hand for a few days, its bearable",
        "my hand feels weak and numb but the pain is mild",
        "i feel a little weak and numb on my left side but its manageable",
        "i occasionally slur a word or two when im tired, its not a big deal",
    ],
)
def test_semantic_margin_suppresses_hedged_variants_of_danger_sign_wording(message):
    assert not detect_red_flag(message)


def test_empty_and_blank_messages_do_not_fire():
    assert not detect_red_flag("")
    assert not detect_red_flag("   ")


# --- red_flag_message ----------------------------------------------------------------


def test_red_flag_message_is_always_english():
    # Urdu removed per product decision — a red-flagged reply must never be
    # translated at request time (risk of a life-safety message being softened), and
    # a second hardcoded-Urdu variant wasn't being kept in sync with the English one.
    assert red_flag_message() == RED_FLAG_MESSAGE_EN


def test_red_flag_message_includes_first_aid_guidance_immediately():
    # First aid must be part of the fixed deterministic message itself, not something
    # the patient has to ask a follow-up question to get (the LLM never even runs for
    # a red-flagged turn, so there is no other point in the flow where it can appear).
    assert "While you're on your way" in RED_FLAG_MESSAGE_EN
    assert "bleeding" in RED_FLAG_MESSAGE_EN


def test_red_flag_message_names_the_clinic_local_emergency_number():
    # Reported live: the message previously said "your local emergency number" with
    # no actual number, and separately the LLM's own PATH 1 replies were inventing an
    # arbitrary one (e.g. "999") since nothing told it which to use. Must name 1122
    # explicitly now.
    assert "1122" in RED_FLAG_MESSAGE_EN
