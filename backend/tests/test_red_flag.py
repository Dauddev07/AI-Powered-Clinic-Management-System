import pytest

from app.services.red_flag import (
    RED_FLAG_MESSAGE_EN,
    RED_FLAG_MESSAGE_UR,
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
    # "heart attack") deliberately does NOT auto-fire anymore — it ranges from a
    # pulled muscle to a real cardiac emergency, so the chat agent itself asks a
    # same-turn severity-screening question first (see PATH 2 in
    # app.services.llm._AGENT_SYSTEM_PROMPT) instead of an immediate blanket
    # redirect with no chance to ask anything.
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
        # Testicular torsion — no "testicular"/"testicle"/"scrotal" keyword.
        "excruciating pain in one of my balls that started suddenly",
        # Unconsciousness — no "unconscious"/"unresponsive"/"passed out" keyword.
        "he just collapsed and wont wake up no matter what we do",
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
        # _EXEMPLARS' docstring for why chest pain/bleeding/burns/choking/fractures
        # are excluded from the semantic bank specifically.
        "I have severe chest pain",
        "chest pressure and tightness",
        "there is a squeezing pain in my chest",
        "i have a small cut, barely bleeding",
        "i have a minor sunburn",
        "i got a small burn on my finger",
        "i have a bit of a rash on my arm",
        "my knee has been aching for two days",
        # Regression: a benign personal-recall question about a previously-mentioned
        # allergy scored 0.6 against an early "allergic reaction" exemplar purely off
        # the shared word "allergy" — anaphylaxis was dropped from _EXEMPLARS as a
        # result (see its docstring). Kept here so it can't silently regress if
        # anaphylaxis-style wording is ever added back to the exemplar bank.
        "What allergy did I mention earlier?",
    ],
)
def test_semantic_similarity_does_not_false_fire_on_path2_or_benign_messages(message):
    assert not detect_red_flag(message)


def test_empty_and_blank_messages_do_not_fire():
    assert not detect_red_flag("")
    assert not detect_red_flag("   ")


# --- red_flag_message language variants ---------------------------------------------


def test_red_flag_message_returns_correct_language_variant():
    assert red_flag_message("en") == RED_FLAG_MESSAGE_EN
    assert red_flag_message("ur") == RED_FLAG_MESSAGE_UR


def test_red_flag_message_defaults_to_english_for_unknown_language():
    assert red_flag_message("fr") == RED_FLAG_MESSAGE_EN


def test_red_flag_message_includes_first_aid_guidance_immediately():
    # First aid must be part of the fixed deterministic message itself, not something
    # the patient has to ask a follow-up question to get (the LLM never even runs for
    # a red-flagged turn, so there is no other point in the flow where it can appear).
    assert "While you're on your way" in RED_FLAG_MESSAGE_EN
    assert "bleeding" in RED_FLAG_MESSAGE_EN
    assert "راستے میں" in RED_FLAG_MESSAGE_UR
