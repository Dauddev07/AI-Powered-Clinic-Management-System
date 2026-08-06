import pytest

from app.services.orchestrator.symptom_hints import departments_hinted_by_patient_symptom_words

DEPARTMENTS = [
    "Cardiology", "Dentistry", "Dermatology", "ENT", "General Medicine", "Gynecology",
    "Neurology", "Ophthalmology", "Orthopedics", "Pediatrics", "Psychiatry", "Pulmonology",
]


@pytest.mark.parametrize(
    "message, expected_department",
    [
        ("I have an itchy rash on my arm", "Dermatology"),
        ("there's a mole on my back that looks weird", "Dermatology"),
        ("I have bad acne on my face", "Dermatology"),
        ("I have a boil that won't go away", "Dermatology"),
        ("I have a wart on my hand", "Dermatology"),
        ("my ear hurts and I have trouble hearing", "ENT"),
        ("I have a sore throat", "ENT"),
        ("my nose is blocked and I have sinus pain", "ENT"),
        ("my voice is hoarse", "ENT"),
        ("my eyes hurt and my vision is blurry", "Ophthalmology"),
        ("I have a toothache", "Dentistry"),
        ("pain in my teeths", "Dentistry"),  # typo regression
        ("my knee and ankle hurt", "General Medicine"),  # bare limb pain, no injury signal
        ("I sprained my joint", "Orthopedics"),
        ("my shoulder and back hurt", "General Medicine"),
        ("my wrist and elbow are sore", "General Medicine"),
        ("my leg is swollen after I fell", "Orthopedics"),  # limb pain + injury signal
        ("my hand hurts a lot", "General Medicine"),  # previously-missing "hand" keyword
        ("I have a headache", "General Medicine"),
        ("I get migraines often", "General Medicine"),
        ("I have chest pain and palpitations", "Cardiology"),
        ("I have hypertension", "Cardiology"),
        ("my stomach hurts and I have diarrhea", "General Medicine"),
        ("I have kidney pain and blood in my urine", "General Medicine"),
        ("I feel sad and hopeless lately", "Psychiatry"),
        ("I've been having panic attacks", "Psychiatry"),
        ("I have insomnia", "Psychiatry"),
        ("I have a persistent cough and I'm wheezing", "Pulmonology"),
        ("my asthma is acting up", "Pulmonology"),
        ("I have high blood sugar and I'm always thirsty", "General Medicine"),
        ("I think I have a brain tumor", "Neurology"),
        ("I've been having seizures", "Neurology"),
        ("I have numbness and weakness in my arm", "Neurology"),
        ("my hand has a tremor", "Neurology"),
        ("I am pregnant and having pelvic pain", "Gynecology"),
        ("I'm having irregular periods", "Gynecology"),
        ("I have a fever and body aches", "General Medicine"),
        ("I feel mild dizziness", "ENT"),
        ("I have vertigo", "ENT"),
        ("I feel lightheaded", "ENT"),
        ("I feel dizzy and numb", "Neurology"),  # dizziness + a real neuro red flag
        ("pain in my chest and in testies as well", "Cardiology"),
        ("pain in my chest and in testies as well", "General Medicine"),
        ("I have testicular pain", "General Medicine"),
        ("there's pain in my groin", "General Medicine"),
    ],
)
def test_symptom_words_hint_the_expected_department(message, expected_department):
    hinted = departments_hinted_by_patient_symptom_words(message, [], DEPARTMENTS, set())
    assert expected_department in hinted


def test_pediatrics_is_never_hinted_by_symptom_words():
    # Pediatrics is routed by patient AGE, not symptom vocabulary — no symptom
    # keyword should ever imply it, by design (see this module's own comment).
    hinted = departments_hinted_by_patient_symptom_words(
        "my child has a fever and a cough and a rash", [], DEPARTMENTS, set()
    )
    assert "Pediatrics" not in hinted


def test_already_covered_departments_are_excluded():
    hinted = departments_hinted_by_patient_symptom_words(
        "I have a headache", [], DEPARTMENTS, {"General Medicine"}
    )
    assert hinted == {}


def test_unrelated_message_hints_nothing():
    assert departments_hinted_by_patient_symptom_words(
        "what are your clinic hours?", [], DEPARTMENTS, set()
    ) == {}


def test_plain_cough_alone_does_not_hint_pulmonology():
    # Reported live: "mild cough with fever" got an unwanted extra Pulmonology
    # card next to the General Medicine one. A plain cough is ordinary General
    # Medicine territory unless paired with an actual breathing-distress symptom.
    hinted = departments_hinted_by_patient_symptom_words(
        "I have a mild cough and fever", [], DEPARTMENTS, set()
    )
    assert "Pulmonology" not in hinted


def test_cough_with_difficulty_breathing_hints_pulmonology():
    hinted = departments_hinted_by_patient_symptom_words(
        "I have a cough and difficulty breathing", [], DEPARTMENTS, set()
    )
    assert "Pulmonology" in hinted


def test_ankle_swelling_does_not_hint_pulmonology():
    # "swelling" alone is too generic (ankle/jaw/face) to safely imply Pulmonology
    # given per-word, non-co-occurrence-aware matching.
    hinted = departments_hinted_by_patient_symptom_words(
        "my ankle is swollen and sprained", [], DEPARTMENTS, set()
    )
    assert "Pulmonology" not in hinted


def test_limb_pain_with_injury_signal_hints_orthopedics_only_not_general_medicine():
    # Reported live: "leg pain after i fell should go to orthopedics" — a real
    # limb injury should resolve to Orthopedics ALONE, not Orthopedics alongside a
    # default General Medicine card. General Medicine only appears when something
    # else independently calls for it (e.g. a separately mentioned fever).
    hinted = departments_hinted_by_patient_symptom_words(
        "my leg is swollen after I fell", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms"}


def test_limb_pain_with_injury_signal_and_unrelated_symptom_still_includes_general_medicine():
    # General Medicine should still show up here, but because of the fever, not
    # because of the limb pain — confirms the limb/joint branch doesn't suppress
    # OTHER categories' independent General Medicine hint.
    hinted = departments_hinted_by_patient_symptom_words(
        "my leg is swollen after I fell and i also have a fever", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms", "General Medicine": "fever/body aches"}


def test_plain_swelling_alone_does_not_hint_orthopedics():
    # Reported live: "swelling on hand" alone still routed to Orthopedics, but
    # plain swelling isn't specific to injury (infection, allergic reaction, edema
    # can all cause it) — it should fall back to General Medicine like any other
    # bare limb symptom unless an actual injury/trauma word is also present.
    hinted = departments_hinted_by_patient_symptom_words(
        "swelling on hand and leg", [], DEPARTMENTS, set()
    )
    assert hinted == {"General Medicine": "limb/joint pain"}
    assert "Orthopedics" not in hinted


def test_swelling_with_a_real_injury_word_still_hints_orthopedics():
    hinted = departments_hinted_by_patient_symptom_words(
        "my hand is swollen and bruised", [], DEPARTMENTS, set()
    )
    assert "Orthopedics" in hinted


def test_limb_pain_does_not_add_a_redundant_general_medicine_card_when_orthopedics_already_covers_it():
    # Reported live: a patient described hand swelling + chest pain. The LLM's own
    # tool call already (correctly) produced an Orthopedics card for the hand, but
    # the bare-word fallback independently recomputed "hand" as General Medicine
    # and added a redundant second card for the same body part — General Medicine
    # and Orthopedics are two possible outcomes of the SAME limb/joint family, so
    # Orthopedics already being covered (by anything — the LLM, an earlier turn)
    # must suppress the generic fallback for that same family.
    hinted = departments_hinted_by_patient_symptom_words(
        "its only swelling on hand no other symptoms and chest pain as well",
        [], DEPARTMENTS, {"Orthopedics"},
    )
    assert "General Medicine" not in hinted
    assert hinted == {"Cardiology": "chest pain"}


def test_general_medicine_already_covered_does_not_suppress_a_genuine_orthopedics_addition():
    # Deliberately one-directional: General Medicine already being covered (e.g.
    # for an unrelated fever) must NOT suppress a real Orthopedics addition — that's
    # still a genuinely more specific need, unlike the reverse direction above.
    hinted = departments_hinted_by_patient_symptom_words(
        "my leg is swollen after I fell and i have a fever", [], DEPARTMENTS, {"General Medicine"},
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms"}


def test_dizziness_alone_hints_ent_only_not_neurology():
    # Reported live: "i am feeling very sad today and dizzy as well... mild... also
    # have nausea" produced FOUR cards (General Medicine, ENT, Neurology,
    # Psychiatry) for what's really one coherent complaint — dizziness
    # unconditionally hinting Neurology too, with zero actual neuro signs present,
    # was one of the two causes. ENT (vestibular/inner-ear) is the far more common
    # cause and is hinted alone; Neurology only comes in via its own dedicated
    # trigger when a real neuro red flag is present (see the next test).
    hinted = departments_hinted_by_patient_symptom_words(
        "i feel dizzy and mild and have nausea", [], DEPARTMENTS, set()
    )
    assert hinted == {"ENT": "dizziness"}


def test_dizziness_with_a_real_neuro_red_flag_still_hints_neurology():
    hinted = departments_hinted_by_patient_symptom_words(
        "i feel dizzy and numb", [], DEPARTMENTS, set()
    )
    assert hinted == {"ENT": "dizziness", "Neurology": "neurological symptoms"}


def test_passing_mood_word_alone_does_not_hint_psychiatry():
    # Reported live: "feeling very sad today" (a single passing adjective, said
    # alongside an unrelated physical complaint) fired a full Psychiatry card with
    # zero screening — unlike physical symptoms, which always go through PATH-2
    # duration/severity screening before a card appears. "today" signals recency,
    # not persistence, so this should NOT fire.
    hinted = departments_hinted_by_patient_symptom_words(
        "i am feeling very sad today", [], DEPARTMENTS, set()
    )
    assert "Psychiatry" not in hinted


def test_passing_mood_word_with_persistence_still_hints_psychiatry():
    hinted = departments_hinted_by_patient_symptom_words(
        "i have been feeling sad for weeks", [], DEPARTMENTS, set()
    )
    assert "Psychiatry" in hinted


def test_named_clinical_mood_terms_still_hint_psychiatry_unconditionally():
    # Anxiety/depression/panic/insomnia already NAME a real presenting complaint,
    # unlike a passing adjective like "sad" — these keep firing without needing a
    # persistence word alongside them.
    for message in ("I have anxiety", "I have depression", "I have panic attacks", "I have insomnia"):
        hinted = departments_hinted_by_patient_symptom_words(message, [], DEPARTMENTS, set())
        assert "Psychiatry" in hinted, message
