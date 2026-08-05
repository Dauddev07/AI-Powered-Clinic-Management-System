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
        ("my knee and ankle hurt", "Orthopedics"),
        ("I sprained my joint", "Orthopedics"),
        ("my shoulder and back hurt", "Orthopedics"),
        ("my wrist and elbow are sore", "Orthopedics"),
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
        ("I feel mild dizziness", "Neurology"),
        ("I have vertigo", "ENT"),
        ("I feel lightheaded", "Neurology"),
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
