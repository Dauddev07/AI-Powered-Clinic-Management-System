import pytest

from app.services.orchestrator.symptom_hints import (
    departments_hinted_by_patient_symptom_words,
    symptom_words_with_no_matching_department,
    unsupported_symptom_labels,
)

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
        ("pain in my joints all along the body", "Orthopedics"),  # bare "joints" (plural), no injury word needed
        ("my shoulder and back hurt", "General Medicine"),
        ("my wrist and elbow are sore", "General Medicine"),
        ("my leg is swollen after I fell", "Orthopedics"),  # limb pain + injury signal
        ("back pain while bending", "Orthopedics"),  # back pain + bending/movement signal
        ("my back hurts", "General Medicine"),  # bare back pain, no movement/injury signal
        ("i am having difficulty in walking properly", "General Medicine"),  # bare gait complaint
        ("i fell and now i am having difficulty walking", "Orthopedics"),  # gait complaint + injury signal
        ("i am having a bit difficulty in breething", "Pulmonology"),  # "breething" typo regression
        ("my hand hurts a lot", "General Medicine"),  # previously-missing "hand" keyword
        ("I have a headache", "General Medicine"),
        ("I get migraines often", "General Medicine"),
        ("I have chest pain and palpitations", "Cardiology"),
        ("I have hypertension", "Cardiology"),
        ("my stomach hurts and I have diarrhea", "General Medicine"),
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
        ("I feel lightheaded", "General Medicine"),  # bare lightheadedness, no ear/vertigo signal
        ("I feel lightheaded and the room keeps spinning", "ENT"),  # lightheaded + vertigo signal
        ("I feel dizzy and numb", "Neurology"),  # dizziness + a real neuro red flag
        ("pain in my chest and in testies as well", "Cardiology"),
    ],
)
def test_symptom_words_hint_the_expected_department(message, expected_department):
    hinted = departments_hinted_by_patient_symptom_words(message, [], DEPARTMENTS, set())
    assert expected_department in hinted


@pytest.mark.parametrize(
    "message, expected_label",
    [
        ("my height is not increasing", "growth/height concerns"),
        ("I have kidney pain and blood in my urine", "urinary symptoms"),
        ("I have testicular pain", "groin/testicular symptoms"),
        ("there's pain in my groin", "groin/testicular symptoms"),
    ],
)
def test_orphan_symptom_categories_hint_no_department_at_a_clinic_with_no_matching_specialty(
    message, expected_label
):
    # Instructed live: a symptom this clinic genuinely has no matching specialist
    # for (no Urology, no Endocrinology/growth clinic) must never be silently
    # rerouted to General Medicine just because SOME department exists — General
    # Medicine isn't actually equipped for it. See unsupported_symptom_labels,
    # which is what turns this into a gentle apology instead (tested in
    # test_orchestrator.py against the full symptom_agent flow).
    assert departments_hinted_by_patient_symptom_words(message, [], DEPARTMENTS, set()) == {}
    assert unsupported_symptom_labels(message, [], DEPARTMENTS) == [expected_label]


def test_orphan_symptom_category_hints_real_department_when_clinic_has_one():
    departments_with_urology = DEPARTMENTS + ["Urology"]
    hinted = departments_hinted_by_patient_symptom_words(
        "I have testicular pain", [], departments_with_urology, set()
    )
    assert hinted == {"Urology": "groin/testicular symptoms"}
    assert unsupported_symptom_labels("I have testicular pain", [], departments_with_urology) == []


# Instructed live: the hardcoded _ORPHAN_SYMPTOM_CATEGORIES list only ever
# covers a symptom category someone happened to add — any OTHER symptom this
# clinic can't treat used to be left to the LLM to freehand a guess (the same
# class of bug as the old brain-tumor->Neurology mistake). This general
# function checks the WHOLE symptom-hint table, not just the hardcoded list,
# and names the specialty the patient should look for elsewhere.
@pytest.mark.parametrize(
    "message, expected_label, expected_department",
    [
        ("I have testicular pain", "groin/testicular symptoms", "Urology"),  # already in the orphan list too
        ("I have kidney pain and blood in my urine", "urinary symptoms", "Urology"),
        ("I think I have cancer", "cancer/tumor-related symptoms", "Oncology"),
    ],
)
def test_symptom_words_with_no_matching_department_names_the_missing_specialty(
    message, expected_label, expected_department
):
    # These three symptom categories have NO General Medicine (or any other)
    # fallback in the hint table — only a genuine Urology/Oncology match would
    # resolve them, and DEPARTMENTS above has neither, so all three are
    # genuinely unmatched with a specific specialty name to suggest.
    assert departments_hinted_by_patient_symptom_words(message, [], DEPARTMENTS, set()) == {}
    result = symptom_words_with_no_matching_department(message, [], DEPARTMENTS)
    assert (expected_label, expected_department) in result


def test_symptom_words_with_no_matching_department_does_not_flag_a_category_with_a_real_fallback():
    # "my stomach hurts" hints Gastroenterology first but falls back to General
    # Medicine — DEPARTMENTS has General Medicine, so this must NOT be reported
    # as unmatched, unlike the no-fallback categories above.
    assert symptom_words_with_no_matching_department("my stomach hurts", [], DEPARTMENTS) == []


def test_symptom_words_with_no_matching_department_returns_nothing_when_a_real_department_covers_it():
    # "chest pain" resolves via the real Cardiology department already in
    # DEPARTMENTS — must not also be reported as unmatched.
    result = symptom_words_with_no_matching_department("I have chest pain", [], DEPARTMENTS)
    assert result == []


def test_symptom_words_with_no_matching_department_returns_nothing_for_non_symptom_text():
    result = symptom_words_with_no_matching_department("hi, how are you?", [], DEPARTMENTS)
    assert result == []


def test_orphan_symptom_category_does_not_fire_when_something_real_is_also_hinted():
    # A compound complaint that also names something this clinic DOES treat still
    # gets that real department — the unsupported part just isn't separately
    # hinted (it's still reported by unsupported_symptom_labels itself; whether to
    # mention it alongside a real card is the caller's decision, see
    # symptom_agent.run_symptom_agent).
    hinted = departments_hinted_by_patient_symptom_words(
        "pain in my chest and in testies as well", [], DEPARTMENTS, set()
    )
    assert hinted == {"Cardiology": "chest pain"}
    assert unsupported_symptom_labels(
        "pain in my chest and in testies as well", [], DEPARTMENTS
    ) == ["groin/testicular symptoms"]


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


def test_bare_neck_pain_hints_orthopedics():
    # Reported live (1st report): "i am having pain in my neck" got no department
    # hint at all — "neck" wasn't in _LIMB_JOINT_WORDS, so the whole limb/joint
    # co-occurrence branch never fired for it, leaving the LLM to free-text ask
    # the patient to pick a specialist themselves instead of showing a routed
    # card. Neck was first added to the conditional bare-anatomy bucket
    # (General Medicine default, like leg/arm), but reported live again (2nd
    # report): a patient's neck pain got mislabeled "limb/joint pain" and routed
    # to General Medicine when the expected department was Orthopedics — the
    # neck is a spine/joint region, not a generic limb, so like "joint" itself
    # it hints Orthopedics unconditionally, no injury signal required.
    hinted = departments_hinted_by_patient_symptom_words(
        "i am having pain in my neck", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "neck/joint pain"}


def test_neck_pain_with_injury_signal_still_hints_orthopedics_only():
    hinted = departments_hinted_by_patient_symptom_words(
        "i twisted my neck and it hurts", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "neck/joint pain"}


def test_neck_pain_with_numbness_hints_neurology_only_not_orthopedics():
    # Reported live (3rd report): "pain in my neck and i am feeling numb" got
    # BOTH a Neurology card (correct, from numbness) and a redundant Orthopedics
    # card for the same neck pain — same case, not two. Numbness alongside neck
    # pain points to a neurological cause, so Orthopedics must not also fire.
    hinted = departments_hinted_by_patient_symptom_words(
        "i am having pain in my neck and i am feeling numb", [], DEPARTMENTS, set()
    )
    assert hinted == {"Neurology": "neurological symptoms"}


def test_neck_pain_with_weakness_hints_neurology_only_not_orthopedics():
    hinted = departments_hinted_by_patient_symptom_words(
        "my neck hurts and i have some weakness", [], DEPARTMENTS, set()
    )
    assert hinted == {"Neurology": "neurological symptoms"}


def test_neck_and_eye_pain_still_hints_both_orthopedics_and_ophthalmology():
    # No neurological signal present here — confirms the suppression above is
    # specific to numbness/weakness/tremor/paralysis, not to "neck co-occurring
    # with any other symptom" in general.
    hinted = departments_hinted_by_patient_symptom_words(
        "i am having pain in my neck and also have pain in my eyes as well",
        [], DEPARTMENTS, set(),
    )
    assert hinted == {"Orthopedics": "neck/joint pain", "Ophthalmology": "eye symptoms"}


def test_swelling_on_a_limb_hints_orthopedics_not_general_medicine():
    # Reported live (8th report): explicit instruction that swelling on a limb,
    # alongside limb pain, should route to Orthopedics rather than the bare-limb
    # General Medicine default — reverses the earlier "swelling alone is too
    # generic" stance from the 4th report.
    hinted = departments_hinted_by_patient_symptom_words(
        "swelling on hand and leg", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms"}
    assert "General Medicine" not in hinted


def test_swelling_with_a_real_injury_word_still_hints_orthopedics():
    hinted = departments_hinted_by_patient_symptom_words(
        "my hand is swollen and bruised", [], DEPARTMENTS, set()
    )
    assert "Orthopedics" in hinted


def test_leg_pain_after_moving_hints_orthopedics_not_general_medicine():
    # Reported live (8th report): "leg pain after moving my leg" should route to
    # Orthopedics ALONE, not the bare-limb General Medicine default — pain
    # specifically triggered by movement is a concrete orthopedic red flag.
    hinted = departments_hinted_by_patient_symptom_words(
        "i have leg pain after moving my leg", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms"}


def test_bare_leg_pain_with_no_other_signal_still_hints_general_medicine():
    # The plain, symptom-free case still defaults to General Medicine — only an
    # actual injury/movement/swelling signal escalates to Orthopedics.
    hinted = departments_hinted_by_patient_symptom_words(
        "i have pain in my leg", [], DEPARTMENTS, set()
    )
    assert hinted == {"General Medicine": "limb/joint pain"}


def test_hand_pain_while_moving_hints_orthopedics_not_general_medicine():
    # Same movement-triggered escalation as leg pain — applies to the whole
    # shared limb/joint word family, not leg-specific.
    hinted = departments_hinted_by_patient_symptom_words(
        "i have pain in my hand while moving it", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms"}


def test_bare_hand_pain_with_no_other_signal_still_hints_general_medicine():
    hinted = departments_hinted_by_patient_symptom_words(
        "i have pain in my hand", [], DEPARTMENTS, set()
    )
    assert hinted == {"General Medicine": "limb/joint pain"}


def test_inability_to_bear_weight_hints_orthopedics_only():
    # Reported live: "swelling as well and i cant put weight on it" still only got
    # General Medicine — difficulty bearing weight is one of the strongest,
    # most unambiguous orthopedic red flags there is, but "weight" wasn't a
    # tracked injury-signal word at all.
    hinted = departments_hinted_by_patient_symptom_words(
        "there is swelling as well and i cant put weight on it, pain in my leg", [], DEPARTMENTS, set()
    )
    assert hinted == {"Orthopedics": "limb/joint injury symptoms"}


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


def test_bare_head_as_a_body_location_does_not_hint_general_medicine():
    # Reported live: "i am feeling numb... where? in the head... having weakness"
    # got a spurious General Medicine "head pain" card — but the patient never
    # described a headache; "head" was their answer to WHERE the numbness was, a
    # body location for a different symptom, not a pain complaint. Bare "head"
    # alone must not imply "headache".
    hinted = departments_hinted_by_patient_symptom_words(
        "i am feeling numb in the head and having weakness", [], DEPARTMENTS, set()
    )
    assert "General Medicine" not in hinted
    assert hinted == {"Neurology": "neurological symptoms"}


def test_head_with_an_actual_pain_word_still_hints_general_medicine():
    hinted = departments_hinted_by_patient_symptom_words("my head hurts", [], DEPARTMENTS, set())
    assert "General Medicine" in hinted


def test_forehead_pain_hints_general_medicine():
    # Reported live: "forehead pain" with no other symptoms produced ZERO hints at
    # all — "forehead" tokenizes as its own single word (re.findall splits on word
    # boundaries, not substrings), so it never matched bare "head" here, unlike
    # "headache"/"migraine" (their own unconditional table entry) or "head" + a
    # pain word just above. Confirmed as the correct target: plain, isolated
    # forehead pain with no red-flag/neuro signs is standard General Medicine
    # territory, same reasoning as bare "head" + pain — this makes that outcome
    # reliable instead of left entirely to the LLM's own free-form judgment.
    hinted = departments_hinted_by_patient_symptom_words("i have forehead pain", [], DEPARTMENTS, set())
    assert "General Medicine" in hinted


def test_bare_forehead_as_a_body_location_does_not_hint_general_medicine():
    # Same corroboration requirement as bare "head" above — a bare "forehead"
    # mention with no actual pain word must not imply a pain complaint either.
    hinted = departments_hinted_by_patient_symptom_words(
        "i am feeling numb in the forehead and having weakness", [], DEPARTMENTS, set()
    )
    assert "General Medicine" not in hinted
    assert hinted == {"Neurology": "neurological symptoms"}


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
