import pytest

from app.services.diagnosis_guard import (
    SAFE_REDIRECT_EN,
    SAFE_REDIRECT_UR,
    enforce_no_diagnosis,
    violates_no_diagnosis_rule,
)

# --- positive cases: an actual diagnosis attempt must be caught ---------------------


@pytest.mark.parametrize(
    "reply",
    [
        "You might have the flu",
        "This sounds like a migraine",
        "Based on your symptoms, this could be appendicitis",
        "you might have diabetes",
        "this is probably a bacterial infection",
        "I think you have a stomach ulcer",
        "this could be gastritis",
        "it is likely a migraine",
        "it's most likely a kidney stone",
        "you're suffering from anxiety",
        "you're dealing with a chronic condition",
        "I can diagnose that as bronchitis",
        "you are experiencing an asthma attack",
        "this looks like a viral infection",
    ],
)
def test_diagnostic_phrasing_is_flagged(reply):
    assert violates_no_diagnosis_rule(reply)


# --- negative cases: ordinary, non-diagnostic bot replies must NOT be flagged -------


@pytest.mark.parametrize(
    "reply",
    [
        # Regression cases from the SOW's explicit "over-broad" warning.
        "this looks like a great time to book",
        "this is a good idea",
        "this is a great question",
        "that sounds like a plan",
        "it is probably fine to wait a day",
        "it is likely open until 6pm",
        # get_my_appointments-style summaries the agent composes itself.
        "you have an appointment tomorrow at 3pm",
        "you have 2 upcoming appointments",
        "you might have missed your last visit",
        "you could have booked online too",
        "this is a good time to book cardiology",
        # Plain triage/booking conversation.
        "Could you tell me how long you've had this pain?",
        "I recommend booking with cardiology for this.",
        "Which department would you like to book with?",
        # Regression cases: benign clarifying-question lead-ins that name no
        # condition but were being caught by the same pattern that flags "you
        # might have the flu" — see diagnosis_guard.py's _BENIGN_CONTINUATIONS
        # comment on the hedging-word additions.
        "This is a common combination of symptoms. Could you tell me how long you've had the headache?",
        "It's probably nothing serious, but let's ask a couple of quick questions first.",
        "This could be a number of things. Is the headache the worst you've ever had?",
        "This is a typical presentation. Is your headache the worst you've ever had?",
        # Regression: reproduces a reported bug where a genuine, safe PATH 2/3
        # screening question was discarded and replaced with the generic
        # SAFE_REDIRECT text because it contained "you have any" as part of the
        # QUESTION form "do you have any ...", not an assertion of a diagnosis.
        "Do you have any other symptoms such as cough, sore throat, sinus pressure, "
        "body aches, or nasal congestion?",
        "Did you have any nausea or vomiting along with the headache?",
    ],
)
def test_ordinary_non_diagnostic_replies_are_not_flagged(reply):
    assert not violates_no_diagnosis_rule(reply)


# --- enforce_no_diagnosis behavior ---------------------------------------------------


def test_enforce_no_diagnosis_passes_through_safe_reply_unchanged():
    reply = "Could you tell me how long you've had this pain?"
    assert enforce_no_diagnosis(reply, "en") == reply


def test_enforce_no_diagnosis_replaces_diagnostic_reply_with_safe_redirect_en():
    assert enforce_no_diagnosis("You might have the flu", "en") == SAFE_REDIRECT_EN


def test_enforce_no_diagnosis_replaces_diagnostic_reply_with_safe_redirect_ur():
    assert enforce_no_diagnosis("You might have the flu", "ur") == SAFE_REDIRECT_UR


def test_enforce_no_diagnosis_never_partially_edits_a_flagged_reply():
    # The replacement must be the whole fixed redirect, never the original text with
    # just the diagnostic fragment stripped out — a partial edit could still leave a
    # named condition in place elsewhere in the sentence.
    reply = "You might have the flu, but let's book you into General Medicine anyway."
    result = enforce_no_diagnosis(reply, "en")
    assert result == SAFE_REDIRECT_EN
    assert "flu" not in result


# --- regression: required emergency/urgency phrasing must never be false-flagged ----
#
# Reproduces a reported bug: a patient described high fever + 1-day duration + neck
# stiffness (a PATH 2 screening answer that should trigger PATH 1's emergency
# escalation), and separately a thumb injury with swelling — in both real
# conversations, the model's required reply (naming no condition, just urgency/
# reasoning) was silently swapped for the generic SAFE_REDIRECT text because "this
# sounds like an emergency" and "this could be a sprain or a fracture" match the same
# structural pattern as an actual diagnosis ("this sounds like the flu") and neither
# "emergency" nor "sprain"/"fracture" were on the benign-continuation allowlist. The
# system prompt explicitly requires this exact phrasing for a real emergency ("this
# sounds like an emergency, please seek immediate care... names no condition and
# diagnoses nothing") and explicitly permits hedged injury-type language as
# reassurance ("this sounds like it may just be a mild muscle strain") — losing this
# text left the patient with a useless non-answer instead of urgent-care guidance.


@pytest.mark.parametrize(
    "reply",
    [
        "This sounds like an emergency — please go to the nearest emergency room right away. "
        "While you get there, try to stay calm and avoid moving your neck.",
        "This could be a medical emergency given the high fever and neck stiffness together — "
        "please seek immediate care.",
        "This sounds like it could need urgent attention. Is there any confusion or sensitivity "
        "to light along with the neck stiffness?",
        "This sounds like it could be a sprain or a fracture from the impact. Is there any "
        "numbness, or can you move your thumb normally?",
        "This could be a fracture or a bad sprain — is there any numbness, visible deformity, "
        "or does the pain get worse when you try to move it?",
        "This sounds like it may just be a mild muscle strain from sleeping awkwardly, but "
        "here is who to see if it does not improve.",
    ],
)
def test_required_emergency_and_hedged_injury_phrasing_is_never_flagged(reply):
    assert not violates_no_diagnosis_rule(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "This is the flu, so just rest and drink fluids.",
        "You have appendicitis based on that pain location.",
        "This sounds like meningitis given the fever and stiff neck.",
        "It is probably strep throat.",
    ],
)
def test_actual_disease_names_are_still_flagged_after_the_allowlist_expansion(reply):
    # The fix above must not have widened the allowlist so far that it stops catching
    # a genuine named diagnosis — only the emergency/urgency and generic-injury
    # vocabulary was added, real disease names are untouched.
    assert violates_no_diagnosis_rule(reply)
