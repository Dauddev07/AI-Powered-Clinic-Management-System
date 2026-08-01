import pytest

from app.services.language import detect_language


# --- Perso-Arabic script (primary, unchanged) ----------------------------------------


def test_urdu_script_message_detected_as_urdu():
    assert detect_language("کلینک کب کھلتا ہے؟") == "ur"


def test_urdu_script_takes_priority_even_when_mixed_with_english():
    assert detect_language("Hello, کلینک کب کھلتا ہے؟") == "ur"


# --- English (including Roman Urdu, which has no script signal and is out of scope) --


@pytest.mark.parametrize(
    "message",
    [
        "How many departments does this clinic have?",
        "I have a fever and a headache.",
        "What time does the clinic open?",
        "Can I book an appointment with a cardiologist?",
        "How are you today?",
    ],
)
def test_english_message_detected_as_english(message):
    assert detect_language(message) == "en"


def test_roman_urdu_message_is_treated_as_english():
    # Roman Urdu (Latin-script Urdu) has no Perso-Arabic script signal, and Roman
    # Urdu detection was removed — this must fall through to "en", its original
    # pre-Roman-Urdu-fix behavior, not be misdetected as Urdu.
    assert detect_language("mujhay bukhar ho raha hai") == "en"
