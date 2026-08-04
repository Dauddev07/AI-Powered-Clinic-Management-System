import uuid

import pytest

from app.models.clinic import Clinic
from app.models.conversation_memory import ConversationMemory
from app.models.patient_memory_profile import PatientMemoryProfile
from app.models.user import User
from app.services import memory_summary
from app.services.memory_summary import refresh_patient_summary_for_new_session


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}")
    db.add(c)
    db.flush()
    return c


@pytest.fixture
def patient(db, clinic):
    p = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Pat Ient",
    )
    db.add(p)
    db.flush()
    return p


def _message(db, clinic, patient, role, content, session_id=None):
    row = ConversationMemory(
        clinic_id=clinic.id, session_id=session_id or uuid.uuid4(), user_id=patient.id, role=role, content=content,
    )
    db.add(row)
    db.flush()
    return row


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        from types import SimpleNamespace

        return SimpleNamespace(content=self._content)


def test_system_prompt_captures_general_topics_asked_about_not_just_symptoms_and_personal_info():
    # Reported gap: a chat containing only logistics questions ("is there any
    # cardiologist available on fri") had nothing extracted at all (symptoms/personal
    # info are the only two categories the original prompt captured), so a later
    # "what are the things I told you" in a new chat got "I don't have anything
    # stored" even though the patient clearly had asked something. The digest must
    # capture the general SUBJECT of any question asked, not just symptoms/personal
    # facts — while still excluding volatile specifics (booking results, exact
    # date/time/slot details) that can go stale.
    assert "The general SUBJECT of anything else the patient asked about" in memory_summary._SYSTEM_PROMPT
    assert "the RESULT of any appointment booking/reschedule/cancellation" in memory_summary._SYSTEM_PROMPT
    assert "Capture only the topic/subject, never the specific answer given" in memory_summary._SYSTEM_PROMPT


def test_no_prior_messages_returns_empty_summary_and_creates_a_profile_row(db, clinic, patient):
    result = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    assert result == ""
    profile = db.query(PatientMemoryProfile).filter_by(clinic_id=clinic.id, user_id=patient.id).one()
    assert profile.summary_text == ""
    assert profile.last_summarized_seq == 0


def test_new_messages_are_summarized_and_persisted(db, clinic, patient, monkeypatch):
    _message(db, clinic, patient, "user", "I have recurring migraines.")
    _message(db, clinic, patient, "assistant", "Noted, let's find you a Neurology slot.")

    monkeypatch.setattr(memory_summary, "ChatGroq", lambda **kwargs: _FakeLLM("Patient has recurring migraines."))

    result = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    assert result == "Patient has recurring migraines."
    profile = db.query(PatientMemoryProfile).filter_by(clinic_id=clinic.id, user_id=patient.id).one()
    assert profile.summary_text == "Patient has recurring migraines."
    assert profile.last_summarized_seq > 0


def test_already_summarized_messages_are_not_resent_on_a_later_refresh(db, clinic, patient, monkeypatch):
    _message(db, clinic, patient, "user", "I have recurring migraines.")

    captured = []

    def fake_chatgroq(**kwargs):
        return _FakeLLM("Patient has recurring migraines.")

    monkeypatch.setattr(memory_summary, "ChatGroq", fake_chatgroq)
    first = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)
    assert first == "Patient has recurring migraines."

    # No new messages since — a second refresh (e.g. a second new session) must be a
    # cheap no-op: the existing summary is returned without another LLM call.
    def fail_if_called(**kwargs):
        raise AssertionError("ChatGroq must not be invoked when there's nothing new to summarize")

    monkeypatch.setattr(memory_summary, "ChatGroq", fail_if_called)
    second = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    assert second == "Patient has recurring migraines."


def test_only_messages_since_last_summarized_seq_are_sent_to_the_llm(db, clinic, patient, monkeypatch):
    _message(db, clinic, patient, "user", "I have recurring migraines.")

    monkeypatch.setattr(memory_summary, "ChatGroq", lambda **kwargs: _FakeLLM("Patient has recurring migraines."))
    refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    _message(db, clinic, patient, "user", "Also, I'm allergic to penicillin.")

    seen_messages = {}

    def fake_chatgroq(**kwargs):
        class _Capturing(_FakeLLM):
            def invoke(self, messages):
                seen_messages["text"] = messages[1].content
                return super().invoke(messages)

        return _Capturing("Patient has recurring migraines and a penicillin allergy.")

    monkeypatch.setattr(memory_summary, "ChatGroq", fake_chatgroq)
    result = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    assert result == "Patient has recurring migraines and a penicillin allergy."
    new_messages_section = seen_messages["text"].split("New messages:\n")[1]
    assert new_messages_section.strip() == "user: Also, I'm allergic to penicillin.", (
        "already-summarized messages should not be resent as 'new messages' — only "
        "genuinely new rows since last_summarized_seq"
    )
    assert "Existing summary: Patient has recurring migraines." in seen_messages["text"], (
        "the existing summary must still be passed as context"
    )


def test_llm_failure_leaves_existing_summary_untouched_but_advances_the_watermark(db, clinic, patient, monkeypatch):
    _message(db, clinic, patient, "user", "I have recurring migraines.")
    monkeypatch.setattr(memory_summary, "ChatGroq", lambda **kwargs: _FakeLLM("Patient has recurring migraines."))
    refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    _message(db, clinic, patient, "user", "Some new message.")
    monkeypatch.setattr(
        memory_summary, "ChatGroq", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    result = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    # Existing summary preserved rather than lost...
    assert result == "Patient has recurring migraines."
    # ...but the watermark still advances, so a persistently failing call doesn't
    # force an ever-growing backlog to be re-attempted on every future new session.
    profile = db.query(PatientMemoryProfile).filter_by(clinic_id=clinic.id, user_id=patient.id).one()
    latest_seq = db.query(ConversationMemory.seq).filter_by(user_id=patient.id).order_by(
        ConversationMemory.seq.desc()
    ).first()[0]
    assert profile.last_summarized_seq == latest_seq


def test_no_llm_api_key_configured_leaves_existing_summary_untouched(db, clinic, patient, monkeypatch):
    from app.core.config import settings

    _message(db, clinic, patient, "user", "I have recurring migraines.")
    monkeypatch.setattr(settings, "LLM_API_KEY", "")

    result = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    assert result == ""


def test_degenerate_overlong_llm_output_is_discarded(db, clinic, patient, monkeypatch):
    _message(db, clinic, patient, "user", "I have recurring migraines.")
    monkeypatch.setattr(memory_summary, "ChatGroq", lambda **kwargs: _FakeLLM("x" * 5000))

    result = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    assert result == ""


def test_different_patients_and_clinics_never_mix_summaries(db, clinic, patient, monkeypatch):
    other_patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Other Patient",
    )
    db.add(other_patient)
    db.flush()

    _message(db, clinic, patient, "user", "I have recurring migraines.")
    _message(db, clinic, other_patient, "user", "I have a broken wrist.")

    monkeypatch.setattr(memory_summary, "ChatGroq", lambda **kwargs: _FakeLLM("Patient has recurring migraines."))
    result_a = refresh_patient_summary_for_new_session(db, clinic.id, patient.id)

    monkeypatch.setattr(memory_summary, "ChatGroq", lambda **kwargs: _FakeLLM("Patient has a broken wrist."))
    result_b = refresh_patient_summary_for_new_session(db, clinic.id, other_patient.id)

    assert result_a == "Patient has recurring migraines."
    assert result_b == "Patient has a broken wrist."
