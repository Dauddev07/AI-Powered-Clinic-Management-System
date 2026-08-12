import uuid

import pytest
from pywebpush import WebPushException

from app.core.config import settings
from app.models.clinic import Clinic
from app.models.notification import Notification
from app.models.push_subscription import PushSubscription
from app.models.user import User
from app.services import push_notifications
from app.services.push_notifications import send_push_for_notification


@pytest.fixture
def clinic(db):
    c = Clinic(name="Test Clinic", slug=f"test-{uuid.uuid4().hex[:8]}", timezone="UTC")
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


@pytest.fixture
def notification(db, clinic, patient):
    n = Notification(
        clinic_id=clinic.id, user_id=patient.id, type="appointment_cancelled",
        message="Your appointment has been cancelled.",
    )
    db.add(n)
    db.flush()
    return n


@pytest.fixture
def subscription(db, clinic, patient):
    s = PushSubscription(
        clinic_id=clinic.id, user_id=patient.id,
        endpoint="https://fcm.googleapis.com/fcm/send/abc123",
        p256dh_key="fake-p256dh-key", auth_key="fake-auth-key",
    )
    db.add(s)
    db.flush()
    return s


@pytest.fixture(autouse=True)
def vapid_configured(monkeypatch):
    """Every test in this file assumes VAPID keys are configured, unless it
    overrides this itself — see test_no_op_when_vapid_not_configured."""
    monkeypatch.setattr(settings, "VAPID_PRIVATE_KEY_B64", "ZmFrZS1wZW0=")  # base64("fake-pem")
    monkeypatch.setattr(settings, "VAPID_PUBLIC_KEY", "fake-public-key")
    monkeypatch.setattr(settings, "VAPID_CONTACT_EMAIL", "clinic@example.com")


def test_no_op_when_vapid_not_configured(db, notification, subscription, monkeypatch):
    monkeypatch.setattr(settings, "VAPID_PRIVATE_KEY_B64", "")
    calls = []
    monkeypatch.setattr(push_notifications, "webpush", lambda **kwargs: calls.append(kwargs))
    send_push_for_notification(db, notification)
    assert calls == []


def test_no_op_when_patient_has_no_subscriptions(db, notification, monkeypatch):
    calls = []
    monkeypatch.setattr(push_notifications, "webpush", lambda **kwargs: calls.append(kwargs))
    send_push_for_notification(db, notification)
    assert calls == []


def test_sends_to_every_subscription_for_the_patient(db, clinic, patient, notification, subscription, monkeypatch):
    second = PushSubscription(
        clinic_id=clinic.id, user_id=patient.id,
        endpoint="https://fcm.googleapis.com/fcm/send/xyz789",
        p256dh_key="fake-p256dh-key-2", auth_key="fake-auth-key-2",
    )
    db.add(second)
    db.flush()

    calls = []
    monkeypatch.setattr(push_notifications, "webpush", lambda **kwargs: calls.append(kwargs))
    send_push_for_notification(db, notification)

    assert len(calls) == 2
    endpoints = {c["subscription_info"]["endpoint"] for c in calls}
    assert endpoints == {subscription.endpoint, second.endpoint}


def test_payload_includes_notification_message(db, notification, subscription, monkeypatch):
    import json

    calls = []
    monkeypatch.setattr(push_notifications, "webpush", lambda **kwargs: calls.append(kwargs))
    send_push_for_notification(db, notification)

    payload = json.loads(calls[0]["data"])
    assert payload["body"] == notification.message
    assert payload["title"]


def test_does_not_send_to_a_different_patients_subscription(db, clinic, notification, monkeypatch):
    other_patient = User(
        clinic_id=clinic.id, role="patient", email=f"{uuid.uuid4().hex}@example.com",
        hashed_password="x", full_name="Someone Else",
    )
    db.add(other_patient)
    db.flush()
    db.add(
        PushSubscription(
            clinic_id=clinic.id, user_id=other_patient.id,
            endpoint="https://fcm.googleapis.com/fcm/send/not-this-one",
            p256dh_key="k", auth_key="a",
        )
    )
    db.flush()

    calls = []
    monkeypatch.setattr(push_notifications, "webpush", lambda **kwargs: calls.append(kwargs))
    send_push_for_notification(db, notification)
    assert calls == []


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_deletes_subscription_on_410_gone(db, notification, subscription, monkeypatch):
    def _raise(**kwargs):
        raise WebPushException("gone", response=_FakeResponse(410))

    monkeypatch.setattr(push_notifications, "webpush", _raise)
    send_push_for_notification(db, notification)

    remaining = db.query(PushSubscription).filter(PushSubscription.id == subscription.id).one_or_none()
    assert remaining is None


def test_keeps_subscription_on_transient_failure(db, notification, subscription, monkeypatch):
    def _raise(**kwargs):
        raise WebPushException("server error", response=_FakeResponse(500))

    monkeypatch.setattr(push_notifications, "webpush", _raise)
    send_push_for_notification(db, notification)

    remaining = db.query(PushSubscription).filter(PushSubscription.id == subscription.id).one_or_none()
    assert remaining is not None


def test_one_subscription_failing_does_not_stop_the_others(db, clinic, patient, notification, subscription, monkeypatch):
    second = PushSubscription(
        clinic_id=clinic.id, user_id=patient.id,
        endpoint="https://fcm.googleapis.com/fcm/send/second",
        p256dh_key="k2", auth_key="a2",
    )
    db.add(second)
    db.flush()

    calls = []

    def _webpush(**kwargs):
        if kwargs["subscription_info"]["endpoint"] == subscription.endpoint:
            raise WebPushException("gone", response=_FakeResponse(410))
        calls.append(kwargs)

    monkeypatch.setattr(push_notifications, "webpush", _webpush)
    send_push_for_notification(db, notification)

    assert len(calls) == 1
    assert calls[0]["subscription_info"]["endpoint"] == second.endpoint
