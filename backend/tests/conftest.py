import pytest
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from app.core.db import engine
from app.services import email as email_service


@pytest.fixture(autouse=True)
def no_real_emails(monkeypatch):
    """Booking/cancel/reschedule/reminder emails (app/services/email.py) fire off a
    background thread straight from the service layer — unlike the OTP/welcome emails,
    which route through FastAPI's BackgroundTasks and never actually execute in a test
    that just constructs BackgroundTasks() without running it. Autouse and session-wide
    so no test anywhere can accidentally fire a real Brevo API call using this repo's
    real .env credentials against a throwaway uuid@example.com test address.
    """
    monkeypatch.setattr(email_service, "send_email", lambda **kwargs: None)


@pytest.fixture
def db():
    """A Session bound to a single connection wrapped in an outer transaction that's
    always rolled back — every test runs against the real (dockerized) Postgres schema
    (UUID columns etc. need the real dialect) but leaves no trace behind.
    """
    connection = engine.connect()
    outer_txn = connection.begin()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = SessionLocal()

    # Nested SAVEPOINT so code under test can call session.commit()/rollback() freely
    # (as validate_csv's callers do) without ending the outer transaction early.
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess, transaction):
        if transaction.nested and not transaction._parent.nested:
            sess.begin_nested()

    try:
        yield session
    finally:
        session.close()
        outer_txn.rollback()
        connection.close()
