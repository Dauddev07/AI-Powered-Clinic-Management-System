import pytest
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from app.core.db import engine


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
