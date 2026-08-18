"""Refresh-token issuance, rotation, and revocation — the counterpart to the
short-lived access token created by app.core.security.create_access_token. See
app/models/refresh_token.py for why rows are hashed and rotated rather than just
re-verified in place.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import generate_refresh_token_value, hash_refresh_token
from app.models.refresh_token import RefreshToken
from app.models.user import User


class InvalidRefreshToken(Exception):
    """Unknown, already-revoked, or expired token — every case collapses to this one
    exception so a caller can't distinguish "never existed" from "was valid once"."""


def issue_refresh_token(db: Session, user: User) -> str:
    plaintext = generate_refresh_token_value()
    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(plaintext),
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
    )
    db.commit()
    return plaintext


def _find_active_token(db: Session, plaintext: str) -> RefreshToken | None:
    row = db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(plaintext))
    ).scalars().first()
    if row is None or row.revoked_at is not None or row.expires_at < datetime.now(timezone.utc):
        return None
    return row


def rotate_refresh_token(db: Session, plaintext: str) -> tuple[User, str]:
    """Validates `plaintext`, revokes it, and issues a fresh replacement — used by
    POST /auth/refresh. Raises InvalidRefreshToken for anything that isn't a
    currently-active token, including one that's already been rotated once before
    (a real client only ever presents a given refresh token once; a second
    presentation of an already-rotated one is exactly what a stolen/replayed token
    looks like).
    """
    row = _find_active_token(db, plaintext)
    if row is None:
        raise InvalidRefreshToken()

    user = db.get(User, row.user_id)
    if user is None or not user.is_active:
        raise InvalidRefreshToken()

    row.revoked_at = datetime.now(timezone.utc)
    new_plaintext = issue_refresh_token(db, user)
    return user, new_plaintext


def revoke_refresh_token(db: Session, plaintext: str) -> None:
    """Backs POST /auth/logout. Silently a no-op for an unknown/already-revoked
    token — logout should never error just because the client's local copy was
    already stale (e.g. a double-click, or a token that expired naturally).
    """
    row = _find_active_token(db, plaintext)
    if row is None:
        return
    row.revoked_at = datetime.now(timezone.utc)
    db.commit()
