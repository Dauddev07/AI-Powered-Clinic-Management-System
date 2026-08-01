import uuid
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.security import InvalidTokenError, decode_access_token
from app.core.tenancy import ClinicContext, ClinicScope
from app.models.user import User

_bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Decodes and verifies the JWT, loads the user, and is the sole source of the
    request's clinic identity. Any endpoint that omits this dependency (directly or via
    get_clinic_scope/require_role) has no way to obtain a clinic-scoped session, so it
    fails closed rather than risking a cross-tenant leak.
    """
    if credentials is None:
        raise _UNAUTHORIZED

    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidTokenError:
        raise _UNAUTHORIZED

    try:
        user_id = uuid.UUID(payload["sub"])
    except (ValueError, TypeError):
        raise _UNAUTHORIZED

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _UNAUTHORIZED

    # Defense against a stale token issued before a role/clinic change: the JWT is
    # trustworthy (server-signed), but we still cross-check it against the current DB row.
    if str(user.clinic_id) != payload["clinic_id"] or user.role != payload["role"]:
        raise _UNAUTHORIZED

    # A password change immediately invalidates every token issued before it, even
    # though the JWT itself is still cryptographically valid and unexpired.
    if user.password_changed_at is not None:
        try:
            token_issued_at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        except (KeyError, TypeError, ValueError):
            raise _UNAUTHORIZED
        if token_issued_at < user.password_changed_at:
            raise _UNAUTHORIZED

    return user


def get_clinic_context(user: User = Depends(get_current_user)) -> ClinicContext:
    return ClinicContext(clinic_id=user.clinic_id, user_id=user.id, role=user.role)


def get_clinic_scope(
    db: Session = Depends(get_db),
    ctx: ClinicContext = Depends(get_clinic_context),
) -> ClinicScope:
    return ClinicScope(db, ctx)


def require_role(*roles: str):
    def _dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user

    return _dependency
