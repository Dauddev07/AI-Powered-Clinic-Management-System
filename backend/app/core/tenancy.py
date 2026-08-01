import uuid
from dataclasses import dataclass
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import Base

ModelT = TypeVar("ModelT", bound=Base)


@dataclass(frozen=True)
class ClinicContext:
    """Built once per request, from the verified JWT only — never from client input."""

    clinic_id: uuid.UUID
    user_id: uuid.UUID
    role: str


class ClinicScopedRepo(Generic[ModelT]):
    """Wraps a single clinic-scoped model so every read is automatically filtered by
    clinic_id and every write automatically has clinic_id injected. clinic_id always
    comes from the request's ClinicContext (i.e. the verified JWT) — any clinic_id
    passed by a caller into create()/update() is discarded, so an endpoint cannot
    accidentally leak or write another tenant's data even if it forgets to filter.
    """

    def __init__(self, db: Session, model: type[ModelT], ctx: ClinicContext):
        self.db = db
        self.model = model
        self.ctx = ctx

    def _base_query(self):
        return select(self.model).where(self.model.clinic_id == self.ctx.clinic_id)

    def get(self, id_: uuid.UUID) -> ModelT | None:
        stmt = self._base_query().where(self.model.id == id_)
        return self.db.execute(stmt).scalar_one_or_none()

    def list(self, *extra_filters) -> list[ModelT]:
        stmt = self._base_query()
        if extra_filters:
            stmt = stmt.where(*extra_filters)
        return list(self.db.execute(stmt).scalars().all())

    def create(self, **kwargs) -> ModelT:
        kwargs.pop("clinic_id", None)
        obj = self.model(clinic_id=self.ctx.clinic_id, **kwargs)
        self.db.add(obj)
        self.db.flush()
        return obj

    def update(self, obj: ModelT, **kwargs) -> ModelT:
        if obj.clinic_id != self.ctx.clinic_id:
            raise PermissionError("cross-clinic update blocked")
        kwargs.pop("clinic_id", None)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        self.db.flush()
        return obj

    def delete(self, obj: ModelT) -> None:
        if obj.clinic_id != self.ctx.clinic_id:
            raise PermissionError("cross-clinic delete blocked")
        self.db.delete(obj)


class ClinicScope:
    """Factory handed to endpoints: scope(Model) -> ClinicScopedRepo bound to the
    current request's clinic. See app.api.deps.get_clinic_scope for the FastAPI
    dependency that constructs this from the verified JWT.
    """

    def __init__(self, db: Session, ctx: ClinicContext):
        self.db = db
        self.ctx = ctx

    def __call__(self, model: type[ModelT]) -> ClinicScopedRepo[ModelT]:
        return ClinicScopedRepo(self.db, model, self.ctx)
