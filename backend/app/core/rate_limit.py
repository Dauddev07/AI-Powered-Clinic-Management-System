import sys

from slowapi import Limiter
from slowapi.util import get_remote_address

# Disabled under pytest: several tests (tests/test_auth_api.py) call route functions
# like login()/register() directly as plain Python functions, bypassing the ASGI
# request cycle entirely — there's no real starlette.Request for slowapi to key off
# of. slowapi's own wrapper only dereferences the `request` argument when
# self.enabled is True (see slowapi.extension.Limiter.limit's sync/async wrapper), so
# leaving it enabled would make every one of those direct calls raise. "pytest" is
# only ever in sys.modules when running under pytest, so production behavior (and
# any real HTTP-level test via TestClient) is unaffected.
_RUNNING_UNDER_PYTEST = "pytest" in sys.modules

# In-memory storage (the slowapi/limits default) — fine for this app's current
# single-process deployment. Would need a shared backend (e.g. Redis, via
# storage_uri="redis://...") the moment this runs as more than one worker/instance,
# since each process would otherwise track its own separate counters.
limiter = Limiter(key_func=get_remote_address, enabled=not _RUNNING_UNDER_PYTEST)
