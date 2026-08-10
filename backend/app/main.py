import logging
import os
from contextlib import asynccontextmanager

from app.core.config import settings

# MUST run before any LangChain/LangSmith-instrumented module is imported — the
# `app.api` import below pulls in app.services.llm (ChatGroq) and app.rag.retrieval
# (EnsembleRetriever/Chroma), both of which participate in LangSmith tracing via
# LangChain's runnable callbacks. pydantic-settings loads LANGSMITH_* into `settings`,
# but LangSmith's tracing library reads os.environ directly at import/call time — it
# has no idea this app's Settings object exists — so without this, tracing silently
# never activates no matter how correct .env looks.
os.environ["LANGSMITH_TRACING"] = str(settings.LANGSMITH_TRACING).lower()
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

# A dedicated, non-propagating logger — not logging.basicConfig(level=INFO), which
# would also crank every third-party library's logger (httpx, langchain, ...) up to
# INFO and flood the startup log with unrelated noise.
_startup_logger = logging.getLogger("app.startup")
_startup_logger.setLevel(logging.INFO)
_startup_logger.propagate = False
if not _startup_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _startup_logger.addHandler(_handler)

_startup_logger.info(
    "LangSmith tracing configured: tracing=%s, api_key_set=%s, project=%r",
    settings.LANGSMITH_TRACING,
    bool(settings.LANGSMITH_API_KEY),
    settings.LANGSMITH_PROJECT,
)

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api import (
    admin_dashboard,
    admin_doctors,
    admin_feedback,
    admin_kb,
    appointments,
    auth,
    chat,
    clinics,
    notifications,
    patients,
    slots,
)
from app.core.db import engine
from app.services.scheduler import shutdown_scheduler, start_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(title="AI-Powered Clinic Management System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Vite auto-increments to the next free port (5174, 5175, ...) whenever 5173
    # is already taken by another running dev server, so more than one of these
    # can be a legitimate local frontend rather than a stray process.
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(clinics.router)
app.include_router(patients.router)
app.include_router(admin_doctors.router)
app.include_router(admin_dashboard.router)
app.include_router(slots.router)
app.include_router(appointments.router)
app.include_router(admin_kb.router)
app.include_router(chat.router)
app.include_router(notifications.router)
app.include_router(admin_feedback.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Flattens Pydantic's structured error list into a single readable string, so the
    frontend's ApiError (which only surfaces a string `detail`) shows a specific,
    field-by-field message instead of falling back to a generic error.
    """
    messages = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = loc[-1] if loc else "field"
        msg = err.get("msg", "Invalid value")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        messages.append(f"{field}: {msg}")

    detail = "; ".join(messages) if messages else "Invalid request"
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": detail})


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {"status": "ok", "database": db_status}
