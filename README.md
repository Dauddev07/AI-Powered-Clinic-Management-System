# AI-Powered Clinic Management System

Multi-tenant clinic intake + appointment app with an AI triage/booking chatbot.
Live at Render (backend) + Vercel (frontend): **https://ai-powered-clinic-management-system.vercel.app**

- **Superadmin**: manages clinics (tenants) via a seeding script — no code change
  needed to onboard a new clinic (see [Adding a new clinic](#adding-a-new-clinic)).
- **Admin**: one per clinic/branch — manages departments, doctors, shifts, the
  chatbot's knowledge base, and reviews patient feedback via the web app.
- **Patient**: registers within a clinic, books/reschedules/cancels appointments
  either through the AI chatbot or the booking UI directly.

A "clinic" is one tenant = one physical branch. There is no cross-branch oversight.
Every core table is scoped by a non-nullable `clinic_id`, always resolved from the
verified JWT server-side — never from client input.

## Stack

- **Backend**: FastAPI, PostgreSQL, SQLAlchemy 2.x, Alembic, Pydantic v2, JWT
  (python-jose), bcrypt (passlib), APScheduler (background jobs)
- **Frontend**: React 19 (Vite), React Router, CSS Modules, Recharts (admin dashboard
  charts)
- **AI chatbot**: LangChain + Groq (`langchain-groq`) — a deterministic intent
  router classifies each patient message to one of three specialist agents
  (symptom triage, appointment booking, general clinic info) before any LLM
  tool-calling loop runs; most turns resolve without an LLM call at all
- **RAG / knowledge base**: ChromaDB (via Docker locally, HttpClient in
  production), BM25 + embedding ensemble retrieval, hosted embeddings via the
  Hugging Face Inference API (`BAAI/bge-base-en-v1.5`) rather than running
  sentence-transformers/torch in-process
- **Tracing**: LangSmith (optional — auto-detected from environment)

## Repo layout

```
backend/
  app/
    core/         # settings, db session, API key rotation
    models/       # SQLAlchemy models
    schemas/      # Pydantic schemas
    api/          # FastAPI routers (auth, appointments, admin_*, chat, ...)
    services/     # business logic
      orchestrator/   # chatbot intent router + specialist agents
        agents/         # appointment / symptom / general-info agents
    rag/          # ChromaDB client + embedding pipeline
    scripts/      # one-off scripts (clinic seeding, maintenance)
  alembic/        # migrations
  tests/
frontend/
  src/
    api/          # fetch wrappers per backend resource
    pages/        # patient/, admin/, and top-level routes
    components/   # shared UI (nav, header, toasts, cards, ...)
    auth/         # auth context + route guards
    hooks/        # shared React hooks
docker-compose.yml  # Postgres + ChromaDB for local dev
```

## Features

- **Patient**: chatbot-driven symptom triage → department/doctor suggestion,
  real-time slot booking, reschedule/cancel (with disambiguation when multiple
  appointments match), appointment history, post-visit doctor ratings,
  in-app notifications, profile management.
- **Admin**: dashboard (slot utilization, busiest doctors, booking trend,
  top-rated doctors), doctor roster + CSV bulk import with ingestion history,
  knowledge base document upload for the chatbot, patient feedback review.
- **Persistent navigation**: a desktop nav row / mobile bottom tab bar surfaces
  the most-used screens per role; the account menu covers the rest (profile,
  settings, CSV import, logout).
- Light/dark theme, fully token-driven (no hardcoded colors in components).

## Local setup

1. Create and activate the virtual environment (always use this venv, never system
   Python):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. Copy the environment template and fill in the values you need (at minimum
   `DATABASE_URL`, `JWT_SECRET`, `LLM_API_KEY` for a Groq key):

   ```bash
   cp .env.example backend/.env
   ```

3. Start local infrastructure (Postgres + ChromaDB):

   ```bash
   docker-compose up -d
   ```

4. Run migrations:

   ```bash
   cd backend
   alembic upgrade head
   ```

5. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

6. In a separate terminal, run the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   By default it points at `http://localhost:8000` (see `frontend/.env`).

7. Check health:

   ```bash
   curl http://localhost:8000/health
   ```

## Adding a new clinic

Adding clinic N+1 is configuration + data only — seed a `clinics` row (and its admin
user, departments, doctors, shifts) via `backend/app/scripts/seed_clinic.py`. No code
change is required.

## Deployment

- **Backend**: Render (free tier) — Postgres + a ChromaDB service, both on Render.
  Free-tier services spin down after ~15 minutes idle; an uptime monitor keeps both
  warm.
- **Frontend**: Vercel, built from `frontend/`.
- Both auto-deploy on push to `main`.
