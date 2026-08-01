# AI-Powered Clinic Management System

Multi-tenant clinic intake + appointment app with an AI triage/booking chatbot.

- **Superadmin**: CLI-only, manages clinics (tenants).
- **Admin**: one per clinic/branch, manages departments/doctors/shifts via the backend.
- **Patient**: registers within a clinic, books appointments via the chatbot.

A "clinic" is one tenant = one physical branch. There is no cross-branch oversight.
Every core table is scoped by a non-nullable `clinic_id`, always resolved from the
verified JWT server-side — never from client input. See
[docs/architecture-data-dictionary.md](docs/architecture-data-dictionary.md) for the
full ERD and data dictionary.

## Stack

- Backend: FastAPI, PostgreSQL, SQLAlchemy 2.x, Alembic, Pydantic v2, JWT (python-jose), bcrypt (passlib)
- Frontend: React (Vite) + CSS Modules
- RAG (later modules): LangChain, ChromaDB, sentence-transformers (all-mpnet-base-v2), BM25 EnsembleRetriever
- LLM: Groq/Gemini via server-side function-calling (key never exposed to the client)

## Repo layout

```
backend/
  app/
    core/       # settings, db session
    models/     # SQLAlchemy models
    schemas/    # Pydantic schemas
    api/        # FastAPI routers
    services/   # business logic (booking rules, tenant scoping, etc.)
    rag/        # retrieval / embedding pipeline
    scripts/    # one-off / CLI scripts (incl. superadmin CLI)
  alembic/      # migrations
  tests/
frontend/       # React (Vite) app
docs/           # architecture & data dictionary
docker-compose.yml  # Postgres + ChromaDB for local dev
```

## Local setup

1. Create and activate the virtual environment (always use this venv, never system Python):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. Copy the environment template and adjust as needed:

   ```bash
   cp .env.example backend/.env
   ```

3. Start local infrastructure (Postgres + ChromaDB):

   ```bash
   docker-compose up -d
   ```

4. Run the API:

   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

5. Check health:

   ```bash
   curl http://localhost:8000/health
   ```

Cloud hosting is client-provided later; this docker-compose setup is local development
infrastructure only.

## Adding a new clinic

Adding clinic N+1 is configuration + data only — insert a `clinics` row (and its admin
user, departments, doctors, shifts) via the superadmin CLI. No code change is required.
