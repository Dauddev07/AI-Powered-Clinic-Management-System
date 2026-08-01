from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    LLM_API_KEY: str = ""
    # Optional additional Groq API keys — when set, app.core.api_keys round-robins
    # every Groq call across whichever of the three are non-empty. Unset (the
    # default) just means rotation degenerates to always using LLM_API_KEY, same as
    # before these existed.
    LLM_API_KEY_2: str = ""
    LLM_API_KEY_3: str = ""
    LLM_PROVIDER: str = "groq"
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # LangSmith tracing — LangChain auto-detects these from the process environment,
    # so nothing here needs to call them explicitly. They're declared as Settings
    # fields purely so a missing/malformed value fails loudly at startup instead of
    # tracing silently going dark.
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "Clinic-Management-System"

    CHROMA_PATH: str = "./chroma_data"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"

    DEFAULT_SLOT_MINUTES: int = 30
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.7

    # Minimum cosine similarity (same embedding model as retrieval) for an uploaded
    # doctor-CSV header to be *suggested* as a synonym of a canonical column name.
    # Suggestions are never auto-applied regardless of score — the admin must
    # explicitly confirm each one in the preview UI before it's used.
    HEADER_MAPPING_SIMILARITY_THRESHOLD: float = 0.6

    # How many days forward slot generation materializes bookable slots for. Re-running
    # regeneration (CSV re-upload, admin block-date) only ever diffs within this window.
    SLOT_GENERATION_HORIZON_DAYS: int = 14

    # How often the background job re-checks/rolls the slot horizon forward (see
    # app/services/scheduler.py). Short enough that the horizon can never fall behind
    # by more than one interval, even across a missed tick.
    SLOT_REGENERATION_INTERVAL_MINUTES: int = 30

    # How often the background job auto-completes past-end appointments (see
    # app/services/scheduler.py). This only covers the gap when nobody hits an
    # endpoint after a slot ends — the lazy per-request check still catches everything
    # else immediately, so this interval doesn't need to be as tight as it sounds.
    APPOINTMENT_AUTO_COMPLETE_INTERVAL_MINUTES: int = 10


settings = Settings()
