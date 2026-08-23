from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    # Short-lived on purpose — this is the token sent on every request. A stolen
    # access token is only ever usable for this long, regardless of whether the
    # theft is ever noticed. Session longevity now comes from the refresh token
    # below instead, which is revocable (see app/models/refresh_token.py).
    JWT_ACCESS_EXPIRE_MINUTES: int = 30
    # How long an unused refresh token stays valid. Each successful /auth/refresh
    # call rotates it (see app/services/refresh_tokens.py), so a session that's
    # actively used never actually hits this ceiling — it only matters for a
    # genuinely abandoned session (patient never returns) or a device the patient
    # never explicitly logs out of.
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    LLM_API_KEY: str = ""
    # Optional additional Groq API keys — when set, app.core.api_keys round-robins
    # every Groq call across whichever of the seven are non-empty. Unset (the
    # default) just means rotation degenerates to always using LLM_API_KEY, same as
    # before these existed.
    LLM_API_KEY_2: str = ""
    LLM_API_KEY_3: str = ""
    LLM_API_KEY_4: str = ""
    LLM_API_KEY_5: str = ""
    LLM_API_KEY_6: str = ""
    LLM_API_KEY_7: str = ""
    LLM_PROVIDER: str = "groq"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    # Used only for small, narrow helper calls (message intent classification, query
    # rewriting) that don't need the main model's reasoning depth — a separate,
    # smaller model here draws from its own rate-limit bucket instead of competing
    # with the main conversational/agent calls for the same TPM budget.
    GROQ_HELPER_MODEL: str = "openai/gpt-oss-20b"

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
    # Set True when CHROMA_HOST points at a public https:// endpoint (e.g. Render's
    # internal service-to-service networking) rather than a local/internal plain-HTTP one.
    CHROMA_SSL: bool = False
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    # Hugging Face Inference API token for hosted embeddings (app/rag/embeddings.py).
    # Works unauthenticated too, just with lower rate limits — get a free one at
    # huggingface.co/settings/tokens if you hit rate limiting.
    HF_TOKEN: str = ""

    # Transactional email (password-reset OTP for now — see app/services/email.py and
    # app/services/password_reset.py). Sent via Brevo's HTTP API rather than raw SMTP —
    # Render blocks outbound connections on SMTP ports (25/465/587) on its free/starter
    # tiers to prevent spam abuse, so smtplib works locally but fails in production with
    # "Network is unreachable". Brevo's API runs over plain HTTPS (443), which isn't
    # blocked, and its free tier (300 emails/day) needs no domain — only the sender
    # email verified in Brevo's dashboard.
    BREVO_API_KEY: str = ""
    BREVO_SENDER_EMAIL: str = ""
    BREVO_SENDER_NAME: str = "QuickCheck Clinic"

    # How long a password-reset OTP stays valid after being emailed.
    PASSWORD_RESET_OTP_TTL_MINUTES: int = 5
    # Failed-guess cap per OTP before it's invalidated and a new one must be requested —
    # without this, 6 digits (1,000,000 possibilities) is brute-forceable.
    PASSWORD_RESET_OTP_MAX_ATTEMPTS: int = 5
    # Minimum time between two OTP requests for the same email, so the "send code"
    # endpoint can't be used to spam a stranger's inbox.
    PASSWORD_RESET_OTP_RESEND_COOLDOWN_SECONDS: int = 60

    # Same three knobs as the password-reset OTP above, for the separate
    # self-registration email-verification OTP (see app/services/email_verification.py)
    # — kept as their own settings rather than reusing the password-reset ones so the
    # two flows' timing can be tuned independently later without coupling them.
    EMAIL_VERIFICATION_OTP_TTL_MINUTES: int = 5
    EMAIL_VERIFICATION_OTP_MAX_ATTEMPTS: int = 5
    EMAIL_VERIFICATION_OTP_RESEND_COOLDOWN_SECONDS: int = 60

    # Same three knobs again, for the patient self-service account-deletion OTP (see
    # app/services/account_deletion.py) — a permanent, irreversible action, so it gets
    # its own independently-tunable timing rather than reusing either flow above.
    ACCOUNT_DELETE_OTP_TTL_MINUTES: int = 5
    ACCOUNT_DELETE_OTP_MAX_ATTEMPTS: int = 5
    ACCOUNT_DELETE_OTP_RESEND_COOLDOWN_SECONDS: int = 60

    DEFAULT_SLOT_MINUTES: int = 30
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.5

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

    # How often the background job checks for the due 60-minutes-before appointment
    # reminder (see app/services/appointment_reminders.py). Unlike the slot-regeneration
    # interval above, this reminder is meant to fire at close to an exact offset from
    # the appointment's start time, not "eventually" — a 1-minute tick keeps the
    # worst-case lateness under a minute.
    APPOINTMENT_REMINDER_INTERVAL_MINUTES: int = 1


settings = Settings()
