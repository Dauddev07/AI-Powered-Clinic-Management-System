"""Thread-safe round-robin rotation across up to 3 Groq API keys.

Every ChatGroq instantiation in app.services.llm pulls its key from api_key_manager
here instead of a single fixed settings.LLM_API_KEY — this covers every Groq call
(primary model, fallback model, retry) uniformly, without any of that call-site
logic needing to know rotation exists. Order is fixed Key1 -> Key2 -> Key3 -> Key1
..., never randomized, so distribution across keys stays even and predictable.

Only whichever of LLM_API_KEY / LLM_API_KEY_2 / LLM_API_KEY_3 are actually non-empty
participate — an environment with only LLM_API_KEY set (the pre-rotation default)
still works exactly as before, just without anything to rotate across.
"""
import itertools
import threading

from app.core.config import settings


class ApiKeyManager:
    def __init__(self, keys: list[str]):
        self._keys = [key for key in keys if key]
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(self._keys) if self._keys else None

    def next_key(self) -> str:
        """Returns the next key in round-robin order. Thread-safe: itertools.cycle's
        own __next__ isn't safe under concurrent access, so the advance is guarded by
        a lock rather than relying on it directly.
        """
        if self._cycle is None:
            return ""
        with self._lock:
            return next(self._cycle)


api_key_manager = ApiKeyManager([settings.LLM_API_KEY, settings.LLM_API_KEY_2, settings.LLM_API_KEY_3])
