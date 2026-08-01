import threading
from collections import Counter

from app.core.api_keys import ApiKeyManager


def test_next_key_round_robins_in_fixed_order():
    manager = ApiKeyManager(["key1", "key2", "key3"])

    sequence = [manager.next_key() for _ in range(7)]

    assert sequence == ["key1", "key2", "key3", "key1", "key2", "key3", "key1"]


def test_next_key_skips_empty_keys():
    manager = ApiKeyManager(["key1", "", "key3", ""])

    sequence = [manager.next_key() for _ in range(4)]

    assert sequence == ["key1", "key3", "key1", "key3"]


def test_next_key_returns_empty_string_when_no_keys_configured():
    manager = ApiKeyManager(["", ""])

    assert manager.next_key() == ""


def test_next_key_degenerates_to_single_key_when_only_one_configured():
    manager = ApiKeyManager(["only-key", "", ""])

    assert [manager.next_key() for _ in range(3)] == ["only-key"] * 3


def test_next_key_is_thread_safe_and_stays_evenly_distributed():
    manager = ApiKeyManager(["key1", "key2", "key3"])
    calls_per_thread = 300
    thread_count = 8
    results: list[str] = []
    results_lock = threading.Lock()

    def worker():
        local = [manager.next_key() for _ in range(calls_per_thread)]
        with results_lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every call must have returned one of the configured keys (no corruption/empty
    # string from a race), and the total count must match exactly (no call lost or
    # duplicated from a torn read of the underlying cycle).
    assert len(results) == calls_per_thread * thread_count
    counts = Counter(results)
    assert set(counts) == {"key1", "key2", "key3"}
    # Perfect round-robin under concurrency isn't guaranteed to split exactly evenly
    # (thread scheduling can interleave), but it must stay close — a real bug (e.g.
    # the lock missing entirely) would skew this far more than a small margin.
    total = sum(counts.values())
    for key in counts:
        assert abs(counts[key] - total / 3) < total * 0.05
