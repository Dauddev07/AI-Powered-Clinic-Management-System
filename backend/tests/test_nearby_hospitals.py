import httpx
import pytest

from app.services.nearby_hospitals import _haversine_km, find_nearby_hospitals, nearby_hospitals_block


def test_haversine_zero_distance_for_identical_points():
    assert _haversine_km(31.5204, 74.3587, 31.5204, 74.3587) == pytest.approx(0.0)


def test_haversine_known_distance_lahore_to_karachi():
    # Real-world reference distance (~1050 km) — a loose tolerance since this is
    # just checking the formula is wired correctly, not surveying precision.
    km = _haversine_km(31.5204, 74.3587, 24.8607, 67.0011)
    assert 1000 < km < 1100


def _mock_overpass_response(monkeypatch, elements):
    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": elements}

    def _fake_post(*args, **kwargs):
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", _fake_post)


def test_find_nearby_hospitals_sorts_nearest_first(monkeypatch):
    _mock_overpass_response(
        monkeypatch,
        [
            {"tags": {"name": "Far Hospital"}, "lat": 31.60, "lon": 74.40},
            {"tags": {"name": "Near Hospital", "phone": "+92-42-111"}, "lat": 31.521, "lon": 74.359},
        ],
    )
    results = find_nearby_hospitals(31.5204, 74.3587)
    assert [h["name"] for h in results] == ["Near Hospital", "Far Hospital"]
    assert results[0]["phone"] == "+92-42-111"
    assert results[1]["phone"] is None


def test_find_nearby_hospitals_ranks_general_hospital_above_closer_specialty_one(monkeypatch):
    # Reported live: a broken-leg emergency surfaced the nearest hospital by raw
    # distance alone, which happened to be an Eye Hospital — not equipped for
    # unrelated trauma. A same-distance-ish General Hospital tagged emergency=yes
    # should now outrank the closer but single-specialty one.
    _mock_overpass_response(
        monkeypatch,
        [
            {"tags": {"name": "Lions Eye Hospital"}, "lat": 31.5205, "lon": 74.3588},
            {"tags": {"name": "City General Hospital", "emergency": "yes"}, "lat": 31.530, "lon": 74.370},
        ],
    )
    results = find_nearby_hospitals(31.5204, 74.3587)
    assert results[0]["name"] == "City General Hospital"
    assert results[1]["name"] == "Lions Eye Hospital"


def test_find_nearby_hospitals_prefers_emergency_tagged_over_untagged_same_specialty_class(monkeypatch):
    _mock_overpass_response(
        monkeypatch,
        [
            {"tags": {"name": "Sunrise Hospital"}, "lat": 31.5205, "lon": 74.3588},
            {"tags": {"name": "Riverside Hospital", "emergency": "yes"}, "lat": 31.530, "lon": 74.370},
        ],
    )
    results = find_nearby_hospitals(31.5204, 74.3587)
    assert results[0]["name"] == "Riverside Hospital"


def test_find_nearby_hospitals_still_returns_specialty_hospital_when_its_the_only_option(monkeypatch):
    _mock_overpass_response(
        monkeypatch,
        [{"tags": {"name": "Lions Eye Hospital"}, "lat": 31.521, "lon": 74.359}],
    )
    results = find_nearby_hospitals(31.5204, 74.3587)
    assert results[0]["name"] == "Lions Eye Hospital"


def test_find_nearby_hospitals_reads_way_center_tag(monkeypatch):
    _mock_overpass_response(
        monkeypatch,
        [{"tags": {"name": "Way Hospital"}, "center": {"lat": 31.521, "lon": 74.359}}],
    )
    results = find_nearby_hospitals(31.5204, 74.3587)
    assert results[0]["name"] == "Way Hospital"


def test_find_nearby_hospitals_skips_elements_missing_name_or_coords(monkeypatch):
    _mock_overpass_response(
        monkeypatch,
        [
            {"tags": {}, "lat": 31.521, "lon": 74.359},
            {"tags": {"name": "No Coords Hospital"}},
        ],
    )
    assert find_nearby_hospitals(31.5204, 74.3587) == []


def test_find_nearby_hospitals_caps_at_three(monkeypatch):
    elements = [{"tags": {"name": f"Hospital {i}"}, "lat": 31.52 + i * 0.001, "lon": 74.35} for i in range(5)]
    _mock_overpass_response(monkeypatch, elements)
    assert len(find_nearby_hospitals(31.5204, 74.3587)) == 3


def test_find_nearby_hospitals_falls_back_to_next_mirror_when_first_fails(monkeypatch):
    calls = []

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"elements": [{"tags": {"name": "Mirror 2 Hospital"}, "lat": 31.521, "lon": 74.359}]}

    def _fake_post(url, *args, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ConnectError("blocked")
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", _fake_post)
    results = find_nearby_hospitals(31.5204, 74.3587)
    assert len(calls) == 2
    assert results[0]["name"] == "Mirror 2 Hospital"


def test_find_nearby_hospitals_returns_empty_on_network_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "post", _raise)
    assert find_nearby_hospitals(31.5204, 74.3587) == []


def test_find_nearby_hospitals_returns_empty_on_http_status_error(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse())
    assert find_nearby_hospitals(31.5204, 74.3587) == []


def test_nearby_hospitals_block_empty_without_coordinates():
    assert nearby_hospitals_block(None, None) == ""


def test_nearby_hospitals_block_empty_when_lookup_finds_nothing(monkeypatch):
    _mock_overpass_response(monkeypatch, [])
    assert nearby_hospitals_block(31.5204, 74.3587) == ""


def test_nearby_hospitals_block_formats_results_with_phone(monkeypatch):
    _mock_overpass_response(
        monkeypatch,
        [{"tags": {"name": "Mayo Hospital", "phone": "+92-42-99211100"}, "lat": 31.521, "lon": 74.359}],
    )
    block = nearby_hospitals_block(31.5204, 74.3587)
    assert "Nearest emergency hospitals:" in block
    assert "Mayo Hospital" in block
    assert "+92-42-99211100" in block


def test_nearby_hospitals_block_falls_back_to_1122_when_no_phone_listed(monkeypatch):
    _mock_overpass_response(monkeypatch, [{"tags": {"name": "No Phone Hospital"}, "lat": 31.521, "lon": 74.359}])
    block = nearby_hospitals_block(31.5204, 74.3587)
    assert "phone not listed, call 1122" in block


def test_nearby_hospitals_block_never_raises_on_lookup_failure(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.TimeoutException("boom")))
    assert nearby_hospitals_block(31.5204, 74.3587) == ""
