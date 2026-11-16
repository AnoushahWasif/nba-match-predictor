import json

import pytest
import requests

from nba_predictor import collect
from nba_predictor.collect import CollectError

HEADERS = list(collect.REQUIRED)


def payload(season="2023-24", games=collect.MIN_GAMES):
    """Made-up games in the API's response shape: two rows per game, home team first."""
    rows = []
    for g in range(games):
        game_id, date = f"0022300{g:04d}", f"2023-11-{1 + g % 28:02d}"
        home, away = 1610612700 + g % 30, 1610612700 + (g + 1) % 30
        season_id = "2" + season[:4]
        rows.append([season_id, home, "HOM", game_id, date, "HOM vs. AWY", "W", 110])
        rows.append([season_id, away, "AWY", game_id, date, "AWY @ HOM", "L", 100])
    return {
        "resultSets": [
            {"name": "LeagueGameFinderResults", "headers": list(HEADERS), "rowSet": rows}
        ]
    }


def test_seasons_are_the_agreed_range():
    assert collect.SEASONS[0] == "2014-15" and collect.SEASONS[-1] == "2024-25"
    assert len(collect.SEASONS) == 11


def test_a_good_season_validates():
    df = collect.validate("2023-24", payload())
    assert len(df) == 2 * collect.MIN_GAMES and list(df.columns) == HEADERS


def first_row(p):
    return p["resultSets"][0]["rowSet"][0]


def broken(change):
    p = payload()
    change(p)
    return p


BAD = {
    "not a dict": lambda: "<html>blocked</html>",
    "no result sets": lambda: {"resultSets": []},
    "missing column": lambda: broken(lambda p: p["resultSets"][0]["headers"].remove("WL")),
    "ragged row": lambda: broken(lambda p: p["resultSets"][0]["rowSet"][3].pop()),
    "other season": lambda: broken(lambda p: first_row(p).__setitem__(0, "22022")),
    "playoff game": lambda: broken(lambda p: first_row(p).__setitem__(3, "0042300101")),
    "too few games": lambda: payload(games=collect.MIN_GAMES - 1),
    "too many games": lambda: payload(games=collect.MAX_GAMES + 1),
}


@pytest.mark.parametrize("name", BAD)
def test_bad_responses_are_rejected(name):
    with pytest.raises(CollectError):
        collect.validate("2023-24", BAD[name]())


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def sleep(self, s):
        self.slept.append(s)
        self.now += s

    def __call__(self):
        return self.now


def test_fetch_waits_between_requests():
    clock = Clock()
    fetch = collect.Fetcher(lambda s: payload(s), clock.sleep, clock)
    fetch("2023-24")
    fetch("2023-24")
    assert clock.slept == [collect.MIN_GAP_S]


def test_fetch_retries_transport_errors_then_succeeds():
    clock, calls = Clock(), []

    def flaky(season):
        calls.append(season)
        if len(calls) < 3:
            raise requests.Timeout("slow")
        return payload(season)

    collect.Fetcher(flaky, clock.sleep, clock)("2023-24")
    assert len(calls) == 3 and clock.slept[0] == 5


def test_fetch_gives_up_after_the_retries():
    clock = Clock()

    def blocked(season):
        raise ValueError("Expecting value: line 1 column 1")  # an HTML page instead of JSON

    with pytest.raises(CollectError, match="gave up after 3"):
        collect.Fetcher(blocked, clock.sleep, clock)("2023-24")


def test_a_bad_response_is_not_retried():
    calls = []

    def wrong(season):
        calls.append(1)
        return payload(games=5)

    with pytest.raises(CollectError):
        collect.Fetcher(wrong, Clock().sleep, Clock())("2023-24")
    assert len(calls) == 1


def test_save_and_load_round_trip(tmp_path):
    entry = collect.save("2023-24", payload(), tmp_path)
    assert entry["rows"] == 2 * collect.MIN_GAMES and len(entry["sha256"]) == 64
    assert len(collect.load("2023-24", tmp_path)) == 2 * collect.MIN_GAMES
    assert not list(tmp_path.glob("*.part"))


def test_a_changed_cache_file_is_refused(tmp_path):
    collect.save("2023-24", payload(), tmp_path)
    path = tmp_path / "2023-24.json"
    path.write_bytes(path.read_bytes().replace(b"110", b"111", 1))
    with pytest.raises(CollectError, match="manifest hash"):
        collect.load("2023-24", tmp_path)


def test_missing_cache_entries_and_files(tmp_path):
    with pytest.raises(CollectError, match="isn't cached"):
        collect.load("2023-24", tmp_path)
    collect.save("2023-24", payload(), tmp_path)
    (tmp_path / "2023-24.json").unlink()
    with pytest.raises(CollectError, match="file is missing"):
        collect.load("2023-24", tmp_path)


def test_oversized_responses_are_refused(monkeypatch):
    monkeypatch.setattr(collect, "MAX_BYTES", 100)
    with pytest.raises(CollectError, match="cap"):
        collect.encode(payload())


def test_changes_counts_added_removed_and_edited_rows():
    old = collect.validate("2023-24", payload(games=collect.MIN_GAMES + 1))
    p = payload()
    p["resultSets"][0]["rowSet"][0][7] = 999  # a revised score
    new = collect.validate("2023-24", p)
    assert collect.changes(old, new) == {"added": 0, "removed": 2, "changed": 1}


def test_collect_skips_cached_seasons_and_reports_changes_on_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "SEASONS", ("2022-23", "2023-24"))
    calls = []

    def fetch(season):
        calls.append(season)
        return payload(season)

    assert collect.collect(raw=tmp_path, fetch=fetch) == {
        "2022-23": "downloaded",
        "2023-24": "downloaded",
    }
    assert collect.collect(raw=tmp_path, fetch=fetch) == {"2022-23": "cached", "2023-24": "cached"}
    assert len(calls) == 2
    report = collect.collect(refresh=True, raw=tmp_path, fetch=fetch)
    assert report["2023-24"] == {"added": 0, "removed": 0, "changed": 0}
    manifest = json.loads((tmp_path / collect.MANIFEST).read_text())
    assert set(manifest) == {"2022-23", "2023-24"}


def test_missing_values_on_both_sides_are_not_a_change():
    old_p, new_p = payload(), payload()
    old_p["resultSets"][0]["rowSet"][0][7] = None
    new_p["resultSets"][0]["rowSet"][0][7] = float("nan")
    old, new = collect.validate("2023-24", old_p), collect.validate("2023-24", new_p)
    assert collect.changes(old, new)["changed"] == 0
