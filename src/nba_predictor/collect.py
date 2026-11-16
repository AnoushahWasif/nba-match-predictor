"""Day 2: team game logs for each regular season from stats.nba.com, through nba_api.

    uv run python -m nba_predictor.collect             # downloads the seasons not cached yet
    uv run python -m nba_predictor.collect --refresh   # downloads again and reports what changed

Raw responses are cached in data/raw/ with a manifest of SHA-256 hashes and download dates
(decision 7). Nothing else in the project calls the API; everything reads the cache.
"""

import argparse
import hashlib
import hmac
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SEASONS = tuple(f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2025))
REQUIRED = (
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS",
)  # fmt: skip
# 2019-20 stopped at 1,059 games and 2020-21 had 1,080; a normal season has 1,230.
MIN_GAMES, MAX_GAMES = 1000, 1230
MIN_GAP_S, TIMEOUT_S, RETRIES = 1.0, 60, 3
MAX_BYTES = 20 * 1024 * 1024
MANIFEST = "manifest.json"


class CollectError(RuntimeError):
    pass


def data_dir() -> Path:
    return Path(os.environ.get("NBA_DATA_DIR", "data"))


def request_season(season: str) -> dict:
    from nba_api.stats.endpoints import leaguegamefinder

    return leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
        timeout=TIMEOUT_S,
    ).get_dict()


def validate(season: str, payload) -> pd.DataFrame:
    """The team-game rows of one season, or CollectError if the response isn't what we expect."""
    try:
        result = payload["resultSets"][0]
        headers, rows = result["headers"], result["rowSet"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CollectError(f"{season}: unexpected response shape") from exc
    missing = [c for c in REQUIRED if c not in headers]
    if missing:
        raise CollectError(f"{season}: missing columns {missing}")
    if any(not isinstance(r, list) or len(r) != len(headers) for r in rows):
        raise CollectError(f"{season}: a row doesn't match the header")
    df = pd.DataFrame(rows, columns=headers)
    # SEASON_ID is "2" (regular season) plus the starting year; GAME_ID "002..." is a regular
    # season game. Anything else means the filter didn't apply.
    if (df["SEASON_ID"].astype(str) != "2" + season[:4]).any():
        raise CollectError(f"{season}: rows from another season or season type")
    if not df["GAME_ID"].astype(str).str.startswith("002").all():
        raise CollectError(f"{season}: non-regular-season games")
    games = df["GAME_ID"].nunique()
    if not MIN_GAMES <= games <= MAX_GAMES:
        raise CollectError(f"{season}: {games} games, expected {MIN_GAMES} to {MAX_GAMES}")
    return df


class Fetcher:
    """Calls the API at most once per MIN_GAP_S and retries transport errors with backoff."""

    def __init__(self, request=request_season, sleep=time.sleep, clock=time.monotonic):
        self.request, self.sleep, self.clock = request, sleep, clock
        self.last = None

    def __call__(self, season: str) -> dict:
        import requests

        for attempt in range(RETRIES):
            if self.last is not None:
                self.sleep(max(0.0, MIN_GAP_S - (self.clock() - self.last)))
            self.last = self.clock()
            try:
                payload = self.request(season)
            # A blocked or throttled request often comes back as HTML, which fails JSON parsing.
            except (requests.RequestException, ValueError) as exc:
                if attempt == RETRIES - 1:
                    raise CollectError(f"{season}: gave up after {RETRIES} tries: {exc}") from exc
                self.sleep(5 * 2**attempt)
                continue
            validate(season, payload)
            return payload
        raise AssertionError("unreachable")


def encode(payload: dict) -> bytes:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if len(data) > MAX_BYTES:
        raise CollectError(f"response is {len(data) / 1e6:.0f} MB; the cap is {MAX_BYTES >> 20} MB")
    return data


def read_manifest(raw: Path) -> dict:
    path = raw / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {}


def save(season: str, payload: dict, raw: Path, now: datetime | None = None) -> dict:
    data = encode(payload)
    raw.mkdir(parents=True, exist_ok=True)
    tmp = raw / f"{season}.json.part"
    tmp.write_bytes(data)
    tmp.replace(raw / f"{season}.json")
    manifest = read_manifest(raw)
    entry = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(payload["resultSets"][0]["rowSet"]),
        "downloaded_utc": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
    }
    manifest[season] = entry
    (raw / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return entry


def load(season: str, raw: Path | None = None) -> pd.DataFrame:
    """One cached season, only if its bytes still match the manifest."""
    raw = raw or data_dir() / "raw"
    entry = read_manifest(raw).get(season)
    if entry is None:
        raise CollectError(f"{season} isn't cached; run `python -m nba_predictor.collect`")
    try:
        data = (raw / f"{season}.json").read_bytes()
    except OSError as exc:
        raise CollectError(f"{season}: in the manifest but the file is missing") from exc
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), entry["sha256"]):
        raise CollectError(f"{season}: cached file doesn't match its manifest hash")
    return validate(season, json.loads(data))


def _text(col: pd.Series) -> pd.Series:
    # Missing values (FT_PCT is null when a team took no free throws) can come back as None or
    # NaN, and str() of those differs, so they're mapped to one marker before comparing.
    return col.astype(str).where(col.notna(), "<missing>")


def changes(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Team-game rows added, removed or changed between two downloads of a season."""
    key = ["GAME_ID", "TEAM_ID"]
    keys_old = pd.MultiIndex.from_frame(old[key])
    keys_new = pd.MultiIndex.from_frame(new[key])
    # Inner join for the comparison: an outer join turns integer columns with gaps into floats,
    # and "110" vs "110.0" would count as a change.
    both = old.merge(new, on=key, suffixes=("_old", "_new"))
    cols = [c for c in old.columns if c in new.columns and c not in key]
    differs = pd.concat([_text(both[f"{c}_old"]) != _text(both[f"{c}_new"]) for c in cols], axis=1)
    differs = differs.any(axis=1)
    return {
        "added": len(keys_new.difference(keys_old)),
        "removed": len(keys_old.difference(keys_new)),
        "changed": int(differs.sum()),
    }


def collect(refresh: bool = False, raw: Path | None = None, fetch=None) -> dict:
    raw = raw or data_dir() / "raw"
    fetch = fetch or Fetcher()
    report = {}
    for season in SEASONS:
        cached = season in read_manifest(raw)
        if cached and not refresh:
            report[season] = "cached"
            continue
        payload = fetch(season)
        if cached:
            report[season] = changes(load(season, raw), validate(season, payload))
        else:
            report[season] = "downloaded"
        save(season, payload, raw)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="download cached seasons again")
    args = parser.parse_args(argv)
    for season, result in collect(args.refresh).items():
        print(season, result)


if __name__ == "__main__":
    main()
