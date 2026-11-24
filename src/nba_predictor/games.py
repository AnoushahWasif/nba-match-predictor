"""Day 3: one row per game from the API's two team rows.

    uv run python -m nba_predictor.games

Writes data/processed/games.parquet and data/processed/report.json.
"""

import json
import re
from pathlib import Path

import pandas as pd

from nba_predictor import collect

# "BOS vs. NYK" is Boston's row of a game at Boston; "NYK @ BOS" is New York's row of the same game.
# Neutral-site games (Paris, Mexico City, the NBA Cup semifinals in Las Vegas) write both rows as
# "away @ home", with the designated home team after the "@".
MATCHUP = re.compile(r"^([A-Z]{3}) (vs\.|@) ([A-Z]{3})$")
STATS = ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "TOV"]
# The 2019-20 season restarted in the Orlando bubble on 30 July 2020: no crowds, no travel.
BUBBLE_START = pd.Timestamp("2020-07-30")
MAX_BAD_PER_SEASON = 3


class GamesError(ValueError):
    pass


def parse_matchup(matchup: str, team: str) -> tuple[str, str]:
    """(away, home) abbreviations from one team's MATCHUP text."""
    m = MATCHUP.match(str(matchup))
    if not m:
        raise GamesError(f"unreadable MATCHUP {matchup!r}")
    first, sep, second = m.groups()
    # Normally the row's own team comes first, but on a neutral site the home team's row also
    # reads "away @ home", so the team only has to be one of the two.
    if team not in (first, second):
        raise GamesError(f"MATCHUP {matchup!r} on {team}'s row")
    return (second, first) if sep == "vs." else (first, second)


def check_game(rows: pd.DataFrame) -> str | None:
    """Why a game's two rows can't be one game, or None if they can."""
    if len(rows) != 2:
        return f"{len(rows)} rows"
    try:
        pairs = {
            parse_matchup(m, t)
            for m, t in zip(rows["MATCHUP"], rows["TEAM_ABBREVIATION"], strict=True)
        }
    except GamesError as exc:
        return str(exc)
    if len(pairs) != 1:
        return "the two rows disagree on who is home"
    if set(rows["TEAM_ABBREVIATION"]) != set(next(iter(pairs))):
        return "the teams don't match the MATCHUP"
    if rows["GAME_DATE"].nunique() != 1:
        return "two dates"
    if sorted(rows["WL"]) != ["L", "W"]:
        return f"results {sorted(rows['WL'])}"
    winner = rows.loc[rows["WL"] == "W", "PTS"].iloc[0]
    if winner <= rows.loc[rows["WL"] == "L", "PTS"].iloc[0]:
        return "the winner didn't score more"
    return None


def to_games(teams: pd.DataFrame, season: str) -> tuple[pd.DataFrame, list[dict]]:
    """One row per game and the list of games dropped with the reason. More than
    MAX_BAD_PER_SEASON dropped games means the data has changed shape, so it fails instead.

    Vectorised: every game is paired by a merge, and check_game (the readable, row-by-row
    version of the same rules) only runs on the few that fail, to say why."""
    parts = teams["MATCHUP"].astype(str).str.extract(MATCHUP)
    vs = parts[1] == "vs."
    t = teams.assign(_away=parts[2].where(vs, parts[0]), _home=parts[0].where(vs, parts[2]))
    t["_pair"] = t["_away"] + "|" + t["_home"]
    h = t[t["TEAM_ABBREVIATION"] == t["_home"]]
    a = t[t["TEAM_ABBREVIATION"] == t["_away"]]
    m = h.merge(a, on="GAME_ID", suffixes=("_h", "_a"))
    winner_pts = m["PTS_h"].where(m["WL_h"] == "W", m["PTS_a"])
    loser_pts = m["PTS_a"].where(m["WL_h"] == "W", m["PTS_h"])
    ok = (
        (m["_pair_h"] == m["_pair_a"])
        & (m["GAME_DATE_h"] == m["GAME_DATE_a"])
        & (m["WL_h"].isin(["W", "L"]))
        & (m["WL_a"].isin(["W", "L"]))
        & (m["WL_h"] != m["WL_a"])
        & (winner_pts > loser_pts)
    )
    sizes = t.groupby("GAME_ID").size()
    good_ids = set(m.loc[ok, "GAME_ID"]) & set(sizes[sizes == 2].index)
    bad = [
        {"game_id": game_id, "reason": check_game(rows) or "rows didn't pair up"}
        for game_id, rows in teams[~teams["GAME_ID"].isin(good_ids)].groupby("GAME_ID")
    ]
    if len(bad) > MAX_BAD_PER_SEASON:
        raise GamesError(f"{season}: {len(bad)} games couldn't be read, e.g. {bad[:3]}")
    m = m[m["GAME_ID"].isin(good_ids)]
    if m.empty:
        raise GamesError(f"{season}: no readable games")
    out = pd.DataFrame(
        {
            "game_id": m["GAME_ID"],
            "season": season,
            "date": pd.to_datetime(m["GAME_DATE_h"]),
            "home_id": m["TEAM_ID_h"].astype(int),
            "away_id": m["TEAM_ID_a"].astype(int),
            "home": m["_home_h"],
            "away": m["_away_h"],
            "neutral": ~(m["MATCHUP_h"].str.contains(" vs. ")),
            "home_win": (m["WL_h"] == "W").astype(int),
        }
    )
    for stat in STATS:
        out[f"home_{stat.lower()}"] = m[f"{stat}_h"].to_numpy()
        out[f"away_{stat.lower()}"] = m[f"{stat}_a"].to_numpy()
    out["bubble"] = (out["season"] == "2019-20") & (out["date"] >= BUBBLE_START)
    return out.sort_values(["date", "game_id"]).reset_index(drop=True), bad


def build(raw: Path | None = None, out: Path | None = None) -> dict:
    out = out or collect.data_dir() / "processed"
    frames, report = [], {"seasons": {}}
    for season in collect.SEASONS:
        games, bad = to_games(collect.load(season, raw), season)
        frames.append(games)
        report["seasons"][season] = {
            "games": len(games),
            "dropped": bad,
            "neutral_site": int(games["neutral"].sum()),
            "bubble": int(games["bubble"].sum()),
            "home_win_rate": round(float(games["home_win"].mean()), 4),
        }
    games = pd.concat(frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out / "games.parquet", index=False)
    report["games"] = len(games)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
