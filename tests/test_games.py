import pandas as pd
import pytest

from nba_predictor import games
from nba_predictor.collect import REQUIRED

COLS = [*REQUIRED, *[s for s in games.STATS if s != "PTS"]]


def row(team, matchup, wl, pts, game_id="0022300001", date="2023-11-01", team_id=None):
    r = {c: 1 for c in COLS}
    r.update(
        SEASON_ID="22023",
        TEAM_ID=team_id or hash(team) % 1000,
        TEAM_ABBREVIATION=team,
        GAME_ID=game_id,
        GAME_DATE=date,
        MATCHUP=matchup,
        WL=wl,
        PTS=pts,
    )
    return r


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_home_and_away_come_from_the_matchup():
    out, bad = games.to_games(
        frame(row("NYK", "NYK @ BOS", "L", 99), row("BOS", "BOS vs. NYK", "W", 110)), "2023-24"
    )
    assert bad == []
    g = out.iloc[0]
    assert (g["home"], g["away"], g["home_win"], g["home_pts"], g["away_pts"]) == (
        "BOS",
        "NYK",
        1,
        110,
        99,
    )
    assert not g["neutral"] and not g["bubble"]


def test_neutral_site_games_keep_the_designated_home_team():
    out, _ = games.to_games(
        frame(row("SAS", "SAS @ IND", "W", 140), row("IND", "SAS @ IND", "L", 110)), "2024-25"
    )
    g = out.iloc[0]
    assert (g["home"], g["away"], g["home_win"], g["neutral"]) == ("IND", "SAS", 0, True)


def test_bubble_games_are_flagged_by_date():
    pre = [
        row("AAA", "AAA vs. BBB", "W", 100, "0021900001", "2020-03-11"),
        row("BBB", "BBB @ AAA", "L", 90, "0021900001", "2020-03-11"),
    ]
    post = [
        row("AAA", "AAA vs. BBB", "W", 100, "0021900002", "2020-07-30"),
        row("BBB", "BBB @ AAA", "L", 90, "0021900002", "2020-07-30"),
    ]
    out, _ = games.to_games(frame(*pre, *post), "2019-20")
    assert out["bubble"].tolist() == [False, True]


BAD = {
    "one row": [row("BOS", "BOS vs. NYK", "W", 110)],
    "two home rows": [row("BOS", "BOS vs. NYK", "W", 110), row("NYK", "NYK vs. BOS", "L", 99)],
    "two dates": [
        row("BOS", "BOS vs. NYK", "W", 110),
        row("NYK", "NYK @ BOS", "L", 99, date="2023-11-02"),
    ],
    "two wins": [row("BOS", "BOS vs. NYK", "W", 110), row("NYK", "NYK @ BOS", "W", 99)],
    "winner scored less": [row("BOS", "BOS vs. NYK", "W", 90), row("NYK", "NYK @ BOS", "L", 99)],
    "garbled matchup": [row("BOS", "BOS v NYK", "W", 110), row("NYK", "NYK @ BOS", "L", 99)],
    "a team not in the matchup": [
        row("BOS", "NYK @ LAL", "W", 110),
        row("NYK", "NYK @ LAL", "L", 99),
    ],
}


@pytest.mark.parametrize("name", BAD)
def test_unreadable_games_are_dropped_with_a_reason(name):
    good = [
        row("AAA", "AAA vs. BBB", "W", 100, "0022300009"),
        row("BBB", "BBB @ AAA", "L", 90, "0022300009"),
    ]
    out, bad = games.to_games(frame(*BAD[name], *good), "2023-24")
    assert len(out) == 1 and len(bad) == 1 and bad[0]["game_id"] == "0022300001"
    assert bad[0]["reason"]


def test_too_many_bad_games_fail_the_build():
    rows = [row("BOS", "BOS vs. NYK", "W", 110, f"00223000{i:02d}") for i in range(4)]
    with pytest.raises(games.GamesError, match="4 games"):
        games.to_games(frame(*rows), "2023-24")


def test_output_is_in_date_order():
    late = [
        row("AAA", "AAA vs. BBB", "W", 100, "0022300001", "2023-12-01"),
        row("BBB", "BBB @ AAA", "L", 90, "0022300001", "2023-12-01"),
    ]
    early = [
        row("AAA", "AAA vs. BBB", "W", 100, "0022300002", "2023-11-01"),
        row("BBB", "BBB @ AAA", "L", 90, "0022300002", "2023-11-01"),
    ]
    out, _ = games.to_games(frame(*late, *early), "2023-24")
    assert out["game_id"].tolist() == ["0022300002", "0022300001"]


def test_a_season_with_no_readable_games_fails():
    with pytest.raises(games.GamesError, match="no readable"):
        games.to_games(frame(row("BOS", "BOS vs. NYK", "W", 110)), "2023-24")
