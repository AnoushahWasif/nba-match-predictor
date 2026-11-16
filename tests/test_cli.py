import json

import pandas as pd
import pytest

from nba_predictor import cli, elo, features, holdout
from nba_predictor.elo import EloParams

ABBR = {1: "BOS", 2: "NYK", 3: "LAL", 4: "MIA"}
STATS = ["fga", "oreb", "tov", "fta", "fgm", "fg3m", "fg3a", "ftm", "dreb", "ast", "stl"]


def games(days=12, start="2023-10-24", season="2023-24"):
    """Two games a day among four teams, the home side always winning by 5."""
    rows = []
    for d in range(days):
        date = pd.Timestamp(start) + pd.Timedelta(days=2 * d)
        pairs = [(1, 2), (3, 4)] if d % 2 == 0 else [(4, 1), (2, 3)]
        for i, (h, a) in enumerate(pairs):
            rows.append(
                {
                    "game_id": f"{season[:4]}{d:03d}{i}",
                    "season": season,
                    "date": date,
                    "home_id": h,
                    "away_id": a,
                    "home": ABBR[h],
                    "away": ABBR[a],
                    "home_pts": 105,
                    "away_pts": 100,
                    "home_win": 1,
                    "neutral": False,
                    "bubble": False,
                    **{f"{s}_{x}": 10 for s in ("home", "away") for x in STATS},
                }
            )
    return pd.DataFrame(rows)


PARAMS = EloParams(k=20, home=60, carry=0.8, mov=False)


@pytest.fixture
def cached(tmp_path, monkeypatch):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    games().to_parquet(processed / "games.parquet")
    report = tmp_path / "elo.json"
    report.write_text(json.dumps({"params": PARAMS.__dict__}))
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli, "ELO_REPORT", report)
    monkeypatch.setattr(cli, "HOLDOUT_REPORT", tmp_path / "holdout.json")
    return tmp_path


def test_prediction_matches_elo_from_earlier_games_only():
    g = games()
    day = pd.Timestamp("2023-11-10")
    out = cli.predict(g, PARAMS, "bos", "NYK", day)
    rating = elo.current(g, day, "2023-24", PARAMS)
    assert out["home_win_probability"] == pytest.approx(
        elo.expected(rating[1] + 60 - rating[2]), abs=1e-4
    )
    assert out["games_up_to"] < "2023-11-10"


def test_neutral_court_drops_the_home_bonus():
    g, day = games(), pd.Timestamp("2023-10-26")
    home = cli.predict(g, PARAMS, "LAL", "MIA", day)["home_win_probability"]
    neutral = cli.predict(g, PARAMS, "LAL", "MIA", day, neutral=True)["home_win_probability"]
    assert home > neutral


def test_a_new_season_pulls_ratings_toward_the_mean():
    g = games()
    later = cli.predict(g, PARAMS, "BOS", "MIA", pd.Timestamp("2024-10-25"))
    same_season = cli.predict(g, PARAMS, "BOS", "MIA", pd.Timestamp("2023-12-20"))
    assert abs(later["home_elo"] - 1500) < abs(same_season["home_elo"] - 1500)


def test_back_to_backs_are_reported():
    g = games()
    day = pd.Timestamp("2023-10-25")  # the day after BOS-NYK and LAL-MIA
    assert set(cli.predict(g, PARAMS, "BOS", "LAL", day)["back_to_back"]) == {"BOS", "LAL"}
    assert cli.predict(g, PARAMS, "BOS", "LAL", day + pd.Timedelta(days=1))["back_to_back"] == []


@pytest.mark.parametrize(
    "home, away, day, message",
    [
        ("XXX", "NYK", "2023-11-10", "unknown team"),
        ("BOS", "BOS", "2023-11-10", "can't play itself"),
        ("BOS", "NYK", "2023-10-24", "no games before"),
        ("BOS", "NYK", "2010-01-01", "no games before"),
    ],
)
def test_bad_requests_are_refused(home, away, day, message):
    with pytest.raises(cli.UsageError, match=message):
        cli.predict(games(), PARAMS, home, away, pd.Timestamp(day))


@pytest.mark.parametrize("text", ["11/10/2023", "2023-13-01", "yesterday"])
def test_bad_dates_are_refused(text):
    with pytest.raises(cli.UsageError, match="--date"):
        cli.parse_date(text)


def test_season_of_a_date():
    assert cli.season_of(pd.Timestamp("2024-03-01")) == "2023-24"
    assert cli.season_of(pd.Timestamp("2024-10-22")) == "2024-25"
    assert cli.season_of(pd.Timestamp("1999-12-01")) == "1999-00"


def test_cli_prints_text_and_json(cached, capsys):
    assert cli.main(["predict", "--home", "BOS", "--away", "NYK", "--date", "2023-11-10"]) == 0
    assert "BOS win probability" in capsys.readouterr().out
    args = ["predict", "--home", "BOS", "--away", "NYK", "--date", "2023-11-10", "--json"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["home"] == "BOS"


def test_cli_errors_are_messages_not_tracebacks(cached, capsys, monkeypatch):
    assert cli.main(["predict", "--home", "XXX", "--away", "NYK", "--date", "2023-11-10"]) == 1
    assert "error: unknown team" in capsys.readouterr().err
    cli.ELO_REPORT.write_text("{broken")
    assert cli.main(["predict", "--home", "BOS", "--away", "NYK"]) == 1
    assert "make elo" in capsys.readouterr().err
    monkeypatch.setenv("NBA_DATA_DIR", str(cached / "nowhere"))
    assert cli.main(["predict", "--home", "BOS", "--away", "NYK"]) == 1
    assert "make data games" in capsys.readouterr().err


def feature_table():
    seasons = ["2018-19", "2022-23"]
    parts = [
        games(days=40, start=f"{s[:4]}-10-20", season=s).assign(
            home_win=lambda d, i=i: (d.index % (3 + i) != 0).astype(int)
        )
        for i, s in enumerate(seasons)
    ]
    g = pd.concat(parts, ignore_index=True)
    g["home_pts"] = g["home_pts"].where(g["home_win"] == 1, 95)
    return features.game_features(g, elo_params=PARAMS)


def test_holdout_scores_once(tmp_path):
    out = tmp_path / "holdout.json"
    report = holdout.main(out, table=feature_table())
    assert report["games"] == 80 and report["seasons"] == ["2022-23"]
    assert "plain Elo (CLI)" in json.loads(out.read_text())
    assert not list(tmp_path.glob("*.part"))
    with pytest.raises(RuntimeError, match="scored already"):
        holdout.main(out, table=feature_table())


def test_holdout_refusal_is_an_error_through_the_cli(cached, capsys):
    cli.HOLDOUT_REPORT.write_text("{}")
    assert cli.main(["holdout"]) == 1
    assert "scored already" in capsys.readouterr().err


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "predict" in capsys.readouterr().out


def test_dates_past_the_cache_say_what_is_unknown():
    out = cli.predict(games(), PARAMS, "BOS", "NYK", pd.Timestamp("2024-11-01"))
    assert out["back_to_back"] is None and out["season_cached"] is False
    text = cli.describe(out)
    assert "2024-25 isn't in the cache" in text and "back-to-backs are unknown" in text
    inside = cli.predict(games(), PARAMS, "BOS", "NYK", pd.Timestamp("2023-11-10"))
    assert inside["season_cached"] and inside["back_to_back"] is not None
    assert "isn't in the cache" not in cli.describe(inside)
