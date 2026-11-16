import numpy as np
import pandas as pd
import pytest

from nba_predictor import features

STATS = ["fga", "oreb", "tov", "fta", "fgm", "fg3m", "fg3a", "ftm", "dreb", "ast", "stl"]


def schedule(n_days=40, teams=6, seed=0, season="2023-24"):
    """Made-up round robin: every team plays once a day, scores random."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        order = rng.permutation(teams)
        for i in range(0, teams, 2):
            home, away = int(order[i]), int(order[i + 1])
            hp, ap = (int(x) for x in rng.integers(90, 130, 2))
            if hp == ap:
                hp += 1
            row = {
                "game_id": f"g{day:03d}{i}",
                "season": season,
                "date": pd.Timestamp("2023-10-24") + pd.Timedelta(days=day),
                "home_id": home,
                "away_id": away,
                "home_pts": hp,
                "away_pts": ap,
                "home_win": int(hp > ap),
            }
            for side in ("home", "away"):
                for s in STATS:
                    row[f"{side}_{s}"] = int(rng.integers(5, 90))
            rows.append(row)
    return pd.DataFrame(rows)


FEATURE_COLS = [
    *(f"{side}_{c}" for side in ("home", "away") for c in features.TEAM_FEATURES),
    "home_elo",
    "away_elo",
    "elo_prob",
]


def test_deleting_a_game_and_everything_after_changes_no_earlier_feature():
    games = schedule()
    full = features.game_features(games).set_index("game_id")[FEATURE_COLS]
    cut = games["date"].sort_values().iloc[len(games) // 2]
    early = games[games["date"] < cut]
    rebuilt = features.game_features(early).set_index("game_id")[FEATURE_COLS]
    pd.testing.assert_frame_equal(rebuilt, full.loc[rebuilt.index])


def test_changing_a_result_changes_nothing_up_to_that_game():
    games = schedule()
    k = len(games) // 2
    changed = games.copy()
    changed.loc[k, ["home_pts", "away_pts", "home_win"]] = [150, 80, 1]
    a = features.game_features(games).set_index("game_id")[FEATURE_COLS]
    b = features.game_features(changed).set_index("game_id")[FEATURE_COLS]
    on_or_before = games.loc[games["date"] <= games.loc[k, "date"], "game_id"]
    pd.testing.assert_frame_equal(a.loc[on_or_before], b.loc[on_or_before])
    assert not a.equals(b)  # and it does change later games, so the test can fail


def test_a_hand_worked_team():
    games = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": "2023-24",
            "date": pd.to_datetime(["2023-10-24", "2023-10-26", "2023-10-28"]),
            "home_id": [1, 2, 1],
            "away_id": [2, 1, 2],
            "home_pts": [100, 120, 105],
            "away_pts": [90, 110, 100],
            "home_win": [1, 1, 1],
        }
    )
    for side in ("home", "away"):
        for s in STATS:
            games[f"{side}_{s}"] = 10
    out = features.game_features(games, window=2).set_index("game_id")
    # Team 1 before game c: won a (+10), lost b (-10).
    assert out.loc["c", "home_games_played"] == 2
    assert out.loc["c", "home_win_pct"] == 0.5
    assert out.loc["c", "home_margin_avg"] == 0.0
    assert out.loc["c", "home_form_margin"] == 0.0
    # Team 2 before game c: lost a (-10), won b (+10): the same, so every difference is 0.
    assert out.loc["c", "diff_win_pct"] == 0.0
    # Opening night: placeholders, with league rating 100 because no earlier date exists.
    assert out.loc["a", "home_games_played"] == 0
    assert out.loc["a", "home_win_pct"] == 0.5 and out.loc["a", "home_off_rtg"] == 100.0


def test_ratings_use_points_per_possession():
    games = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": "2023-24",
            "date": pd.to_datetime(["2023-10-24", "2023-10-25"]),
            "home_id": [1, 1],
            "away_id": [2, 2],
            "home_pts": [110, 100],
            "away_pts": [100, 90],
            "home_win": [1, 1],
        }
    )
    for side in ("home", "away"):
        for s in STATS:
            games[f"{side}_{s}"] = 0
        games[f"{side}_fga"] = 100  # 100 possessions each
    out = features.game_features(games).set_index("game_id")
    assert out.loc["b", "home_off_rtg"] == pytest.approx(110.0)
    assert out.loc["b", "home_def_rtg"] == pytest.approx(100.0)
    # Team 2 scored 100 on 100 possessions in game a.
    assert out.loc["b", "away_off_rtg"] == pytest.approx(100.0)


def test_a_team_without_games_gets_the_league_average_of_earlier_dates():
    games = pd.DataFrame(
        {
            "game_id": ["a", "b"],
            "season": "2023-24",
            "date": pd.to_datetime(["2023-10-24", "2023-10-25"]),
            "home_id": [1, 3],
            "away_id": [2, 1],
            "home_pts": [120, 100],
            "away_pts": [100, 90],
            "home_win": [1, 1],
        }
    )
    for side in ("home", "away"):
        for s in STATS:
            games[f"{side}_{s}"] = 0
        games[f"{side}_fga"] = 100
    out = features.game_features(games).set_index("game_id")
    # Team 3's first game is b. Day 1 had 220 points on 200 possessions: 110 per 100.
    assert out.loc["b", "home_games_played"] == 0
    assert out.loc["b", "home_off_rtg"] == pytest.approx(110.0)
    assert out.loc["b", "home_def_rtg"] == pytest.approx(110.0)


def test_form_window_only_counts_the_last_games():
    games = schedule(n_days=30, teams=2)
    long = features.add_form(features.team_games(games), window=3)
    team = long[long["team_id"] == 0].reset_index(drop=True)
    expected = team["margin"].shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
    pd.testing.assert_series_equal(team["form_margin"], expected, check_names=False)


def test_a_new_season_starts_from_scratch():
    a, b = schedule(n_days=5, season="2022-23"), schedule(n_days=5, season="2023-24", seed=1)
    b["date"] = b["date"] + pd.Timedelta(days=365)
    b["game_id"] = "n" + b["game_id"]
    out = features.game_features(pd.concat([a, b], ignore_index=True))
    first = out[out["season"] == "2023-24"].iloc[0]
    assert first["home_games_played"] == 0 and first["home_win_pct"] == 0.5


def test_two_games_for_one_team_on_one_date_are_refused():
    games = schedule(n_days=2)
    games.loc[1, "date"] = games.loc[0, "date"]
    games.loc[1, "home_id"] = games.loc[0, "home_id"]
    with pytest.raises(ValueError, match="twice on one date"):
        features.team_games(games)


def test_no_missing_values_in_any_feature():
    out = features.game_features(schedule())
    cols = [c for c in out.columns if c.startswith(("home_", "away_", "diff_"))]
    assert not out[cols].isna().any().any()


def test_rest_days_and_back_to_backs():
    games = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": "2023-24",
            "date": pd.to_datetime(["2023-10-24", "2023-10-25", "2023-10-28", "2023-11-10"]),
            "home_id": [1, 1, 1, 1],
            "away_id": [2, 3, 2, 3],
            "home_pts": [100] * 4,
            "away_pts": [90] * 4,
            "home_win": [1] * 4,
        }
    )
    for side in ("home", "away"):
        for s in STATS:
            games[f"{side}_{s}"] = 10
    out = features.game_features(games).set_index("game_id")
    assert out["home_rest_days"].tolist() == [features.REST_CAP, 1, 3, features.REST_CAP]
    assert out["home_b2b"].tolist() == [0, 1, 0, 0]
    # Team 3's first game is b, so it's fully rested there; d is 16 days later, capped.
    assert out.loc["b", "away_rest_days"] == features.REST_CAP
    assert out.loc["b", "diff_rest_days"] == 1 - features.REST_CAP
