import pandas as pd
import pytest

from nba_predictor import elo
from nba_predictor.elo import EloParams


def games(rows, season="2023-24"):
    """rows: (home_id, away_id, home_pts, away_pts[, season])"""
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "game_id": f"g{i:03d}",
                "season": r[4] if len(r) > 4 else season,
                "date": pd.Timestamp("2023-10-24") + pd.Timedelta(days=i),
                "home_id": r[0],
                "away_id": r[1],
                "home_pts": r[2],
                "away_pts": r[3],
                "home_win": int(r[2] > r[3]),
            }
        )
    return pd.DataFrame(out)


def test_expected_score():
    assert elo.expected(0) == 0.5
    assert elo.expected(400) == pytest.approx(10 / 11)
    assert elo.expected(-400) == pytest.approx(1 / 11)


def test_first_game_uses_starting_ratings_and_the_home_bonus():
    out = elo.ratings(games([(1, 2, 100, 90)]), EloParams(k=20, home=100, mov=False))
    assert out.iloc[0]["home_elo"] == out.iloc[0]["away_elo"] == 1500
    assert out.iloc[0]["elo_prob"] == pytest.approx(elo.expected(100))


def test_winner_gains_what_the_loser_loses():
    out = elo.ratings(games([(1, 2, 100, 90), (1, 2, 100, 90)]), EloParams(k=20, home=0, mov=False))
    second = out.iloc[1]
    assert second["home_elo"] == pytest.approx(1510)
    assert second["away_elo"] == pytest.approx(1490)


def test_bigger_wins_move_ratings_more_with_mov():
    params = EloParams(k=20, home=0, mov=True)
    close = elo.ratings(games([(1, 2, 101, 100), (1, 2, 1, 0)]), params).iloc[1]["home_elo"]
    blowout = elo.ratings(games([(1, 2, 130, 100), (1, 2, 1, 0)]), params).iloc[1]["home_elo"]
    assert blowout > close > 1500


@pytest.mark.parametrize("flag", ["neutral", "bubble"])
def test_no_home_bonus_without_a_real_home(flag):
    g = games([(1, 2, 100, 90)])
    g[flag] = True
    assert elo.ratings(g, EloParams(home=100)).iloc[0]["elo_prob"] == 0.5


def test_ratings_are_pulled_back_between_seasons():
    g = games([(1, 2, 100, 90, "2022-23"), (1, 2, 100, 90, "2023-24")])
    first = elo.ratings(g.iloc[:1].assign(season="2022-23"), EloParams(k=20, home=0, mov=False))
    out = elo.ratings(g, EloParams(k=20, home=0, carry=0.5, mov=False))
    assert first.iloc[0]["home_elo"] == 1500
    assert out.iloc[1]["home_elo"] == pytest.approx(1505)  # 1510 pulled halfway back


def test_output_follows_the_input_row_order():
    g = games([(1, 2, 100, 90), (2, 1, 100, 90), (1, 2, 100, 90)])
    shuffled = g.iloc[[2, 0, 1]]
    a = elo.ratings(g).loc[shuffled.index]
    b = elo.ratings(shuffled)
    pd.testing.assert_frame_equal(a, b)


def test_tuning_never_sees_validation_or_test_seasons():
    rows = [
        (i % 4, (i + 1) % 4, 100 + i % 7, 99, s) for s in ("2018-19", "2019-20") for i in range(40)
    ]
    base = games(rows)
    later = games([(1, 2, 0, 200, "2020-21")] * 30)
    later["date"] = later["date"] + pd.Timedelta(days=800)
    grid = {"k": (10, 20), "home": (0, 50), "carry": (0.5, 1.0), "mov": (False,)}
    with_later = pd.concat([base, later], ignore_index=True)
    a = elo.tune(base, grid)[0]
    b = elo.tune(with_later, grid)[0]
    assert a == b
