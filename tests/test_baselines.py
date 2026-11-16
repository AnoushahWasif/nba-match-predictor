import numpy as np
import pandas as pd
import pytest

from nba_predictor import baselines


def test_log5_basics():
    assert baselines.log5(0.5, 0.5) == 0.5
    assert baselines.log5(0.6, 0.4) == pytest.approx(0.6923, abs=1e-4)
    assert baselines.log5(0.6, 0.4) + baselines.log5(0.4, 0.6) == pytest.approx(1.0)


def test_shrinking_pulls_early_records_toward_500():
    assert baselines.shrunk(1.0, 1) == pytest.approx(6 / 11)  # .545
    assert baselines.shrunk(0.0, 0) == 0.5
    assert abs(baselines.shrunk(0.8, 70) - 0.8) < 0.04


def test_predictions_are_probabilities_with_the_train_home_rate():
    games = pd.DataFrame(
        {
            "home_win_pct": [0.5, 0.9, 0.1],
            "away_win_pct": [0.5, 0.1, 0.9],
            "home_games_played": [0, 40, 40],
            "away_games_played": [0, 40, 40],
            "elo_prob": [0.55, 0.8, 0.2],
        }
    )
    train = pd.DataFrame({"home_win": [1, 1, 0, 1]})
    preds = baselines.predictions(train, games)
    assert np.allclose(preds["always home"], 0.75)
    assert preds["better record (log5)"][0] == 0.5
    assert preds["better record (log5)"][1] > 0.8 > 0.2 > preds["better record (log5)"][2]
    for p in preds.values():
        assert ((p >= 0) & (p <= 1)).all()
