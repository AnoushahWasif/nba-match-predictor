import numpy as np
import pandas as pd

from nba_predictor import train


def table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame(
        {
            "elo_prob": rng.uniform(0.2, 0.8, n),
            **{c: rng.normal(size=n) for c in train.FEATURES[1:6]},
            "home_rest_days": rng.integers(1, 5, n),
            "away_rest_days": rng.integers(1, 5, n),
            "home_b2b": rng.integers(0, 2, n),
            "away_b2b": rng.integers(0, 2, n),
            "home_games_played": rng.integers(0, 80, n),
            "away_games_played": rng.integers(0, 80, n),
            "neutral": False,
            "bubble": rng.uniform(size=n) < 0.05,
        }
    )
    t["home_win"] = (rng.uniform(size=n) < t["elo_prob"]).astype(int)
    return t


def test_matrix_has_the_listed_features_and_no_gaps():
    x = train.matrix(table())
    assert list(x.columns) == train.FEATURES
    assert not x.isna().any().any()


def test_elo_enters_as_log_odds_and_extreme_probabilities_stay_finite():
    t = table(4)
    t["elo_prob"] = [0.5, 0.0, 1.0, 0.75]
    x = train.matrix(t)
    assert x["elo_logit"].iloc[0] == 0
    assert np.isfinite(x["elo_logit"]).all()
    assert x["elo_logit"].iloc[3] == np.log(3)


def test_no_home_covers_neutral_and_bubble():
    t = table(3)
    t["neutral"] = [True, False, False]
    t["bubble"] = [False, True, False]
    assert train.matrix(t)["no_home"].tolist() == [1, 1, 0]


def test_every_candidate_fits_and_gives_probabilities():
    t = table()
    x, y = train.matrix(t), t["home_win"]
    for name, model in train.candidates().items():
        p = model.fit(x, y).predict_proba(x)[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name
