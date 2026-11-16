import numpy as np
import pandas as pd
import pytest

from nba_predictor import ablation
from nba_predictor.train import FEATURES


def test_feature_sets_only_use_known_features_and_drop_what_they_say():
    sets = ablation.feature_sets()
    for name, cols in sets.items():
        assert set(cols) <= set(FEATURES), name
    assert "elo_logit" not in sets["everything minus elo"]
    assert not set(ablation.GROUPS["rest"]) & set(sets["everything minus rest"])
    assert sorted(sets["everything"]) == sorted(FEATURES)


def test_groups_cover_every_feature_once():
    grouped = [f for cols in ablation.GROUPS.values() for f in cols]
    assert sorted(grouped) == sorted(FEATURES)


def frame(n, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        {
            "elo_logit": rng.normal(0, 0.8, n),
            "home_b2b": rng.integers(0, 2, n),
            "away_b2b": rng.integers(0, 2, n),
        }
    )
    return x, rng


def test_the_adjustment_recovers_known_shifts():
    x, rng = frame(40_000)
    z = x["elo_logit"] - 0.1 - 0.3 * x["home_b2b"] + 0.4 * x["away_b2b"]
    y = (rng.uniform(size=len(x)) < 1 / (1 + np.exp(-z))).astype(int)
    coef = ablation.EloRestAdjustment().fit(x, y).coef_
    assert coef == pytest.approx([-0.1, -0.3, 0.4], abs=0.06)


def test_with_no_rest_effect_the_adjustment_stays_near_plain_elo():
    x, rng = frame(20_000, seed=1)
    y = (rng.uniform(size=len(x)) < 1 / (1 + np.exp(-x["elo_logit"]))).astype(int)
    m = ablation.EloRestAdjustment().fit(x, y)
    assert np.abs(m.coef_).max() < 0.08
    p = m.predict_proba(x)
    assert p.shape == (len(x), 2) and np.allclose(p.sum(axis=1), 1)
