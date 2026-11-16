import numpy as np
import pandas as pd
import pytest

from nba_predictor.evaluate import (
    bootstrap,
    log_loss,
    metrics,
    paired_log_loss_difference,
    split,
    wilson,
)


def test_wilson_matches_a_known_value():
    # 707 of 1,230 home wins (2014-15): 54.7% to 60.2%
    low, high = wilson(707, 1230)
    assert (round(low, 3), round(high, 3)) == (0.547, 0.602)


def test_wilson_stays_inside_zero_and_one():
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0
    assert 0 < wilson(0, 5)[1] < 1


@pytest.mark.parametrize("k, n", [(1, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


def table(seasons):
    return pd.DataFrame({"season": seasons, "home_win": [1] * len(seasons)})


def test_split_by_season():
    train, val, test = split(table(["2014-15", "2020-21", "2021-22", "2024-25"]))
    assert list(train["season"]) == ["2014-15"]
    assert list(val["season"]) == ["2020-21", "2021-22"]
    assert list(test["season"]) == ["2024-25"]


def test_split_refuses_unknown_seasons():
    with pytest.raises(ValueError, match="2025-26"):
        split(table(["2014-15", "2025-26"]))


def test_metrics_on_known_values():
    m = metrics([1, 0], [0.8, 0.4])
    assert m["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)
    assert m["brier"] == pytest.approx((0.04 + 0.16) / 2)
    assert m["accuracy"] == 1.0 and m["auc"] == 1.0


@pytest.mark.parametrize("p", [[], [1.2], [-0.1]])
def test_metrics_reject_bad_probabilities(p):
    with pytest.raises(ValueError):
        metrics([1] * len(p), p)


def test_log_loss_is_clipped():
    assert log_loss([1, 0], [1.0, 0.0]) < 1e-6
    assert log_loss([1], [0.0]) == pytest.approx(27.63, abs=0.01)


def test_bootstrap_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0.2, 0.6, 500), 0, 1)
    point = metrics(y, p)
    ci = bootstrap(y, p, n=500)
    assert ci["log_loss_95"][0] <= point["log_loss"] <= ci["log_loss_95"][1]
    assert ci["accuracy_95"][0] <= point["accuracy"] <= ci["accuracy_95"][1]


def test_paired_difference_signs():
    y = np.array([1, 0] * 100)
    good, bad = np.where(y == 1, 0.7, 0.3), np.full(200, 0.5)
    low, high = paired_log_loss_difference(y, good, bad, n=300)
    assert high < 0
    assert paired_log_loss_difference(y, good, good, n=50) == [0.0, 0.0]
