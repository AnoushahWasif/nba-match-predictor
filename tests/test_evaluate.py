import pytest

from nba_predictor.evaluate import wilson


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
