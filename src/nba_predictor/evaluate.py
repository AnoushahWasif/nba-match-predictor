"""Shared evaluation helpers: the season split, metrics and bootstrap intervals."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion. Better than p ± 1.96·se for small n or p near 0 or 1,
    which matters for the 88 bubble games and the 5 neutral-site ones."""
    if n <= 0:
        raise ValueError("need at least one trial")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


TRAIN = ("2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20")
VALIDATION = ("2020-21", "2021-22")
TEST = ("2022-23", "2023-24", "2024-25")
N_BOOT = 2000


def split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, validation and test by season (decision 2). Any other season is an error, so a
    new download can't slip rows into the wrong split."""
    unknown = set(table["season"]) - set(TRAIN + VALIDATION + TEST)
    if unknown:
        raise ValueError(f"seasons outside the split: {sorted(unknown)}")
    return tuple(table[table["season"].isin(s)] for s in (TRAIN, VALIDATION, TEST))


def log_loss(y, p, eps: float = 1e-12) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    if len(y) == 0 or ((p < 0) | (p > 1)).any():
        raise ValueError("need probabilities in [0, 1] for at least one game")
    return {
        "log_loss": log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == y)),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        "mean_predicted": float(p.mean()),
        "home_win_rate": float(np.mean(y)),
    }


def bootstrap(y, p, n: int = N_BOOT, seed: int = 0) -> dict:
    """95% intervals for log loss, Brier and accuracy, resampling games."""
    y, p = np.asarray(y), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    ys, ps = y[idx], np.clip(p[idx], 1e-12, 1 - 1e-12)
    ll = -np.mean(ys * np.log(ps) + (1 - ys) * np.log(1 - ps), axis=1)
    brier = np.mean((ps - ys) ** 2, axis=1)
    acc = np.mean((ps > 0.5) == ys, axis=1)
    return {
        f"{name}_95": [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]
        for name, vals in (("log_loss", ll), ("brier", brier), ("accuracy", acc))
    }


def paired_log_loss_difference(y, p_a, p_b, n: int = N_BOOT, seed: int = 0) -> list[float]:
    """95% interval for log loss of a minus log loss of b on the same resampled games.
    Below zero means a is better."""
    y = np.asarray(y, float)
    a = np.clip(np.asarray(p_a, float), 1e-12, 1 - 1e-12)
    b = np.clip(np.asarray(p_b, float), 1e-12, 1 - 1e-12)
    per_game = -(y * np.log(a) + (1 - y) * np.log(1 - a)) + (
        y * np.log(b) + (1 - y) * np.log(1 - b)
    )
    idx = np.random.default_rng(seed).integers(0, len(y), (n, len(y)))
    return [round(float(v), 4) for v in np.percentile(per_game[idx].mean(axis=1), [2.5, 97.5])]


def per_season(table: pd.DataFrame, prob: str, n: int = 500) -> pd.DataFrame:
    """Log loss with a bootstrap interval, accuracy, and mean predicted vs actual home win rate
    for each season. Seasons with a single outcome class can't have an AUC, so it's left out."""
    rows = []
    for season, g in table.groupby("season", sort=True):
        m = metrics(g["home_win"], g[prob])
        low, high = bootstrap(g["home_win"], g[prob], n=n)["log_loss_95"]
        rows.append(
            {
                "season": season,
                "games": len(g),
                "log_loss": m["log_loss"],
                "low": low,
                "high": high,
                "accuracy": m["accuracy"],
                "predicted": m["mean_predicted"],
                "won": m["home_win_rate"],
            }
        )
    return pd.DataFrame(rows).set_index("season")
