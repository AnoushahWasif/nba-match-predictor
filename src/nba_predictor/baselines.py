"""Day 6: probability baselines on validation. Writes reports/baselines.json.

    uv run python -m nba_predictor.baselines

- always home: every game gets the training seasons' home win rate.
- better record: log5 on each team's win % so far, shrunk toward .500 early in the season,
  with no home advantage. It's what "pick the better team" means as a probability.
- Elo: the day 5 rating's own probability.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor import collect
from nba_predictor.evaluate import bootstrap, metrics, paired_log_loss_difference, split

# Games of .500 added to each record, so 1-0 after one game reads as .545, not 1.000.
PRIOR_GAMES = 10


def shrunk(win_pct, games_played, prior: int = PRIOR_GAMES):
    return (np.asarray(win_pct) * games_played + 0.5 * prior) / (np.asarray(games_played) + prior)


def log5(a, b):
    """Chance that a team with record quality a beats one with b (Bill James)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a * (1 - b) / (a * (1 - b) + b * (1 - a))


def predictions(train: pd.DataFrame, games: pd.DataFrame) -> dict[str, np.ndarray]:
    home = shrunk(games["home_win_pct"], games["home_games_played"])
    away = shrunk(games["away_win_pct"], games["away_games_played"])
    return {
        "always home": np.full(len(games), train["home_win"].mean()),
        "better record (log5)": log5(home, away),
        "Elo": games["elo_prob"].to_numpy(),
    }


def main(out: Path = Path("reports/baselines.json")) -> dict:
    table = pd.read_parquet(collect.data_dir() / "processed" / "features.parquet")
    train, validation, _ = split(table)
    y = validation["home_win"].to_numpy()
    preds = predictions(train, validation)
    report = {"split": "validation", "games": len(validation), "baselines": {}}
    for name, p in preds.items():
        by_season = {
            season: round(metrics(y[m], p[m])["log_loss"], 4)
            for season in validation["season"].unique()
            for m in [(validation["season"] == season).to_numpy()]
        }
        report["baselines"][name] = {
            **{k: round(v, 4) for k, v in metrics(y, p).items()},
            **bootstrap(y, p),
            "log_loss_by_season": by_season,
            "log_loss_minus_elo_95": paired_log_loss_difference(y, p, preds["Elo"]),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
