"""Day 5: an Elo rating written from scratch (decision 5).

Each team starts at 1500. Before a game the home team's win probability is
    1 / (1 + 10 ** (-(home + bonus - away) / 400))
where `bonus` is the home advantage in rating points (0 on a neutral court or in the bubble).
After the game both ratings move by K × (result − expected), optionally scaled by the margin of
victory. Between seasons every rating is pulled part of the way back to 1500.

    uv run python -m nba_predictor.elo     # tunes K, bonus and carry-over on the training seasons
"""

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor import collect
from nba_predictor.evaluate import log_loss

MEAN = 1500.0
TRAIN_LAST = "2019-20"
BURN_IN = "2014-15"


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home: float = 75.0
    carry: float = 0.75  # share of a team's distance from 1500 kept into the next season
    mov: bool = True  # scale updates by the margin of victory


def expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def mov_multiplier(margin: float, winner_diff: float) -> float:
    """Bigger wins move ratings more, damped when the favourite wins (FiveThirtyEight's form),
    so a strong team doesn't gain endlessly from blowouts of weak ones."""
    return ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)


def ratings(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Pre-game ratings and home win probability for every game, indexed like `games`.

    Games are processed in date order. A team plays at most once a day (features.team_games
    checks), so within a date the order doesn't matter: no game's result can reach another
    game on the same date."""
    pre, _, _ = _run(games, params)
    return pre.loc[games.index]


def current(
    games: pd.DataFrame, before: pd.Timestamp, season: str, params: EloParams = EloParams()
) -> dict[int, float]:
    """Every team's rating going into a game on `before` in `season`: all earlier games played,
    and the between-season pull applied if the team hasn't played yet in that season."""
    _, rating, season_of = _run(games[games["date"] < before], params)
    return {
        team: r if season_of[team] == season else MEAN + params.carry * (r - MEAN)
        for team, r in rating.items()
    }


def _run(games: pd.DataFrame, params: EloParams):
    g = games.sort_values(["date", "game_id"])
    home_ids = g["home_id"].to_numpy(int)
    away_ids = g["away_id"].to_numpy(int)
    seasons = g["season"].to_numpy()
    margins = (g["home_pts"] - g["away_pts"]).to_numpy(float)
    no_bonus = np.zeros(len(g), bool)
    for col in ("neutral", "bubble"):
        if col in g:
            no_bonus |= g[col].to_numpy(bool)

    rating: dict[int, float] = {}
    season_of: dict[int, str] = {}
    home_pre, away_pre, prob = (np.empty(len(g)) for _ in range(3))
    for i in range(len(g)):
        h, a, season = home_ids[i], away_ids[i], seasons[i]
        for team in (h, a):
            if team not in rating:
                rating[team] = MEAN
            elif season_of[team] != season:
                rating[team] = MEAN + params.carry * (rating[team] - MEAN)
            season_of[team] = season
        diff = rating[h] + (0.0 if no_bonus[i] else params.home) - rating[a]
        p = expected(diff)
        home_pre[i], away_pre[i], prob[i] = rating[h], rating[a], p
        margin = margins[i]
        step = params.k * ((1.0 if margin > 0 else 0.0) - p)
        if params.mov:
            step *= mov_multiplier(margin, diff if margin > 0 else -diff)
        rating[h] += step
        rating[a] -= step
    pre = pd.DataFrame(
        {"home_elo": home_pre, "away_elo": away_pre, "elo_prob": prob}, index=g.index
    )
    return pre, rating, season_of


GRID = {
    "k": (10, 15, 20, 25, 30),
    "home": (40, 60, 80, 100),
    "carry": (0.5, 0.65, 0.8, 1.0),
    "mov": (False, True),
}


def tune(games: pd.DataFrame, grid: dict = GRID) -> tuple[EloParams, pd.DataFrame]:
    """Grid search on the training seasons, scoring log loss after the burn-in season.
    Validation and test seasons are not passed in, so they can't influence the choice."""
    train = games[games["season"] <= TRAIN_LAST]
    scored = (train["season"] != BURN_IN).to_numpy()
    rows = []
    for k, home, carry, mov in itertools.product(*grid.values()):
        params = EloParams(k, home, carry, mov)
        p = ratings(train, params)["elo_prob"].to_numpy()
        y = train["home_win"].to_numpy()[scored]
        rows.append({**asdict(params), "log_loss": log_loss(y, p[scored])})
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = table.iloc[0]
    params = EloParams(
        float(best["k"]), float(best["home"]), float(best["carry"]), bool(best["mov"])
    )
    return params, table


def main(out: Path = Path("reports/elo.json")) -> dict:
    games = pd.read_parquet(collect.data_dir() / "processed" / "games.parquet")
    best, table = tune(games)
    report = {
        "params": asdict(best),
        "train_log_loss": round(float(table.iloc[0]["log_loss"]), 5),
        "grid_size": len(table),
        "top_5": table.head(5).round(5).to_dict("records"),
        "best_on_grid_edge": {
            name: getattr(best, name) in (min(vals), max(vals)) and len(vals) > 2
            for name, vals in GRID.items()
            if name != "mov"
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
