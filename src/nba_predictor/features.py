"""Day 4: per-team form features from strictly earlier games (decision 4).

Every feature for a game is computed from that team's games on earlier dates in the same season.
Each team plays at most once a day, so "earlier rows in date order" and "earlier dates" are the
same thing here; `team_games` checks that rather than assuming it.

    uv run python -m nba_predictor.features
"""

from pathlib import Path

import pandas as pd

from nba_predictor import collect

WINDOW = 10
TEAM_FEATURES = ["games_played", "win_pct", "margin_avg", "form_margin", "off_rtg", "def_rtg"]


def possessions(fga, oreb, tov, fta):
    """The usual box-score estimate of possessions."""
    return fga - oreb + tov + 0.44 * fta


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one from each team's side, in date order within each team."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        t = pd.DataFrame(
            {
                "game_id": games["game_id"],
                "season": games["season"],
                "date": games["date"],
                "team_id": games[f"{side}_id"],
                "is_home": side == "home",
                "pts_for": games[f"{side}_pts"],
                "pts_against": games[f"{other}_pts"],
                "poss_for": possessions(
                    games[f"{side}_fga"],
                    games[f"{side}_oreb"],
                    games[f"{side}_tov"],
                    games[f"{side}_fta"],
                ),
                "poss_against": possessions(
                    games[f"{other}_fga"],
                    games[f"{other}_oreb"],
                    games[f"{other}_tov"],
                    games[f"{other}_fta"],
                ),
            }
        )
        t["win"] = (t["pts_for"] > t["pts_against"]).astype(int)
        sides.append(t)
    long = pd.concat(sides, ignore_index=True).sort_values(["team_id", "date", "game_id"])
    if long.duplicated(["team_id", "date"]).any():
        raise ValueError("a team plays twice on one date; earlier-date features would leak")
    return long.reset_index(drop=True)


def add_form(long: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Features known before each game. The current game is excluded by subtracting it from
    running totals (for season-to-date) or shifting by one (for the rolling window)."""
    out = long.copy()
    out["margin"] = out["pts_for"] - out["pts_against"]
    g = out.groupby(["team_id", "season"], sort=False)
    played = g.cumcount()
    out["games_played"] = played
    out["win_pct"] = (g["win"].cumsum() - out["win"]) / played
    out["margin_avg"] = (g["margin"].cumsum() - out["margin"]) / played
    out["form_margin"] = g["margin"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    prior = {
        c: g[c].cumsum() - out[c] for c in ["pts_for", "pts_against", "poss_for", "poss_against"]
    }
    out["off_rtg"] = 100 * prior["pts_for"] / prior["poss_for"]
    out["def_rtg"] = 100 * prior["pts_against"] / prior["poss_against"]
    # Before a team's first game of the season there is no evidence: neutral values, with
    # games_played = 0 telling the model they're placeholders. The rating placeholder is the
    # league average over earlier dates only; an average over the whole table would leak.
    daily = out.groupby("date")[["pts_for", "poss_for"]].sum().sort_index().cumsum().shift(1)
    league = (100 * daily["pts_for"] / daily["poss_for"]).fillna(100.0)
    league_rtg = out["date"].map(league)
    for col in ("off_rtg", "def_rtg"):
        out[col] = out[col].fillna(league_rtg)
    return out.fillna({"win_pct": 0.5, "margin_avg": 0.0, "form_margin": 0.0})


def game_features(games: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """The games table with each side's pre-game features and their home-minus-away differences."""
    form = add_form(team_games(games), window).set_index(["game_id", "team_id"])[TEAM_FEATURES]
    out = games.copy()
    for side in ("home", "away"):
        keys = pd.MultiIndex.from_arrays([out["game_id"], out[f"{side}_id"]])
        side_form = form.reindex(keys)
        for col in TEAM_FEATURES:
            out[f"{side}_{col}"] = side_form[col].to_numpy()
    for col in TEAM_FEATURES:
        out[f"diff_{col}"] = out[f"home_{col}"] - out[f"away_{col}"]
    return out


def build(processed: Path | None = None) -> pd.DataFrame:
    processed = processed or collect.data_dir() / "processed"
    out = game_features(pd.read_parquet(processed / "games.parquet"))
    out.to_parquet(processed / "features.parquet", index=False)
    return out


if __name__ == "__main__":
    table = build()
    print(f"{len(table):,} games, {sum(c.startswith('diff_') for c in table.columns)} differences")
