"""nba predict | holdout.

    nba predict --home BOS --away NYK --date 2025-03-01    home win probability from plain Elo
    nba holdout                                            score the test seasons, once

Predictions use only the cached games (`make data games`) before the given date. Nothing here
calls the NBA's servers.
"""

import argparse
import json
import sys
from datetime import date as date_type
from pathlib import Path

import pandas as pd

from nba_predictor import collect, elo

ELO_REPORT = Path("reports/elo.json")
HOLDOUT_REPORT = Path("reports/holdout.json")


class UsageError(ValueError):
    pass


def season_of(day: pd.Timestamp) -> str:
    """The season a date belongs to: seasons start in October, so August onwards is the next."""
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def load_games() -> pd.DataFrame:
    path = collect.data_dir() / "processed" / "games.parquet"
    if not path.exists():
        raise UsageError(f"{path} is missing; run `make data games` first")
    return pd.read_parquet(path)


def load_params(path: Path | None = None) -> elo.EloParams:
    path = path or ELO_REPORT  # looked up at call time, so a changed setting is honoured
    try:
        return elo.EloParams(**json.loads(path.read_text())["params"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UsageError(f"can't read Elo settings from {path}; run `make elo`") from exc


def team_ids(games: pd.DataFrame) -> dict[str, int]:
    """Abbreviation to TEAM_ID, from the most recent season each abbreviation appears in."""
    sides = [
        games[["date", side, f"{side}_id"]].set_axis(["date", "abbr", "id"], axis=1)
        for side in ("home", "away")
    ]
    latest = pd.concat(sides).sort_values("date").drop_duplicates("abbr", keep="last")
    return dict(zip(latest["abbr"], latest["id"].astype(int), strict=True))


def parse_date(text: str | None) -> pd.Timestamp:
    if text is None:
        return pd.Timestamp(date_type.today())
    try:
        return pd.Timestamp(date_type.fromisoformat(text))
    except ValueError as exc:
        raise UsageError(f"--date must look like 2025-03-01, not {text!r}") from exc


def predict(
    games: pd.DataFrame, params: elo.EloParams, home: str, away: str, day: pd.Timestamp,
    neutral: bool = False,
) -> dict:  # fmt: skip
    ids = team_ids(games)
    home, away = home.upper(), away.upper()
    for abbr in (home, away):
        if abbr not in ids:
            raise UsageError(f"unknown team {abbr!r}; known: {', '.join(sorted(ids))}")
    if home == away:
        raise UsageError("a team can't play itself")
    if day <= games["date"].min():
        raise UsageError(f"no games before {day:%Y-%m-%d} in the cache")
    rating = elo.current(games, day, season_of(day), params)
    h, a = rating[ids[home]], rating[ids[away]]
    prob = elo.expected(h + (0.0 if neutral else params.home) - a)
    yesterday = day - pd.Timedelta(days=1)
    played = games[games["date"] == yesterday]
    on_b2b = [
        abbr
        for abbr in (home, away)
        if ((played["home_id"] == ids[abbr]) | (played["away_id"] == ids[abbr])).any()
    ]
    return {
        "home": home,
        "away": away,
        "date": f"{day:%Y-%m-%d}",
        "home_win_probability": round(prob, 4),
        "home_elo": round(h, 1),
        "away_elo": round(a, 1),
        "neutral": neutral,
        "back_to_back": on_b2b,
        "games_up_to": f"{games.loc[games['date'] < day, 'date'].max():%Y-%m-%d}",
    }


def describe(result: dict) -> str:
    venue = "neutral court" if result["neutral"] else f"at {result['home']}"
    lines = [
        f"{result['away']} at {result['home']} on {result['date']} ({venue})",
        f"  {result['home']} win probability: {result['home_win_probability']:.0%}"
        f"  (Elo {result['home_elo']:.0f} vs {result['away_elo']:.0f})",
        f"  ratings use games up to {result['games_up_to']}",
    ]
    if result["back_to_back"]:
        lines.append(
            f"  note: {', '.join(result['back_to_back'])} played the day before. Plain Elo "
            "ignores this; on validation, tired teams did a few points worse than Elo said."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nba", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("predict", help="home win probability for one game")
    pr.add_argument("--home", required=True, help="home team abbreviation, e.g. BOS")
    pr.add_argument("--away", required=True, help="away team abbreviation, e.g. NYK")
    pr.add_argument("--date", help="game date, YYYY-MM-DD (default: today)")
    pr.add_argument("--neutral", action="store_true", help="no home advantage (neutral court)")
    pr.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub.add_parser("holdout", help="score the test seasons, once")
    args = parser.parse_args(argv)
    try:
        if args.command == "holdout":
            from nba_predictor.holdout import main as holdout_main

            report = holdout_main(HOLDOUT_REPORT)
            elo_ll = report["plain Elo (CLI)"]["log_loss"]
            print(f"test seasons: plain Elo log loss {elo_ll:.4f}; wrote {HOLDOUT_REPORT}")
            return 0
        result = predict(
            load_games(), load_params(), args.home, args.away, parse_date(args.date), args.neutral
        )
        print(json.dumps(result, indent=2) if args.json else describe(result))
        return 0
    except (UsageError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
