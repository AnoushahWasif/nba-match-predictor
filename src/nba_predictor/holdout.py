"""Day 10: the test seasons (2022-23 to 2024-25), scored once (decision 9).

    uv run nba holdout

Scores what decision 12 declared before the test set was opened: plain Elo, which the CLI uses,
and the back-to-back adjustment as the one secondary model, plus the always-home baseline. It
also checks day 9's two upset hypotheses. A second run is refused while reports/holdout.json
exists, because tuning after seeing these numbers would make them one more validation set.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor import collect
from nba_predictor.ablation import EloRestAdjustment
from nba_predictor.evaluate import bootstrap, metrics, paired_log_loss_difference, per_season, split
from nba_predictor.train import matrix


def favourite_check(table: pd.DataFrame, mask: np.ndarray) -> dict:
    """How often Elo favourites of 70% or more won, against what Elo said, within `mask`."""
    fav = np.maximum(table["elo_prob"], 1 - table["elo_prob"]).to_numpy()
    won = np.where(table["elo_prob"] >= 0.5, table["home_win"], 1 - table["home_win"])
    keep = (fav >= 0.7) & mask
    return {
        "games": int(keep.sum()),
        "predicted": round(float(fav[keep].mean()), 4) if keep.any() else None,
        "won": round(float(won[keep].mean()), 4) if keep.any() else None,
    }


def run_metadata() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = collect.read_manifest(collect.data_dir() / "raw")
    return {"git_commit": commit, "raw_sha256": {s: e["sha256"] for s, e in manifest.items()}}


def main(out: Path = Path("reports/holdout.json"), table: pd.DataFrame | None = None) -> dict:
    if out.exists():
        raise RuntimeError(
            f"{out} exists, so the test seasons have been scored already. Delete it only to "
            "score a model chosen without looking at these results."
        )
    if table is None:
        table = pd.read_parquet(collect.data_dir() / "processed" / "features.parquet")
    train, _, test = split(table)
    if test.empty:
        raise RuntimeError("no test-season games in the features table")
    y = test["home_win"].to_numpy()
    elo = test["elo_prob"].to_numpy()
    adjusted = EloRestAdjustment().fit(matrix(train), train["home_win"]).predict_proba(matrix(test))
    adjusted = adjusted[:, 1]
    home = np.full(len(test), train["home_win"].mean())

    def scored(p):
        return {**{k: round(v, 4) for k, v in metrics(y, p).items()}, **bootstrap(y, p)}

    games_played = test[["home_games_played", "away_games_played"]].min(axis=1).to_numpy()
    fav_home = (test["elo_prob"] >= 0.5).to_numpy()
    fav_b2b = np.where(fav_home, test["home_b2b"], test["away_b2b"])
    dog_b2b = np.where(fav_home, test["away_b2b"], test["home_b2b"])
    report = {
        "split": "test",
        "seasons": sorted(test["season"].unique()),
        "games": len(test),
        "plain Elo (CLI)": scored(elo),
        "Elo + back-to-back adjustment (declared secondary)": scored(adjusted),
        "always home": scored(home),
        "adjustment_minus_elo_95": paired_log_loss_difference(y, adjusted, elo),
        "elo_by_season": per_season(test.assign(p=elo), "p", n=2000).round(4).to_dict("index"),
        "day_9_hypotheses": {
            "favourites in either team's first 10 games": favourite_check(test, games_played < 10),
            "favourites past 10 games": favourite_check(test, games_played >= 10),
            "favourite on a back-to-back, underdog rested": favourite_check(
                test, (fav_b2b == 1) & (dog_b2b == 0)
            ),
        },
        "run": run_metadata(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.part")
    tmp.write_text(json.dumps(report, indent=2, default=float))
    tmp.replace(out)
    return report
