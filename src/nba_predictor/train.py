"""Day 7: logistic regression and gradient boosting against Elo on validation.

    uv run python -m nba_predictor.train

Every model is fitted on the training seasons and compared by validation log loss. The best one
is saved to models/best-ml with a SHA-256 manifest; reports/models.json has every result.
Whether the CLI uses it or plain Elo is decided after the day 8 ablation.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_predictor import collect
from nba_predictor.evaluate import bootstrap, metrics, paired_log_loss_difference, split
from nba_predictor.model_store import save_model

FEATURES = [
    "elo_logit",
    "diff_margin_avg",
    "diff_win_pct",
    "diff_form_margin",
    "diff_off_rtg",
    "diff_def_rtg",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "min_games_played",
    "no_home",
]


def matrix(table: pd.DataFrame) -> pd.DataFrame:
    """The model's inputs. Elo enters as log-odds, which is linear in the rating difference."""
    p = table["elo_prob"].clip(1e-6, 1 - 1e-6)
    return pd.DataFrame(
        {
            "elo_logit": np.log(p / (1 - p)),
            **{c: table[c] for c in FEATURES[1:10]},
            "min_games_played": table[["home_games_played", "away_games_played"]].min(axis=1),
            "no_home": (table["neutral"] | table["bubble"]).astype(int),
        },
        index=table.index,
    )[FEATURES]


def candidates() -> dict:
    models = {
        f"logistic C={c}": make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        for c in (0.01, 0.1, 1.0, 10.0)
    }
    for depth in (2, 3):
        for iters in (100, 300):
            models[f"boosting depth={depth} iters={iters}"] = HistGradientBoostingClassifier(
                learning_rate=0.05, max_depth=depth, max_iter=iters, min_samples_leaf=50,
                random_state=0,
            )  # fmt: skip
    return models


def main(out: Path = Path("reports/models.json"), model_root: Path = Path("models")) -> dict:
    table = pd.read_parquet(collect.data_dir() / "processed" / "features.parquet")
    train, validation, _ = split(table)
    x_train, y_train = matrix(train), train["home_win"].to_numpy()
    x_val, y_val = matrix(validation), validation["home_win"].to_numpy()
    elo = validation["elo_prob"].to_numpy()

    results, probs, fitted = {}, {}, {}
    for name, model in candidates().items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        p = model.predict_proba(x_val)[:, 1]
        results[name] = {
            **{k: round(v, 4) for k, v in metrics(y_val, p).items()},
            "fit_seconds": round(time.perf_counter() - start, 2),
        }
        probs[name], fitted[name] = p, model
    best = min(results, key=lambda n: results[n]["log_loss"])
    report = {
        "split": "validation",
        "features": FEATURES,
        "results": results,
        "elo": {k: round(v, 4) for k, v in metrics(y_val, elo).items()},
        "best": best,
        "best_95": bootstrap(y_val, probs[best]),
        "best_minus_elo_95": paired_log_loss_difference(y_val, probs[best], elo),
    }
    for family in ("logistic", "boosting"):
        top = min(
            (n for n in results if n.startswith(family)), key=lambda n: results[n]["log_loss"]
        )
        report[f"best_{family}"] = top
        report[f"best_{family}_minus_elo_95"] = paired_log_loss_difference(y_val, probs[top], elo)
    # Model first, so the report never names a hash that wasn't saved.
    report["model_sha256"] = save_model(
        fitted[best], model_root / "best-ml", {"name": best, "features": FEATURES}
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
