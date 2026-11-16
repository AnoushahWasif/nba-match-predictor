"""Day 8: which feature groups earn their place. Writes reports/ablation.json.

    uv run python -m nba_predictor.ablation

The same logistic regression (C = 0.1, day 7's best) is refitted on the training seasons with
each feature set and scored on validation against plain Elo.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_predictor import collect
from nba_predictor.evaluate import metrics, paired_log_loss_difference, split
from nba_predictor.train import FEATURES, matrix

GROUPS = {
    "elo": ["elo_logit"],
    "form": [
        "diff_margin_avg", "diff_win_pct", "diff_form_margin", "diff_off_rtg", "diff_def_rtg",
        "min_games_played",
    ],
    "rest": ["home_rest_days", "away_rest_days", "home_b2b", "away_b2b"],
    "venue": ["no_home"],
}  # fmt: skip


def feature_sets() -> dict[str, list[str]]:
    sets = {
        "Elo only": GROUPS["elo"] + GROUPS["venue"],
        "Elo + rest": GROUPS["elo"] + GROUPS["rest"] + GROUPS["venue"],
        "Elo + form": GROUPS["elo"] + GROUPS["form"] + GROUPS["venue"],
        "form + rest, no Elo": GROUPS["form"] + GROUPS["rest"] + GROUPS["venue"],
        "everything": list(FEATURES),
    }
    for group in ("elo", "form", "rest"):
        sets[f"everything minus {group}"] = [f for f in FEATURES if f not in GROUPS[group]]
    return sets


REST_ADJUST = ["home_b2b", "away_b2b"]


class EloRestAdjustment:
    """Plain Elo's log-odds kept exactly as they are, plus a fitted intercept and a shift for
    each side's back-to-back. Refitting Elo's own slope (the "Elo only" set) made it worse, so
    here only the adjustment is learned."""

    def fit(self, x: pd.DataFrame, y) -> "EloRestAdjustment":
        design = self._design(x)
        offset = x["elo_logit"].to_numpy(float)
        y = np.asarray(y, float)

        def loss(b):
            z = offset + design @ b
            return float(np.mean(np.logaddexp(0, z) - y * z))

        result = minimize(loss, np.zeros(design.shape[1]), method="BFGS")
        if not result.success:
            raise RuntimeError(f"back-to-back adjustment didn't converge: {result.message}")
        self.coef_ = result.x
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        z = x["elo_logit"].to_numpy(float) + self._design(x) @ self.coef_
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])

    @staticmethod
    def _design(x: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), x[REST_ADJUST].to_numpy(float)])


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def main(out: Path = Path("reports/ablation.json")) -> dict:
    table = pd.read_parquet(collect.data_dir() / "processed" / "features.parquet")
    train, validation, _ = split(table)
    x_train, x_val = matrix(train), matrix(validation)
    y_train, y_val = train["home_win"], validation["home_win"].to_numpy()
    elo = validation["elo_prob"].to_numpy()
    rows, probs = {}, {}
    for name, cols in feature_sets().items():
        p = model().fit(x_train[cols], y_train).predict_proba(x_val[cols])[:, 1]
        probs[name] = p
        rows[name] = {
            "features": cols,
            **{k: round(v, 4) for k, v in metrics(y_val, p).items()},
            "minus_plain_elo_95": paired_log_loss_difference(y_val, p, elo),
        }
    adjust = EloRestAdjustment().fit(x_train, y_train)
    p = adjust.predict_proba(x_val)[:, 1]
    rows["plain Elo + back-to-back adjustment"] = {
        "features": ["elo_logit (fixed)", *REST_ADJUST],
        "coefficients": dict(
            zip(["intercept", *REST_ADJUST], np.round(adjust.coef_, 4).tolist(), strict=True)
        ),
        **{k: round(v, 4) for k, v in metrics(y_val, p).items()},
        "minus_plain_elo_95": paired_log_loss_difference(y_val, p, elo),
    }
    report = {
        "split": "validation",
        "plain_elo_log_loss": round(metrics(y_val, elo)["log_loss"], 4),
        "sets": rows,
        # What rest adds inside the same regression, without the comparison against plain Elo.
        "elo_plus_rest_minus_elo_only_95": paired_log_loss_difference(
            y_val, probs["Elo + rest"], probs["Elo only"]
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    r = main()
    print("plain Elo", r["plain_elo_log_loss"])
    for name, row in r["sets"].items():
        print(
            f"{name:28s} {row['log_loss']:.4f}  {row['minus_plain_elo_95']}  {row['accuracy']:.3f}"
        )
