# nba-match-predictor

The probability that the home team wins an NBA regular-season game, using only what was known before tip-off.

The point is to do time-series features without leaking the future. Every rolling stat and rating is built from earlier games only, and tests prove it. Models are judged by season on log loss and calibration, not just accuracy.

**Status:** in progress (day 8 of 10). Results will go here once there are any.

## Planned approach

1. Collect team game logs for the 2014-15 to 2024-25 regular seasons through `nba_api`, cached and rate-limited
2. One row per game; how big home advantage is, including the 2020 bubble
3. Rolling form, rest days, back-to-backs and an Elo rating, all from past games only, with leakage tests
4. Train on early seasons, tune on the next two, score the last three once
5. Logistic regression and gradient boosting against Elo, with calibration and feature ablation

## Stack

Python 3.12, uv, pandas, scikit-learn, matplotlib, pytest, ruff. CI: GitHub Actions with ruff, pytest, pip-audit and Gitleaks.

## Data

Team game logs from stats.nba.com through [nba_api](https://github.com/swar/nba_api) (MIT), an unofficial client. The data is subject to NBA.com's terms of use: it's downloaded by a script, used for this non-commercial project, and never committed or re-hosted.

## Quick start

```bash
uv sync
make data      # downloads 11 seasons once (under a minute), cached in data/raw
make games     # one row per game: data/processed/games.parquet
make elo       # tunes Elo on the training seasons: reports/elo.json
make features  # pre-game features from earlier games only: data/processed/features.parquet
make lint test
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
