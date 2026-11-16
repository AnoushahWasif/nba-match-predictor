.PHONY: install lint format test data games elo features baselines train ablation

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

data:
	uv run python -m nba_predictor.collect

games:
	uv run python -m nba_predictor.games

features:
	uv run python -m nba_predictor.features

elo:
	uv run python -m nba_predictor.elo

baselines:
	uv run python -m nba_predictor.baselines

train:
	uv run python -m nba_predictor.train

ablation:
	uv run python -m nba_predictor.ablation
