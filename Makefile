.PHONY: install lint format test data games

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
