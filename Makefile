.PHONY: install lint format test verify clean

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run pytest tests/ -v

verify:
	uv run python scripts/verify_setup.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache htmlcov .coverage